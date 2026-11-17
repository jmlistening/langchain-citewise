"""Load supported documents and split them into stable, addressable chunks."""

from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CiteWiseError, UsageError

TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
PDF_SUFFIXES = frozenset({".pdf"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")

WarnFn = Callable[[str], None]


def _stderr_warn(message: str) -> None:
    print(f"citewise: warning: {message}", file=sys.stderr)


def iter_source_files(source: Path) -> list[Path]:
    """Return the supported files under ``source`` in deterministic order.

    ``source`` may be a single supported file or a directory that is searched
    recursively for ``.md``, ``.markdown``, ``.txt``, and ``.pdf`` files.
    """

    source = Path(source).expanduser()
    if not source.exists():
        raise UsageError(f"source not found: {source}")
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise UsageError(f"unsupported file type '{source.suffix}'; supported: {supported}")
        return [source]
    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    raise UsageError(f"source is not a regular file or directory: {source}")


def derive_title(text: str, fallback: str) -> str:
    """Use the first markdown H1 when present, otherwise a humanized fallback."""

    match = _H1_RE.search(text)
    if match:
        title = match.group(1).strip()
        if title:
            return title
    return fallback.replace("-", " ").replace("_", " ").strip() or fallback


def _fallback_title(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").strip() or path.stem


def _strip_title_heading(text: str) -> str:
    """Drop the first H1 line from stored content; it lives on as metadata."""

    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if _H1_RE.match(line):
            del lines[index]
            break
    return "".join(lines)


def _read_text_file(path: Path, rel_source: str, warn: WarnFn) -> list[Document]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warn(f"skipping unreadable file {rel_source}: {exc}")
        return []
    if not text.strip():
        warn(f"skipping empty file: {rel_source}")
        return []
    title = derive_title(text, _fallback_title(path))
    body = _strip_title_heading(text)
    if not body.strip():
        warn(f"skipping file with no body text: {rel_source}")
        return []
    return [Document(page_content=body, metadata={"source": rel_source, "title": title})]


def _load_pdf(path: Path, rel_source: str, warn: WarnFn) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise CiteWiseError(
            "PDF support requires the 'pdf' extra: run `uv sync --extra pdf` or install pypdf.",
            exit_code=2,
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf raises several exception types
        warn(f"skipping unreadable PDF {rel_source}: {exc}")
        return []

    fallback = _fallback_title(path)
    page_texts: list[str] = []
    try:
        for page in reader.pages:
            page_texts.append(page.extract_text() or "")
    except Exception as exc:
        warn(f"skipping unreadable PDF {rel_source}: {exc}")
        return []

    title = fallback
    for text in page_texts:
        if text.strip():
            title = derive_title(text, fallback)
            break

    documents: list[Document] = []
    for page_number, text in enumerate(page_texts, start=1):
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": rel_source, "title": title, "page": page_number},
            )
        )
    if not documents:
        warn(f"skipping PDF with no extractable text: {rel_source}")
    return documents


def load_documents(source: Path, *, warn: WarnFn | None = None) -> list[Document]:
    """Load all supported documents beneath ``source``.

    Stored source paths are POSIX-style paths relative to the ingest root.
    Empty files are skipped with a warning. PDFs are loaded lazily so that the
    core package does not depend on pypdf.
    """

    warning = warn or _stderr_warn
    source_path = Path(source).expanduser()
    files = iter_source_files(source_path)
    root = source_path if source_path.is_dir() else source_path.parent

    documents: list[Document] = []
    for path in files:
        rel_source = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix in PDF_SUFFIXES:
            documents.extend(_load_pdf(path, rel_source, warning))
        else:
            documents.extend(_read_text_file(path, rel_source, warning))
    return documents


def build_chunk_id(source: str, chunk_index: int, content: str) -> str:
    """Stable chunk id: sha256(source::chunk_index::content) truncated to 16 hex chars."""

    payload = f"{source}::{chunk_index}::{content}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split documents into de-duplicated chunks with stable ids and relative sources."""

    if chunk_size <= 0:
        raise UsageError("--chunk-size must be greater than zero")
    if chunk_overlap < 0:
        raise UsageError("--chunk-overlap must be zero or greater")
    if chunk_overlap >= chunk_size:
        raise UsageError("--chunk-overlap must be smaller than --chunk-size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks: list[Document] = []
    indices: dict[str, int] = {}
    for document in documents:
        source = str(document.metadata.get("source", "")).replace("\\", "/")
        if not source:
            raise UsageError("document is missing its relative 'source' metadata")
        for part in splitter.split_documents([document]):
            content = _WHITESPACE_RE.sub(" ", part.page_content).strip()
            if not content:
                continue
            chunk_index = indices.get(source, 0)
            indices[source] = chunk_index + 1
            chunk_id = build_chunk_id(source, chunk_index, content)
            metadata = dict(part.metadata)
            metadata["source"] = source
            metadata["chunk_index"] = chunk_index
            metadata["chunk_id"] = chunk_id
            chunks.append(Document(id=chunk_id, page_content=content, metadata=metadata))
    return chunks
