"""Build, persist, validate, and load the local vector index."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .config import (
    HASHING_DIM,
    UsageError,
    get_api_key,
    get_base_url,
    get_embedding_model,
)
from .embeddings import HashingEmbeddings

MANIFEST_VERSION = 1
INDEX_FILENAME = "index.json"
MANIFEST_FILENAME = "manifest.json"
EMBEDDING_PROVIDERS = ("hashing", "openai")


@dataclass
class IndexContents:
    """A loaded index and the metadata describing how it was built."""

    index_dir: Path
    manifest: dict[str, Any]
    vector_store: InMemoryVectorStore
    documents: list[Document]


def make_embeddings(provider: str, model: str | None = None) -> Embeddings:
    """Construct an embedding backend from its manifest description."""

    if provider == "hashing":
        if model:
            raise UsageError("--embedding-model only applies to the 'openai' embedding provider")
        return HashingEmbeddings(dim=HASHING_DIM)
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise UsageError(
                "OpenAI embeddings require the 'openai' extra: run "
                "`uv sync --extra openai`, or ingest with --embeddings hashing."
            ) from exc
        api_key = get_api_key()
        if api_key is None:
            raise UsageError(
                "OpenAI embeddings require CITEWISE_API_KEY or OPENAI_API_KEY, "
                "or ingest with --embeddings hashing."
            )
        kwargs: dict[str, Any] = {
            "model": model or get_embedding_model(),
            "api_key": api_key,
        }
        base_url = get_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIEmbeddings(**kwargs)
    raise UsageError(
        f"unknown embedding provider {provider!r}; choose one of: " + ", ".join(EMBEDDING_PROVIDERS)
    )


def _embedding_dim(embedding: Embeddings, vector_store: InMemoryVectorStore) -> int:
    """Read the embedding dimension from the stored vectors."""

    if isinstance(embedding, HashingEmbeddings):
        return embedding.dim
    for entry in vector_store.store.values():
        vector = entry.get("vector") if isinstance(entry, dict) else None
        if vector:
            return len(vector)
    raise UsageError(
        "could not determine the embedding dimension from the index; "
        "the embedding provider returned no vectors"
    )


def build_index(
    index_dir: Path,
    chunks: list[Document],
    *,
    provider: str = "hashing",
    model: str | None = None,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    """Embed chunks, dump the vector store, and write a manifest.

    Source paths in the manifest are relative POSIX paths supplied by the
    loader; absolute paths are never written.
    """

    if not chunks:
        raise UsageError(
            "no chunks to index; provide a file or directory with non-empty "
            ".md, .markdown, .txt, or .pdf files"
        )

    sources: set[str] = set()
    for chunk in chunks:
        raw_source = chunk.metadata.get("source")
        if not raw_source:
            raise UsageError("chunks are missing relative 'source' metadata")
        source = str(raw_source).replace("\\", "/")
        if source.startswith("/") or re.match(r"^[A-Za-z]:", source):
            raise UsageError(
                f"chunk source must be a relative path, got {source!r}; "
                "ingest through the loader so absolute paths are never persisted"
            )
        sources.add(source)
    if not sources:
        raise UsageError("chunks are missing relative 'source' metadata")

    if provider != "openai" and model:
        raise UsageError("--embedding-model only applies to the 'openai' embedding provider")

    resolved_model = (model or get_embedding_model()) if provider == "openai" else None
    embedding = make_embeddings(provider, resolved_model)
    vector_store = InMemoryVectorStore(embedding=embedding)
    vector_store.add_documents(chunks)

    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedding": {
            "provider": provider,
            "dim": _embedding_dim(embedding, vector_store),
            "model": resolved_model,
        },
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "file_count": len(sources),
        "chunk_count": len(chunks),
        "sources": sorted(sources),
    }

    index_path = Path(index_dir)
    try:
        index_path.mkdir(parents=True, exist_ok=True)
        vector_store.dump(str(index_path / INDEX_FILENAME))
        manifest_path = index_path / MANIFEST_FILENAME
        temporary_path = manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary_path.replace(manifest_path)
    except OSError as exc:
        raise UsageError(
            f"could not write the index at {index_path}: {exc}. "
            "Choose a writable directory for --index."
        ) from exc
    return manifest


def _read_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / MANIFEST_FILENAME
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UsageError(
            f"could not read index manifest at {manifest_path}: {exc}. "
            "Run `citewise ingest <source>` to rebuild the index."
        ) from exc
    except UnicodeDecodeError as exc:
        raise UsageError(
            f"index manifest at {manifest_path} is not valid UTF-8: {exc}. "
            "Run `citewise ingest <source>` to rebuild the index."
        ) from exc
    except json.JSONDecodeError as exc:
        raise UsageError(
            f"index manifest at {manifest_path} is corrupt: {exc}. "
            "Run `citewise ingest <source>` to rebuild the index."
        ) from exc
    if not isinstance(data, dict):
        raise UsageError(
            f"index manifest at {manifest_path} is invalid. "
            "Run `citewise ingest <source>` to rebuild the index."
        )
    return data


def load_index(
    index_dir: Path,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> IndexContents:
    """Load a persisted index, validating its manifest and embedding settings.

    ``provider``/``model`` let a caller request a specific embedding
    configuration; a mismatch fails with a re-ingest hint instead of silently
    mixing incompatible vectors.
    """

    directory = Path(index_dir)
    index_path = directory / INDEX_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    if not directory.is_dir() or not index_path.is_file() or not manifest_path.is_file():
        raise UsageError(f"No index found at {directory}. Run `citewise ingest <source>` first.")

    manifest = _read_manifest(directory)
    if manifest.get("version") != MANIFEST_VERSION:
        raise UsageError(
            f"index at {directory} has unsupported version "
            f"{manifest.get('version')!r} (expected {MANIFEST_VERSION}). "
            "Run `citewise ingest <source>` to rebuild the index."
        )

    embedding_info = manifest.get("embedding")
    if not isinstance(embedding_info, dict):
        raise UsageError(
            f"index manifest at {manifest_path} has no embedding description. "
            "Run `citewise ingest <source>` to rebuild the index."
        )
    stored_provider = str(embedding_info.get("provider", ""))
    stored_dim = embedding_info.get("dim")
    stored_model = embedding_info.get("model")

    if provider is not None and provider != stored_provider:
        raise UsageError(
            f"index embedding mismatch: index uses provider '{stored_provider}' "
            f"(dim {stored_dim}, model {stored_model}), but '{provider}' was requested. "
            "Re-run `citewise ingest` with the desired embedding configuration."
        )
    if model is not None and stored_provider == "openai" and model != stored_model:
        raise UsageError(
            f"index embedding mismatch: index uses model '{stored_model}', but "
            f"'{model}' was requested. Re-run `citewise ingest` with the desired "
            "embedding configuration."
        )
    if model is not None and stored_provider != "openai":
        raise UsageError(
            f"index embedding mismatch: index uses provider '{stored_provider}', "
            f"which has no embedding model, but model '{model}' was requested. "
            "Re-run `citewise ingest` with the desired embedding configuration."
        )

    embedding = make_embeddings(stored_provider, stored_model)
    if isinstance(embedding, HashingEmbeddings) and stored_dim != embedding.dim:
        raise UsageError(
            f"index embedding mismatch: manifest says dim {stored_dim}, but the "
            f"hashing embedder produces dim {embedding.dim}. "
            "Run `citewise ingest <source>` to rebuild the index."
        )

    try:
        vector_store = InMemoryVectorStore.load(str(index_path), embedding=embedding)
    except Exception as exc:
        raise UsageError(
            f"could not load index at {index_path}: {exc}. "
            "Run `citewise ingest <source>` to rebuild the index."
        ) from exc

    documents = [
        Document(
            id=str(key),
            page_content=str(entry.get("text", "")),
            metadata=dict(entry.get("metadata") or {}),
        )
        for key, entry in vector_store.store.items()
    ]
    expected_chunks = manifest.get("chunk_count")
    if isinstance(expected_chunks, int) and expected_chunks != len(documents):
        raise UsageError(
            f"index is inconsistent: manifest lists {expected_chunks} chunks but "
            f"{len(documents)} were loaded. Run `citewise ingest <source>` to rebuild it."
        )

    return IndexContents(
        index_dir=directory,
        manifest=manifest,
        vector_store=vector_store,
        documents=documents,
    )
