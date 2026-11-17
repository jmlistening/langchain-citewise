from __future__ import annotations

from pathlib import Path

import pytest

from citewise.config import UsageError
from citewise.corpus import (
    build_chunk_id,
    chunk_documents,
    derive_title,
    iter_source_files,
    load_documents,
)


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_documents_uses_relative_posix_paths_and_titles(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    _write_markdown(
        corpus / "sub" / "notes.md",
        "# Notes Title\n\nSome useful notes about the instrument.\n",
    )
    _write_markdown(corpus / "plain.txt", "Plain text without a markdown heading.\n")

    documents = load_documents(corpus)
    by_source = {str(document.metadata["source"]): document for document in documents}

    assert set(by_source) == {"sub/notes.md", "plain.txt"}
    assert all(not Path(source).is_absolute() for source in by_source)
    assert "\\" not in "".join(by_source)

    notes = by_source["sub/notes.md"]
    assert notes.metadata["title"] == "Notes Title"
    assert "Notes Title" not in notes.page_content
    assert "instrument" in notes.page_content

    plain = by_source["plain.txt"]
    assert plain.metadata["title"] == "plain"


def test_load_single_file_uses_filename_as_relative_source(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "field-notes.md"
    _write_markdown(path, "Field observations of the night sky.\n")

    documents = load_documents(path)
    assert len(documents) == 1
    assert documents[0].metadata["source"] == "field-notes.md"
    assert documents[0].metadata["title"] == "field notes"


def test_empty_files_are_skipped_with_warning(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "empty.md").write_text("   \n\n", encoding="utf-8")

    warnings: list[str] = []
    documents = load_documents(corpus, warn=warnings.append)

    assert documents == []
    assert any("empty file" in warning for warning in warnings)


def test_heading_only_file_is_skipped(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "title.md").write_text("# Just A Title\n", encoding="utf-8")

    warnings: list[str] = []
    assert load_documents(corpus, warn=warnings.append) == []
    assert any("no body text" in warning for warning in warnings)


def test_iter_source_files_rejects_unsupported_and_missing(tmp_path: Path) -> None:
    unsupported = tmp_path / "data.json"
    unsupported.write_text("{}", encoding="utf-8")
    with pytest.raises(UsageError):
        iter_source_files(unsupported)

    with pytest.raises(UsageError):
        iter_source_files(tmp_path / "does-not-exist")


def test_derive_title_prefers_first_h1() -> None:
    assert derive_title("# Real Title\n\nBody", "fallback") == "Real Title"
    assert derive_title("No heading here", "fallback title") == "fallback title"
    assert derive_title("No heading here", "field-notes") == "field notes"


def test_chunk_ids_are_stable_unique_and_relative(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    body = "Sentence one describes the mirror. Sentence two describes the dome. " * 12
    _write_markdown(corpus / "telescope.md", f"# Telescope Notes\n\n{body}")

    first_pass = chunk_documents(load_documents(corpus), chunk_size=140, chunk_overlap=20)
    second_pass = chunk_documents(load_documents(corpus), chunk_size=140, chunk_overlap=20)

    assert len(first_pass) > 1
    assert [chunk.id for chunk in first_pass] == [chunk.id for chunk in second_pass]
    assert len({chunk.id for chunk in first_pass}) == len(first_pass)
    assert all(len(chunk.id or "") == 16 for chunk in first_pass)

    for chunk in first_pass:
        source = str(chunk.metadata["source"])
        chunk_index = int(chunk.metadata["chunk_index"])
        assert source == "telescope.md"
        assert not Path(source).is_absolute()
        assert chunk.id == build_chunk_id(source, chunk_index, chunk.page_content)
        assert "start_index" in chunk.metadata

    indices = [chunk.metadata["chunk_index"] for chunk in first_pass]
    assert indices == list(range(len(indices)))
    assert all(chunk.page_content.strip() == chunk.page_content for chunk in first_pass)


def test_chunk_documents_skips_blank_chunks_and_validates_arguments(tmp_path: Path) -> None:
    path = tmp_path / "tiny.md"
    _write_markdown(path, "One short sentence about weather.\n")
    documents = load_documents(path)

    with pytest.raises(UsageError):
        chunk_documents(documents, chunk_size=0)
    with pytest.raises(UsageError):
        chunk_documents(documents, chunk_size=100, chunk_overlap=-1)
    with pytest.raises(UsageError):
        chunk_documents(documents, chunk_size=100, chunk_overlap=100)


def test_load_documents_rejects_missing_source_metadata_document() -> None:
    from langchain_core.documents import Document

    with pytest.raises(UsageError):
        chunk_documents([Document(page_content="text", metadata={})])
