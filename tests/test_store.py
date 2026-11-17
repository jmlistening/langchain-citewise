from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from citewise.config import UsageError
from citewise.store import build_index, load_index


def _chunks() -> list[Document]:
    return [
        Document(
            id="a" * 16,
            page_content="alpha beta gamma",
            metadata={
                "source": "corpus/a.md",
                "title": "Alpha",
                "chunk_index": 0,
                "chunk_id": "a" * 16,
            },
        ),
        Document(
            id="b" * 16,
            page_content="delta epsilon",
            metadata={
                "source": "corpus/b.md",
                "title": "Beta",
                "chunk_index": 0,
                "chunk_id": "b" * 16,
            },
        ),
    ]


def test_manifest_round_trip_and_relative_sources(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    manifest = build_index(
        index_dir, _chunks(), provider="hashing", chunk_size=800, chunk_overlap=120
    )

    assert manifest["version"] == 1
    assert manifest["file_count"] == 2
    assert manifest["chunk_count"] == 2
    assert manifest["sources"] == ["corpus/a.md", "corpus/b.md"]
    assert manifest["embedding"] == {"provider": "hashing", "dim": 512, "model": None}

    stored = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stored == manifest
    assert all(not Path(source).is_absolute() for source in stored["sources"])

    loaded = load_index(index_dir)
    assert loaded.manifest == manifest
    assert sorted(document.metadata["source"] for document in loaded.documents) == [
        "corpus/a.md",
        "corpus/b.md",
    ]
    assert sorted(document.id for document in loaded.documents) == ["a" * 16, "b" * 16]
    assert all("title" in document.metadata for document in loaded.documents)


def test_build_index_rejects_absolute_source_paths(tmp_path: Path) -> None:
    drive_absolute = "C" + ":" + "/somewhere/private.md"
    absolute_chunks = [
        Document(
            id="c" * 16,
            page_content="contents",
            metadata={"source": drive_absolute, "chunk_index": 0},
        )
    ]
    with pytest.raises(UsageError) as excinfo:
        build_index(tmp_path / "index", absolute_chunks, chunk_size=800, chunk_overlap=120)
    assert "relative path" in str(excinfo.value)


def test_load_index_rejects_embedding_mismatch(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    build_index(index_dir, _chunks(), provider="hashing", chunk_size=800, chunk_overlap=120)

    with pytest.raises(UsageError) as excinfo:
        load_index(index_dir, provider="openai")
    assert "embedding mismatch" in str(excinfo.value)
    assert "Re-run `citewise ingest`" in str(excinfo.value)

    with pytest.raises(UsageError) as excinfo:
        load_index(index_dir, provider="hashing", model="text-embedding-3-small")
    assert "no embedding model" in str(excinfo.value)


def test_load_index_reports_missing_and_corrupt_indexes(tmp_path: Path) -> None:
    with pytest.raises(UsageError) as excinfo:
        load_index(tmp_path / "missing")
    assert "No index found" in str(excinfo.value)
    assert "citewise ingest" in str(excinfo.value)

    index_dir = tmp_path / "broken"
    index_dir.mkdir()
    (index_dir / "index.json").write_text("{}", encoding="utf-8")
    (index_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(UsageError) as excinfo:
        load_index(index_dir)
    assert "corrupt" in str(excinfo.value)

    (index_dir / "manifest.json").write_text('{"version": 99}', encoding="utf-8")
    with pytest.raises(UsageError) as excinfo:
        load_index(index_dir)
    assert "unsupported version" in str(excinfo.value)


def test_load_index_detects_chunk_count_mismatch(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    build_index(index_dir, _chunks(), provider="hashing", chunk_size=800, chunk_overlap=120)

    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunk_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(UsageError) as excinfo:
        load_index(index_dir)
    assert "inconsistent" in str(excinfo.value)


def test_load_index_rejects_manifest_with_invalid_utf8(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    build_index(index_dir, _chunks(), provider="hashing", chunk_size=800, chunk_overlap=120)

    (index_dir / "manifest.json").write_bytes(b'\xff\xfe{"version": 1}')

    with pytest.raises(UsageError) as excinfo:
        load_index(index_dir)
    message = str(excinfo.value)
    assert "not valid UTF-8" in message
    assert "citewise ingest" in message


def test_build_index_reports_unwritable_index_path(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("this is a file, not a directory", encoding="utf-8")

    with pytest.raises(UsageError) as excinfo:
        build_index(blocker, _chunks(), provider="hashing", chunk_size=800, chunk_overlap=120)

    assert "could not write the index" in str(excinfo.value)
    assert "--index" in str(excinfo.value)
