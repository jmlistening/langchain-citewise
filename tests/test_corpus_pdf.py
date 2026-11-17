"""Tests for PDF ingestion. Skipped when the optional ``pdf`` extra is absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from citewise.corpus import load_documents

pytest.importorskip("pypdf")


def _write_text_pdf(path: Path, page_texts: list[str]) -> None:
    """Write a minimal PDF with one text page per non-empty entry.

    An empty string produces a blank page without a content stream.
    """

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    font_reference = writer._add_object(font)

    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if not text:
            continue
        resources = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_reference
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer.write(handle)


def test_pdf_pages_get_page_metadata_and_relative_source(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_text_pdf(
        pdf_path,
        [
            "The primary mirror is cleaned with CO2 snow.",
            "",
            "The dome opens at dusk.",
        ],
    )

    warnings: list[str] = []
    documents = load_documents(pdf_path, warn=warnings.append)

    assert warnings == []
    assert [document.metadata["page"] for document in documents] == [1, 3]
    assert all(document.metadata["source"] == "sample.pdf" for document in documents)
    assert all(document.metadata["title"] == "sample" for document in documents)
    assert "CO2 snow" in documents[0].page_content
    assert "dome" in documents[1].page_content


def test_pdf_in_directory_uses_path_relative_to_ingest_root(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_text_pdf(corpus / "sub" / "mirror.pdf", ["The mirror has 36 segments."])

    documents = load_documents(corpus, warn=lambda message: pytest.fail(message))

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "sub/mirror.pdf"
    assert "\\" not in documents[0].metadata["source"]


def test_pdf_without_extractable_text_warns_and_is_skipped(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    _write_text_pdf(pdf_path, ["", ""])

    warnings: list[str] = []
    assert load_documents(pdf_path, warn=warnings.append) == []
    assert any("no extractable text" in warning for warning in warnings)


def test_unreadable_pdf_is_skipped_with_warning(tmp_path: Path) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"this is not a PDF file")

    warnings: list[str] = []
    assert load_documents(pdf_path, warn=warnings.append) == []
    assert any("skipping unreadable PDF" in warning for warning in warnings)
