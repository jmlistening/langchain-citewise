from __future__ import annotations

import math

import pytest

from citewise.embeddings import HashingEmbeddings, cosine_similarity


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def test_dimension_and_determinism() -> None:
    first = HashingEmbeddings()
    second = HashingEmbeddings()

    vector = first.embed_query("The primary mirror is cleaned with CO2 snow.")
    assert len(vector) == 512
    assert vector == second.embed_query("The primary mirror is cleaned with CO2 snow.")
    assert first.dim == 512


def test_vectors_are_l2_normalized() -> None:
    embeddings = HashingEmbeddings()
    for text in ["alpha", "The telescope dome opens at dusk.", "CO2 snow cleaning"]:
        vector = embeddings.embed_query(text)
        assert _norm(vector) == pytest.approx(1.0, abs=1e-9)


def test_empty_text_returns_zero_vector() -> None:
    embeddings = HashingEmbeddings()
    vector = embeddings.embed_query("")
    assert len(vector) == 512
    assert all(component == 0.0 for component in vector)
    assert cosine_similarity(vector, vector) == 0.0


def test_embed_documents_batches() -> None:
    embeddings = HashingEmbeddings()
    vectors = embeddings.embed_documents(["one document", "another document"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 512
    assert vectors[0] != vectors[1]


def test_similarity_orders_related_text_above_unrelated() -> None:
    embeddings = HashingEmbeddings()
    query = embeddings.embed_query("How is the primary mirror cleaned?")
    mirror = embeddings.embed_query(
        "The primary mirror segments are cleaned with carbon dioxide snow."
    )
    weather = embeddings.embed_query(
        "The observatory closes when wind gusts exceed the safety limit."
    )

    assert cosine_similarity(query, mirror) > cosine_similarity(query, weather)
    assert cosine_similarity(query, mirror) > 0.2


def test_identical_text_has_similarity_one() -> None:
    embeddings = HashingEmbeddings()
    vector = embeddings.embed_query("adaptive optics and a laser guide star")
    assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-9)


def test_cosine_similarity_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0])


def test_custom_dimension_and_bad_dimension() -> None:
    embeddings = HashingEmbeddings(dim=64)
    assert len(embeddings.embed_query("small vector")) == 64
    with pytest.raises(ValueError):
        HashingEmbeddings(dim=0)
