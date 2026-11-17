from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from citewise.embeddings import HashingEmbeddings
from citewise.retriever import HybridRetriever, normalize_token, tokenize


class FakeVectorStore:
    """Minimal vector store stand-in that returns a controlled ranking."""

    def __init__(self, results: list[tuple[Document, float]], store: dict | None = None):
        self.results = results
        # A non-empty sentinel keeps the retriever's "store is empty" fast path
        # from skipping the vector search in ranking tests.
        self.store = {"sentinel": True} if store is None else store
        self.last_k: int | None = None

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        self.last_k = k
        return self.results[:k]


def _document(chunk_id: str, text: str, **metadata: Any) -> Document:
    base = {"source": f"{chunk_id}.md", "title": chunk_id.title(), "chunk_index": 0}
    base.update(metadata)
    return Document(id=chunk_id, page_content=text, metadata=base)


def test_tokenize_removes_stopwords_and_normalizes() -> None:
    tokens = tokenize("The cleaning of the mirrors is done with CO2 snow.")
    assert "the" not in tokens
    assert "clean" in tokens
    assert "mirror" in tokens
    assert normalize_token("cleaned") == "clean"
    assert normalize_token("mirrors") == "mirror"
    assert normalize_token("is") == "is"


def test_metadata_preserved_and_store_documents_not_mutated() -> None:
    documents = [
        _document("d1", "alpha beta gamma", custom="kept"),
        _document("d2", "alpha beta"),
        _document("d3", "delta epsilon"),
    ]
    store = FakeVectorStore([(documents[2], 0.9), (documents[0], 0.8)])
    retriever = HybridRetriever(documents=documents, vector_store=store, top_k=3, fetch_k=8)

    results = retriever.invoke("alpha beta")

    assert results
    assert store.last_k == 8
    d1_result = next(result for result in results if result.id == "d1")
    assert d1_result.metadata["custom"] == "kept"
    for result in results:
        assert "vector_rank" in result.metadata
        assert "bm25_rank" in result.metadata
        assert "rrf" in result.metadata
    assert "vector_rank" not in documents[0].metadata
    assert "rrf" not in documents[0].metadata


def test_rrf_prefers_document_found_by_both_rankers() -> None:
    documents = [
        _document("d1", "alpha beta gamma", custom="kept"),
        _document("d2", "alpha beta"),
        _document("d3", "delta epsilon"),
    ]
    store = FakeVectorStore([(documents[2], 0.9), (documents[0], 0.8)])
    retriever = HybridRetriever(documents=documents, vector_store=store, top_k=3)

    results = retriever.invoke("alpha beta")

    assert results[0].id == "d1"
    assert results[0].metadata["vector_rank"] == 2
    assert results[0].metadata["bm25_rank"] in {1, 2}
    expected = 1.0 / (60 + results[0].metadata["vector_rank"]) + 1.0 / (
        60 + results[0].metadata["bm25_rank"]
    )
    assert results[0].metadata["rrf"] == pytest.approx(expected)
    assert results[0].metadata["rrf"] > results[1].metadata["rrf"]

    single_list_scores = [result.metadata["rrf"] for result in results if result.id != "d1"]
    assert all(result.metadata["rrf"] < results[0].metadata["rrf"] for result in results[1:])
    assert max(single_list_scores) == pytest.approx(1.0 / 61)


def test_top_k_truncates_results() -> None:
    documents = [_document(f"d{index}", f"alpha token{index}") for index in range(6)]
    store = FakeVectorStore([(document, 0.5) for document in documents])
    retriever = HybridRetriever(documents=documents, vector_store=store, top_k=2)
    assert len(retriever.invoke("alpha")) == 2


def test_empty_vector_store_still_returns_bm25_results() -> None:
    documents = [
        _document("d1", "alpha beta gamma"),
        _document("d2", "delta epsilon"),
        _document("d3", "eta theta"),
        _document("d4", "iota kappa"),
    ]
    store = FakeVectorStore([], store={})
    retriever = HybridRetriever(documents=documents, vector_store=store, top_k=2)

    results = retriever.invoke("alpha")

    assert [result.id for result in results] == ["d1"]
    assert results[0].metadata["vector_rank"] is None
    assert results[0].metadata["bm25_rank"] == 1


def test_no_bm25_hits_falls_back_to_vector_results() -> None:
    documents = [
        _document("d1", "alpha beta gamma"),
        _document("d2", "delta epsilon"),
    ]
    store = FakeVectorStore([(documents[1], 0.9), (documents[0], 0.5)])
    retriever = HybridRetriever(documents=documents, vector_store=store, top_k=2)

    results = retriever.invoke("zeta")

    assert results
    assert all(result.metadata["bm25_rank"] is None for result in results)
    assert results[0].metadata["vector_rank"] == 1


def test_empty_documents_return_nothing() -> None:
    retriever = HybridRetriever(documents=[], vector_store=FakeVectorStore([]), top_k=2)
    assert retriever.invoke("anything") == []


def test_real_inmemory_vector_store_integration() -> None:
    embeddings = HashingEmbeddings()
    documents = [
        Document(
            id="mirror",
            page_content="The primary mirror is cleaned with CO2 snow.",
            metadata={"source": "telescope.md", "title": "Telescope", "chunk_index": 0},
        ),
        Document(
            id="weather",
            page_content="Observing stops when wind gusts exceed 45 km/h.",
            metadata={"source": "operations.md", "title": "Operations", "chunk_index": 0},
        ),
    ]
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(documents)

    retriever = HybridRetriever(documents=documents, vector_store=store, top_k=2)
    results = retriever.invoke("How is the primary mirror cleaned?")

    assert results
    assert results[0].id == "mirror"
    assert results[0].metadata["source"] == "telescope.md"
    assert isinstance(results[0].metadata["rrf"], float)


def test_zero_vector_queries_fall_back_to_bm25_without_crashing() -> None:
    embeddings = HashingEmbeddings()
    documents = [
        Document(
            id="mirror",
            page_content="The primary mirror is cleaned with CO2 snow.",
            metadata={"source": "telescope.md", "title": "Telescope", "chunk_index": 0},
        ),
        Document(
            id="weather",
            page_content="Observing stops when wind gusts exceed 45 km/h.",
            metadata={"source": "operations.md", "title": "Operations", "chunk_index": 0},
        ),
    ]
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(documents)
    retriever = HybridRetriever(documents=documents, vector_store=store, top_k=2)

    # These questions contain no ASCII alphanumerics, so the hashing embedder
    # returns the all-zero vector and BM25 has no tokens either.
    for query in ("?", "...", "望远镜的主镜是什么材料做的？"):
        assert retriever.invoke(query) == []


def test_partly_non_latin_query_still_retrieves_bm25_match() -> None:
    embeddings = HashingEmbeddings()
    documents = [
        Document(
            id="mirror",
            page_content="The primary mirror is cleaned with CO2 snow.",
            metadata={"source": "telescope.md", "title": "Telescope", "chunk_index": 0},
        ),
        Document(
            id="weather",
            page_content="Observing stops when wind gusts exceed 45 km/h.",
            metadata={"source": "operations.md", "title": "Operations", "chunk_index": 0},
        ),
        Document(
            id="dome",
            page_content="The dome opens at dusk and closes at dawn.",
            metadata={"source": "dome.md", "title": "Dome", "chunk_index": 0},
        ),
    ]
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(documents)
    retriever = HybridRetriever(documents=documents, vector_store=store, top_k=2)

    results = retriever.invoke("mirror 望远镜")

    assert results[0].id == "mirror"
    assert results[0].metadata["bm25_rank"] == 1


def test_all_zero_vector_index_degrades_to_bm25() -> None:
    embeddings = HashingEmbeddings()
    documents = [
        Document(
            id="mirror",
            page_content="望远镜的主镜由三十六块玻璃拼接而成。",
            metadata={"source": "cjk-mirror.md", "title": "Mirror Notes", "chunk_index": 0},
        ),
        Document(
            id="weather",
            page_content="大风天气会停止观测。",
            metadata={"source": "cjk-weather.md", "title": "Weather Notes", "chunk_index": 0},
        ),
        Document(
            id="dome",
            page_content="圆顶在黄昏时打开。",
            metadata={"source": "cjk-dome.md", "title": "Dome Notes", "chunk_index": 0},
        ),
    ]
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(documents)
    retriever = HybridRetriever(documents=documents, vector_store=store, top_k=2)

    # Every indexed vector is zero, so all cosine similarities are NaN and
    # langchain-core raises ValueError; retrieval must still use BM25, which
    # indexes the ASCII titles even though the bodies are non-Latin text.
    results = retriever.invoke("mirror")

    assert [result.id for result in results] == ["mirror"]
    assert results[0].metadata["vector_rank"] is None
    assert results[0].metadata["bm25_rank"] == 1


def test_zero_query_vector_skips_the_vector_store_call() -> None:
    embeddings = HashingEmbeddings()
    document = _document("d1", "alpha beta gamma")
    store = FakeVectorStore([(document, 0.9)])
    store.embedding = embeddings

    retriever = HybridRetriever(documents=[document], vector_store=store, top_k=2)

    assert retriever.invoke("?") == []
    assert store.last_k is None


def test_nan_similarity_scores_are_dropped() -> None:
    documents = [_document("d1", "alpha beta gamma"), _document("d2", "delta epsilon")]
    store = FakeVectorStore([(documents[0], float("nan")), (documents[1], 0.5)])
    retriever = HybridRetriever(documents=documents, vector_store=store, top_k=2)

    results = retriever.invoke("zeta")

    assert [result.id for result in results] == ["d2"]
    assert results[0].metadata["vector_rank"] == 1
