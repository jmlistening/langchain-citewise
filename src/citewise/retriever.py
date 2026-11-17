"""Hybrid BM25 + vector retrieval fused with reciprocal rank fusion (RRF)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from rank_bm25 import BM25Okapi

from .embeddings import HashingEmbeddings

STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from
    had has have having he her here hers herself him himself his how i if in into
    is it its itself just me more most my myself no nor not now of off on once
    only or other our ours ourselves out over own same she should so some such
    than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while
    who whom why will with would you your yours yourself yourselves
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ing", "ed", "es", "s")


def normalize_token(token: str) -> str:
    """Apply a deliberately light suffix strip to improve lexical matching."""

    if len(token) > 4:
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed and light normalization."""

    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if len(raw) < 2 or raw in STOPWORDS:
            continue
        normalized = normalize_token(raw)
        if normalized in STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


def _document_tokens(document: Document) -> list[str]:
    tokens = tokenize(document.page_content)
    title = str(document.metadata.get("title") or "")
    if title:
        tokens.extend(tokenize(title))
    return tokens


def document_key(document: Document) -> str:
    """Return a stable identity for deduplication during fusion."""

    if document.id:
        return str(document.id)
    source = str(document.metadata.get("source", ""))
    chunk_index = document.metadata.get("chunk_index", "")
    digest = hashlib.sha256(document.page_content.encode()).hexdigest()[:16]
    return f"{source}::{chunk_index}::{digest}"


class HybridRetriever(BaseRetriever):
    """Retrieve chunks by fusing lexical BM25 ranks and vector ranks.

    Only ranks are used from each candidate list, which keeps the fusion robust
    to the very different score scales of BM25 and cosine similarity.
    """

    documents: list[Document] = Field(default_factory=list)
    vector_store: Any = Field(default=None)
    top_k: int = Field(default=4, ge=1)
    fetch_k: int = Field(default=16, ge=1)
    rrf_k: int = Field(default=60, ge=1)

    _bm25: Any = PrivateAttr(default=None)
    _tokenized_documents: list[list[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self._tokenized_documents = [_document_tokens(document) for document in self.documents]
        if any(self._tokenized_documents):
            self._bm25 = BM25Okapi(self._tokenized_documents)

    def _vector_candidates(self, query: str) -> list[Document]:
        if self.vector_store is None or not self.documents:
            return []
        store = getattr(self.vector_store, "store", None)
        if isinstance(store, dict) and not store:
            return []
        embedding = getattr(self.vector_store, "embedding", None)
        if isinstance(embedding, HashingEmbeddings) and not any(embedding.embed_query(query)):
            # A zero query vector has no direction; skip the vector search
            # instead of relying on the similarity backend's zero handling.
            return []
        try:
            results = self.vector_store.similarity_search_with_score(query, k=self.fetch_k)
        except ValueError as exc:
            if "NaN" not in str(exc):
                raise
            # langchain-core's cosine similarity computes 0/0 = NaN when the
            # query or an indexed chunk has an all-zero vector. HashingEmbeddings
            # emits zero vectors for text without ASCII alphanumerics (for
            # example punctuation-only or CJK questions, and non-Latin corpora),
            # so treat the vector ranking as empty and fall back to BM25.
            return []
        # Optional similarity backends (for example simsimd) can return NaN
        # scores instead of raising; drop those entries and keep usable ranks.
        return [document for document, score in results if score == score]

    def _bm25_candidates(self, query: str) -> list[Document]:
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )
        candidates: list[Document] = []
        for index in ranked:
            if float(scores[index]) <= 0.0:
                continue
            candidates.append(self.documents[index])
            if len(candidates) >= self.fetch_k:
                break
        return candidates

    def _fuse(self, vector_docs: list[Document], bm25_docs: list[Document]) -> list[Document]:
        fused: dict[str, dict[str, Any]] = {}

        for label, candidates in (("vector", vector_docs), ("bm25", bm25_docs)):
            for rank, document in enumerate(candidates, start=1):
                key = document_key(document)
                entry = fused.setdefault(
                    key,
                    {
                        "document": document,
                        "vector_rank": None,
                        "bm25_rank": None,
                        "rrf": 0.0,
                    },
                )
                entry[f"{label}_rank"] = rank
                entry["rrf"] += 1.0 / (self.rrf_k + rank)

        ordered = sorted(
            fused.items(),
            key=lambda item: (
                -item[1]["rrf"],
                item[1]["vector_rank"] or 10**9,
                item[1]["bm25_rank"] or 10**9,
                item[0],
            ),
        )

        output: list[Document] = []
        for _key, entry in ordered[: self.top_k]:
            source_document = entry["document"]
            metadata = dict(source_document.metadata)
            metadata["vector_rank"] = entry["vector_rank"]
            metadata["bm25_rank"] = entry["bm25_rank"]
            metadata["rrf"] = entry["rrf"]
            output.append(
                Document(
                    id=source_document.id,
                    page_content=source_document.page_content,
                    metadata=metadata,
                )
            )
        return output

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        vector_docs = self._vector_candidates(query)
        bm25_docs = self._bm25_candidates(query)
        return self._fuse(vector_docs, bm25_docs)
