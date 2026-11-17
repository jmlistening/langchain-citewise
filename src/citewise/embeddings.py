"""Deterministic, dependency-free hashing embeddings.

This is a lexical feature hashing model, not a trained semantic model. It is
the default embedding provider because it is deterministic and works with no
API key and no network access.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

_WORD_RE = re.compile(r"[a-z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_HASH_KEY = b"citewise-hashing-v1"


class HashingEmbeddings(Embeddings):
    """Feature-hashing embeddings over word unigrams, bigrams, and char trigrams.

    Every feature is hashed with keyed BLAKE2b into one of ``dim`` buckets with
    a sign bit, then the vector is L2-normalized. The result is fully
    deterministic across processes and machines.

    Empty or featureless text maps to the zero vector. Similarity code in this
    project treats a zero vector as "no similarity", never as NaN.
    """

    def __init__(
        self,
        dim: int = 512,
        *,
        unigram_weight: float = 1.0,
        bigram_weight: float = 1.4,
        trigram_weight: float = 0.4,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._unigram_weight = unigram_weight
        self._bigram_weight = bigram_weight
        self._trigram_weight = trigram_weight

    @property
    def dim(self) -> int:
        """Embedding dimension."""

        return self._dim

    def _features(self, text: str) -> list[tuple[str, str, float]]:
        lowered = text.lower()
        words = _WORD_RE.findall(lowered)

        features: list[tuple[str, str, float]] = []
        features.extend(("u", word, self._unigram_weight) for word in words)
        features.extend(
            ("b", f"{left} {right}", self._bigram_weight)
            for left, right in zip(words, words[1:], strict=False)
        )
        compact = _NON_ALNUM_RE.sub("", lowered)
        features.extend(
            ("t", compact[index : index + 3], self._trigram_weight)
            for index in range(max(0, len(compact) - 2))
        )
        return features

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for _kind, feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8, key=_HASH_KEY).digest()
            value = int.from_bytes(digest, "big")
            index = (value >> 1) % self._dim
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""

        return self._embed(text)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity with a safe zero-vector result."""

    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
