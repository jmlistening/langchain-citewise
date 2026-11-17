"""Grounded answering: an LCEL LLM chain and an offline extractive fallback."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import CiteWiseError, get_model
from .retriever import document_key, tokenize

CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")

MIN_SENTENCE_CHARS = 20
MAX_SENTENCE_CHARS = 600
MAX_OFFLINE_SENTENCES = 3
SNIPPET_CHARS = 240

SYSTEM_PROMPT = """\
You are CiteWise, a careful research assistant. Answer the user's question \
using ONLY the numbered context passages provided.

Rules:
- Cite every factual claim with the bracketed number of the passage that \
supports it, written as [n], for example [1] or [2][3].
- If the context does not contain the answer, say that the context is \
insufficient. Do not use outside knowledge.
- Never invent citation numbers. Only cite numbers that appear in the context.
- Answer in the same language as the question.
- Be concise: a short paragraph or a few bullets.
"""

HUMAN_PROMPT = """\
Context passages:

{context}

Question: {question}
"""


@dataclass
class Source:
    """One cited source passage, numbered as it appears in the answer."""

    n: int
    source: str
    title: str
    snippet: str
    rrf: float
    chunk_id: str


@dataclass
class AnswerResult:
    """A grounded answer plus its citations and diagnostics."""

    text: str
    sources: list[Source] = field(default_factory=list)
    mode: str = "offline-extractive"
    model: str | None = None
    retrieved_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "text": self.text,
            "sources": [
                {
                    "n": source.n,
                    "source": source.source,
                    "title": source.title,
                    "snippet": source.snippet,
                    "rrf": source.rrf,
                    "chunk_id": source.chunk_id,
                }
                for source in self.sources
            ],
            "mode": self.mode,
            "model": self.model,
            "retrieved_count": self.retrieved_count,
            "warnings": list(self.warnings),
        }


def _shorten(text: str, limit: int = SNIPPET_CHARS) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _source_title(document: Document) -> str:
    title = str(document.metadata.get("title") or "").strip()
    if title:
        return title
    source = str(document.metadata.get("source") or "")
    return Path(source).stem or "Untitled"


def _source_from_document(number: int, document: Document) -> Source:
    return Source(
        n=number,
        source=str(document.metadata.get("source") or "unknown"),
        title=_source_title(document),
        snippet=_shorten(document.page_content),
        rrf=float(document.metadata.get("rrf") or 0.0),
        chunk_id=document_key(document),
    )


def format_context(documents: list[Document]) -> str:
    """Build the numbered context block shown to the model."""

    blocks: list[str] = []
    for number, document in enumerate(documents, start=1):
        source = str(document.metadata.get("source") or "unknown")
        title = _source_title(document)
        content = _WHITESPACE_RE.sub(" ", document.page_content).strip()
        blocks.append(f"[{number}] {title} ({source})\n{content}")
    return "\n\n".join(blocks)


def build_llm_chain(llm: Any) -> Any:
    """Build the LCEL chain ``prompt | llm | StrOutputParser()``."""

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt | llm | StrOutputParser()


def parse_citations(text: str, documents: list[Document]) -> tuple[list[Source], list[str]]:
    """Validate citation numbers and map them to retrieved documents.

    Out-of-range numbers are dropped and reported as warnings. Sources are
    de-duplicated by chunk id, keeping the first citation number seen.
    """

    sources: list[Source] = []
    warnings: list[str] = []
    seen: set[str] = set()
    total = len(documents)

    for match in CITATION_RE.finditer(text):
        number = int(match.group(1))
        if number < 1 or number > total:
            warning = f"model cited missing source [{number}]"
            if warning not in warnings:
                warnings.append(warning)
            continue
        document = documents[number - 1]
        key = document_key(document)
        if key in seen:
            continue
        seen.add(key)
        sources.append(_source_from_document(number, document))

    if documents and not sources and not warnings:
        warnings.append("model answer contained no citations")
    return sources, warnings


def answer_with_llm(question: str, documents: list[Document], llm: Any) -> AnswerResult:
    """Answer with the LCEL chain and validate the citations it produces."""

    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    if not documents:
        return AnswerResult(
            text="No relevant passages found in the index for this question.",
            sources=[],
            mode="llm",
            model=model_name,
            retrieved_count=0,
            warnings=["No passages were retrieved for this question."],
        )

    try:
        text = build_llm_chain(llm).invoke(
            {"context": format_context(documents), "question": question}
        )
    except CiteWiseError:
        raise
    except Exception as exc:
        raise CiteWiseError(
            f"LLM request failed: {exc}. Check your API settings, or rerun with --offline.",
            exit_code=1,
        ) from exc

    sources, warnings = parse_citations(str(text), documents)
    return AnswerResult(
        text=str(text),
        sources=sources,
        mode="llm",
        model=str(model_name) if model_name else get_model(),
        retrieved_count=len(documents),
        warnings=warnings,
    )


def split_sentences(text: str) -> list[str]:
    """Split prose into substantive sentences and normalize whitespace."""

    sentences: list[str] = []
    for piece in _SENTENCE_RE.split(text):
        sentence = _WHITESPACE_RE.sub(" ", piece).strip()
        if not sentence:
            continue
        if len(sentence) < MIN_SENTENCE_CHARS or len(sentence) > MAX_SENTENCE_CHARS:
            continue
        sentences.append(sentence)
    return sentences


@dataclass
class _SentenceCandidate:
    document_rank: int
    sentence_index: int
    sentence: str
    score: float


def _offline_candidates(question: str, documents: list[Document]) -> list[_SentenceCandidate]:
    query_terms = sorted(set(tokenize(question)))
    if not query_terms:
        return []

    rows: list[tuple[int, int, str, list[str]]] = []
    for rank, document in enumerate(documents, start=1):
        for index, sentence in enumerate(split_sentences(document.page_content)):
            rows.append((rank, index, sentence, tokenize(sentence)))
    if not rows:
        return []

    document_frequency = Counter(
        term for _rank, _index, _sentence, tokens in rows for term in set(tokens)
    )
    total_rows = len(rows)
    idf = {
        term: math.log((total_rows + 1) / (document_frequency.get(term, 0) + 1)) + 1.0
        for term in query_terms
    }

    candidates: list[_SentenceCandidate] = []
    for rank, index, sentence, tokens in rows:
        token_set = set(tokens)
        matched = [term for term in query_terms if term in token_set]
        if not matched:
            continue
        overlap = sum(idf[term] for term in matched)
        length_penalty = 1.0 + math.log1p(len(tokens))
        score = overlap / length_penalty
        score += 0.2 / rank
        title_tokens = set(tokenize(str(documents[rank - 1].metadata.get("title") or "")))
        if title_tokens:
            title_matches = sum(1 for term in query_terms if term in title_tokens)
            score += 0.3 * title_matches / len(query_terms)
        candidates.append(
            _SentenceCandidate(
                document_rank=rank,
                sentence_index=index,
                sentence=sentence,
                score=score,
            )
        )
    return candidates


def answer_offline(question: str, documents: list[Document]) -> AnswerResult:
    """Extractive, citation-carrying answer built without any model call."""

    if not documents:
        return AnswerResult(
            text="No relevant passages found in the index for this question.",
            sources=[],
            mode="offline-extractive",
            model=None,
            retrieved_count=0,
            warnings=["No passages were retrieved for this question."],
        )

    candidates = _offline_candidates(question, documents)
    warnings: list[str] = []

    if not candidates:
        top_document = documents[0]
        sentences = split_sentences(top_document.page_content)
        fallback = sentences[0] if sentences else _shorten(top_document.page_content, 300)
        if not fallback:
            fallback = "The retrieved passage is empty."
        text = f"{fallback} [1]"
        return AnswerResult(
            text=text,
            sources=[_source_from_document(1, top_document)],
            mode="offline-extractive",
            model=None,
            retrieved_count=len(documents),
            warnings=[
                "Evidence is weak: no retrieved sentence shared terms with the "
                "question, so the first substantive sentence is shown instead."
            ],
        )

    selected: list[_SentenceCandidate] = []
    seen_sentences: set[str] = set()
    for candidate in sorted(
        candidates, key=lambda item: (-item.score, item.document_rank, item.sentence_index)
    ):
        key = _WHITESPACE_RE.sub(" ", candidate.sentence.lower()).strip()
        if key in seen_sentences:
            continue
        seen_sentences.add(key)
        selected.append(candidate)
        if len(selected) >= MAX_OFFLINE_SENTENCES:
            break

    by_document: dict[int, list[_SentenceCandidate]] = {}
    for candidate in selected:
        by_document.setdefault(candidate.document_rank, []).append(candidate)

    parts: list[str] = []
    sources: list[Source] = []
    for rank in sorted(by_document):
        for candidate in sorted(by_document[rank], key=lambda item: item.sentence_index):
            parts.append(f"{candidate.sentence} [{rank}]")
        sources.append(_source_from_document(rank, documents[rank - 1]))

    text = " ".join(parts)
    if not text:
        text = "No relevant passages found in the index for this question."
        sources = []
        warnings.append("No extractive sentence could be selected.")

    return AnswerResult(
        text=text,
        sources=sources,
        mode="offline-extractive",
        model=None,
        retrieved_count=len(documents),
        warnings=warnings,
    )


def answer_question(
    question: str,
    documents: list[Document],
    *,
    offline: bool,
    llm: Any | None = None,
) -> AnswerResult:
    """Dispatch to the offline extractive answerer or the LCEL LLM chain."""

    if offline or llm is None:
        return answer_offline(question, documents)
    return answer_with_llm(question, documents, llm)
