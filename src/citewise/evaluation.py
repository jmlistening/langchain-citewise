"""Offline retrieval evaluation: hit@k, MRR, and expected-term coverage."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .answer import AnswerResult
from .config import UsageError

_WHITESPACE_RE = re.compile(r"\s+")

AnswerFn = Callable[[str, list[Document]], AnswerResult]


@dataclass
class EvalQuestion:
    """One labeled evaluation question."""

    question: str
    relevant_sources: list[str]
    expected_terms: list[str] = field(default_factory=list)


def _parse_question(payload: Any, line_number: int, path: Path) -> EvalQuestion:
    where = f"{path}:{line_number}"
    if not isinstance(payload, dict):
        raise UsageError(f"{where}: each JSONL line must be a JSON object")
    question = payload.get("question")
    relevant = payload.get("relevant_sources")
    terms = payload.get("expected_terms", [])
    if not isinstance(question, str) or not question.strip():
        raise UsageError(f"{where}: 'question' must be a non-empty string")
    if (
        not isinstance(relevant, list)
        or not relevant
        or not all(isinstance(item, str) and item for item in relevant)
    ):
        raise UsageError(f"{where}: 'relevant_sources' must be a non-empty list of strings")
    if not isinstance(terms, list) or not all(isinstance(item, str) for item in terms):
        raise UsageError(f"{where}: 'expected_terms' must be a list of strings")
    return EvalQuestion(
        question=question.strip(),
        relevant_sources=[item.replace("\\", "/") for item in relevant],
        expected_terms=[item for item in terms if item.strip()],
    )


def load_eval_file(path: Path) -> list[EvalQuestion]:
    """Read a JSONL evaluation file and validate every record."""

    eval_path = Path(path).expanduser()
    if not eval_path.is_file():
        raise UsageError(f"evaluation file not found: {eval_path}")

    questions: list[EvalQuestion] = []
    try:
        lines = eval_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise UsageError(f"could not read evaluation file {eval_path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UsageError(f"{eval_path}:{line_number}: invalid JSON: {exc}") from exc
        questions.append(_parse_question(payload, line_number, eval_path))

    if not questions:
        raise UsageError(f"evaluation file has no questions: {eval_path}")
    return questions


def _unique_sources(documents: list[Document]) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for document in documents:
        source = str(document.metadata.get("source") or "").replace("\\", "/")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def _source_matches(retrieved: str, relevant: str) -> bool:
    """Match a retrieved source against a label, tolerating a shared prefix.

    Labels may be written relative to the ingest root (``telescope.md``) or
    relative to a parent folder that was included in the ingest path
    (``corpus/telescope.md``). Either spelling matches the other.
    """

    if retrieved == relevant:
        return True
    return retrieved.endswith("/" + relevant) or relevant.endswith("/" + retrieved)


def _coverage(text: str, terms: list[str]) -> float | None:
    if not terms:
        return None
    haystack = _normalize_text(text)
    matched = sum(1 for term in terms if _normalize_text(term) in haystack)
    return matched / len(terms)


def evaluate(
    questions: list[EvalQuestion],
    retriever: Any,
    *,
    k: int = 5,
    answer_fn: AnswerFn | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval (and optionally offline answer coverage).

    Hit@k and MRR are computed at the source-document level using the unique
    sources in retrieval order. When ``answer_fn`` is supplied, it is called
    with the already-retrieved documents so the evaluator never makes an extra
    retrieval or LLM call.
    """

    if k <= 0:
        raise UsageError("--k must be greater than zero")
    if not questions:
        raise UsageError("no evaluation questions to score")

    rows: list[dict[str, Any]] = []
    hit_values: list[float] = []
    reciprocal_ranks: list[float] = []
    coverages: list[float] = []

    for item in questions:
        documents = list(retriever.invoke(item.question))[:k]
        sources = _unique_sources(documents)
        first_relevant_rank: int | None = None
        for rank, source in enumerate(sources, start=1):
            if any(_source_matches(source, label) for label in item.relevant_sources):
                first_relevant_rank = rank
                break

        hit = first_relevant_rank is not None
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        hit_values.append(1.0 if hit else 0.0)
        reciprocal_ranks.append(reciprocal_rank)

        row: dict[str, Any] = {
            "question": item.question,
            "retrieved_sources": sources,
            "first_relevant_rank": first_relevant_rank,
            "hit": hit,
            "reciprocal_rank": round(reciprocal_rank, 4),
        }

        if answer_fn is not None and item.expected_terms:
            result = answer_fn(item.question, documents)
            text = result.text if isinstance(result, AnswerResult) else str(result)
            coverage = _coverage(text, item.expected_terms)
            if coverage is not None:
                row["expected_term_coverage"] = round(coverage, 4)
                coverages.append(coverage)

        rows.append(row)

    metrics: dict[str, Any] = {
        "hit_at_k": round(sum(hit_values) / len(hit_values), 4),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "answer_term_coverage": (round(sum(coverages) / len(coverages), 4) if coverages else None),
        "question_count": len(rows),
    }
    return {"k": k, "questions": rows, "metrics": metrics}
