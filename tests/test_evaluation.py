from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from citewise.answer import AnswerResult
from citewise.config import UsageError
from citewise.evaluation import EvalQuestion, evaluate, load_eval_file


class StubRetriever:
    def __init__(self, results: dict[str, list[Document]]):
        self.results = results
        self.calls: list[str] = []

    def invoke(self, question: str) -> list[Document]:
        self.calls.append(question)
        return self.results[question]


def _document(source: str, text: str = "text") -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_metrics_math_with_stub_results() -> None:
    questions = [
        EvalQuestion("q1", ["a.md"]),
        EvalQuestion("q2", ["z.md"]),
        EvalQuestion("q3", ["a.md"]),
    ]
    retriever = StubRetriever(
        {
            "q1": [_document("b.md"), _document("a.md"), _document("c.md")],
            "q2": [_document("x.md"), _document("y.md")],
            "q3": [_document("a.md")],
        }
    )

    report = evaluate(questions, retriever, k=3)

    assert report["k"] == 3
    assert report["questions"][0]["first_relevant_rank"] == 2
    assert report["questions"][0]["reciprocal_rank"] == 0.5
    assert report["questions"][0]["hit"] is True
    assert report["questions"][1]["first_relevant_rank"] is None
    assert report["questions"][1]["reciprocal_rank"] == 0.0
    assert report["questions"][1]["hit"] is False
    assert report["questions"][2]["reciprocal_rank"] == 1.0

    metrics = report["metrics"]
    assert metrics["hit_at_k"] == pytest.approx(0.6667)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["answer_term_coverage"] is None
    assert metrics["question_count"] == 3


def test_k_cutoff_uses_unique_sources_in_rank_order() -> None:
    question = EvalQuestion("q", ["a.md"])
    retriever = StubRetriever({"q": [_document("x.md"), _document("y.md"), _document("a.md")]})

    assert evaluate([question], retriever, k=2)["questions"][0]["hit"] is False
    assert evaluate([question], retriever, k=3)["questions"][0]["hit"] is True

    duplicate = StubRetriever({"q": [_document("a.md"), _document("a.md"), _document("x.md")]})
    row = evaluate([question], duplicate, k=3)["questions"][0]
    assert row["retrieved_sources"] == ["a.md", "x.md"]
    assert row["first_relevant_rank"] == 1


def test_source_labels_match_with_or_without_a_parent_prefix() -> None:
    question = EvalQuestion("q", ["corpus/a.md"])
    retriever = StubRetriever({"q": [_document("other.md"), _document("a.md")]})
    row = evaluate([question], retriever, k=2)["questions"][0]
    assert row["first_relevant_rank"] == 2
    assert row["hit"] is True


def test_expected_term_coverage_from_offline_answers() -> None:
    questions = [
        EvalQuestion("q1", ["a.md"], ["borosilicate glass", "CO2 snow"]),
        EvalQuestion("q2", ["b.md"], ["winter closure", "March 1"]),
    ]
    retriever = StubRetriever({"q1": [_document("a.md")], "q2": [_document("b.md")]})

    def answer_fn(question: str, documents: list[Document]) -> AnswerResult:
        text = (
            "The mirror is made of borosilicate glass and cleaned with CO2 snow."
            if question == "q1"
            else "The winter closure runs to February 28."
        )
        return AnswerResult(text=text, retrieved_count=len(documents))

    report = evaluate(questions, retriever, k=1, answer_fn=answer_fn)

    assert report["questions"][0]["expected_term_coverage"] == 1.0
    assert report["questions"][1]["expected_term_coverage"] == 0.5
    assert report["metrics"]["answer_term_coverage"] == pytest.approx(0.75)
    assert "answer" not in report["questions"][0]


def test_load_eval_file_validates_jsonl(tmp_path: Path) -> None:
    eval_path = tmp_path / "questions.jsonl"
    records: list[dict[str, Any]] = [
        {
            "question": "What is the mirror made of?",
            "relevant_sources": ["corpus/telescope.md"],
            "expected_terms": ["glass"],
        },
        {"question": "Second question", "relevant_sources": ["corpus/outreach.md"]},
    ]
    eval_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n\n", encoding="utf-8"
    )

    questions = load_eval_file(eval_path)
    assert len(questions) == 2
    assert questions[0].expected_terms == ["glass"]
    assert questions[1].expected_terms == []


def test_load_eval_file_rejects_bad_records(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"

    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(UsageError):
        load_eval_file(path)

    path.write_text('{"question": "q"}\n', encoding="utf-8")
    with pytest.raises(UsageError):
        load_eval_file(path)

    path.write_text('{"question": "q", "relevant_sources": "a.md"}\n', encoding="utf-8")
    with pytest.raises(UsageError):
        load_eval_file(path)

    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(UsageError):
        load_eval_file(path)


def test_evaluate_guards_invalid_arguments() -> None:
    retriever = StubRetriever({})
    with pytest.raises(UsageError):
        evaluate([], retriever, k=3)
    with pytest.raises(UsageError):
        evaluate([EvalQuestion("q", ["a.md"])], retriever, k=0)
