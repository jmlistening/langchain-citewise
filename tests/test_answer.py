from __future__ import annotations

import re

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from citewise.answer import (
    AnswerResult,
    answer_offline,
    answer_question,
    answer_with_llm,
    build_llm_chain,
    format_context,
)
from citewise.config import CiteWiseError

CITATION_RE = re.compile(r"\[(\d+)\]")


def _document(
    chunk_id: str, text: str, source: str = "corpus/telescope.md", **metadata: object
) -> Document:
    base = {"source": source, "title": "The Meridian Telescope", "rrf": 0.5}
    base.update(metadata)
    return Document(id=chunk_id, page_content=text, metadata=base)


def test_offline_answer_contains_valid_citations_and_sources() -> None:
    document = _document(
        "mirror",
        "The primary mirror is made of borosilicate glass. "
        "Every 18 months it is cleaned with CO2 snow. "
        "The coating is renewed every five years.",
    )
    unrelated = _document(
        "weather",
        "Observing stops when wind gusts exceed the safety limit.",
        source="corpus/operations.md",
    )

    result = answer_offline(
        "What is the primary mirror made of and how is it cleaned?",
        [document, unrelated],
    )

    assert result.mode == "offline-extractive"
    assert result.model is None
    assert result.retrieved_count == 2
    assert "borosilicate glass" in result.text
    assert "CO2 snow" in result.text
    assert result.sources
    assert len(result.sources) == len({source.chunk_id for source in result.sources})
    for source in result.sources:
        assert 1 <= source.n <= 2
        assert source.source.startswith("corpus/")
        assert source.snippet
        assert source.rrf == pytest.approx(0.5)
    for number in CITATION_RE.findall(result.text):
        assert 1 <= int(number) <= 2


def test_offline_answer_preserves_sentence_order_within_a_source() -> None:
    document = _document(
        "mirror",
        "The primary mirror is made of glass. "
        "The mirror is cleaned with CO2 snow. "
        "The mirror is inspected every year.",
    )

    result = answer_offline("How is the primary mirror cleaned?", [document])
    text = result.text

    assert text.index("made of glass") < text.index("cleaned with CO2 snow")
    assert text.count("[1]") >= 1
    assert len(result.sources) == 1


def test_offline_answer_without_documents() -> None:
    result = answer_offline("Anything?", [])

    assert result.sources == []
    assert result.retrieved_count == 0
    assert "No relevant passages found" in result.text


def test_offline_answer_without_lexical_overlap_warns_and_cites_first() -> None:
    document = _document(
        "unrelated", "A calm paragraph about the visitor gallery and its opening hours."
    )
    result = answer_offline("quantum chromodynamics lattice", [document])

    assert "[1]" in result.text
    assert len(result.sources) == 1
    assert result.sources[0].n == 1
    assert any("weak" in warning.lower() for warning in result.warnings)


def test_online_answer_validates_and_deduplicates_citations() -> None:
    document = _document(
        "mirror",
        "The primary mirror is made of borosilicate glass and cleaned with CO2 snow.",
    )
    fake = FakeListChatModel(responses=["The mirror is glass [1][1]. It is cleaned with snow [1]."])

    result = answer_with_llm("What is the mirror made of?", [document], fake)

    assert result.mode == "llm"
    assert result.model
    assert result.retrieved_count == 1
    assert [source.n for source in result.sources] == [1]
    assert result.text.startswith("The mirror is glass")


def test_online_answer_warns_about_missing_citation_number() -> None:
    document = _document("mirror", "The primary mirror is made of glass.")
    fake = FakeListChatModel(responses=["The mirror is glass [1]. A second claim [9]."])

    result = answer_with_llm("What is the mirror made of?", [document], fake)

    assert [source.n for source in result.sources] == [1]
    assert any("missing source [9]" in warning for warning in result.warnings)


def test_online_answer_without_citations_warns() -> None:
    document = _document("mirror", "The primary mirror is made of glass.")
    fake = FakeListChatModel(responses=["I could not find a supported answer."])

    result = answer_with_llm("What is the mirror made of?", [document], fake)

    assert result.sources == []
    assert any("no citations" in warning for warning in result.warnings)


def test_online_answer_without_documents_returns_message() -> None:
    fake = FakeListChatModel(responses=["should not be called"])
    result = answer_with_llm("Anything?", [], fake)
    assert result.retrieved_count == 0
    assert result.sources == []
    assert "No relevant passages found" in result.text


def test_prompt_contains_grounding_rules_and_numbered_context() -> None:
    document = _document(
        "mirror",
        "The primary mirror is made of borosilicate glass.",
    )
    chain = build_llm_chain(FakeListChatModel(responses=["ok"]))
    prompt = chain.steps[0]
    messages = prompt.format_messages(
        context=format_context([document]), question="What is the mirror made of?"
    )

    system = str(messages[0].content)
    human = str(messages[1].content)

    assert "ONLY the numbered context" in system
    assert "[n]" in system
    assert "insufficient" in system
    assert "[1] The Meridian Telescope (corpus/telescope.md)" in human
    assert "What is the mirror made of?" in human


def test_llm_errors_are_wrapped_with_offline_hint() -> None:
    document = _document("mirror", "The primary mirror is made of glass.")

    def explode(_value: object) -> None:
        raise RuntimeError("service unavailable")

    with pytest.raises(CiteWiseError) as excinfo:
        answer_with_llm("What?", [document], RunnableLambda(explode))

    assert excinfo.value.exit_code == 1
    assert "LLM request failed" in str(excinfo.value)
    assert "--offline" in str(excinfo.value)


def test_answer_question_dispatches_modes() -> None:
    document = _document("mirror", "The primary mirror is made of glass.")
    offline = answer_question("What is the mirror made of?", [document], offline=True)
    assert isinstance(offline, AnswerResult)
    assert offline.mode == "offline-extractive"

    fake = FakeListChatModel(responses=["Glass [1]."])
    online = answer_question("What is the mirror made of?", [document], offline=False, llm=fake)
    assert online.mode == "llm"
    assert [source.n for source in online.sources] == [1]
