from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from citewise.cli import main


def _write_corpus(root: Path) -> Path:
    corpus = root / "corpus"
    corpus.mkdir()
    (corpus / "alpha.md").write_text(
        "# Alpha Experiment\n\n"
        "The alpha experiment uses a detector to count charged particles. "
        "The detector is cooled to reduce noise.\n",
        encoding="utf-8",
    )
    (corpus / "beta.md").write_text(
        "# Beta Decay\n\nBeta decay emits electrons. The beta detector notes their energy.\n",
        encoding="utf-8",
    )
    return corpus


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    for name in (
        "CITEWISE_API_KEY",
        "OPENAI_API_KEY",
        "CITEWISE_BASE_URL",
        "CITEWISE_MODEL",
        "CITEWISE_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_ingest_then_ask_json_smoke(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = _write_corpus(workspace)

    assert main(["ingest", str(corpus)]) == 0
    ingest_out, _ = capsys.readouterr()
    assert "Indexed" in ingest_out

    manifest_path = workspace / ".citewise" / "manifest.json"
    assert (workspace / ".citewise" / "index.json").is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["file_count"] == 2
    assert manifest["sources"] == ["alpha.md", "beta.md"]
    assert manifest["embedding"] == {"provider": "hashing", "dim": 512, "model": None}
    assert manifest["chunk_count"] > 0

    assert main(["ask", "What does the alpha experiment use?", "--offline", "--json"]) == 0
    stdout, stderr = capsys.readouterr()
    payload = json.loads(stdout)
    assert payload["question"] == "What does the alpha experiment use?"
    assert payload["mode"] == "offline-extractive"
    assert payload["retrieved_count"] >= 1
    assert payload["sources"]
    assert payload["sources"][0]["source"] == "alpha.md"
    assert payload["sources"][0]["n"] == 1
    assert payload["text"]
    assert "[1]" in payload["text"]
    assert not stdout.lstrip().startswith("+")
    assert stderr == ""


def test_ask_without_index_fails_clearly(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = workspace / "missing-index"
    assert main(["ask", "Any question?", "--index", str(missing), "--offline"]) == 2
    _, stderr = capsys.readouterr()
    assert "No index found" in stderr
    assert "citewise ingest" in stderr


def test_ask_without_api_key_falls_back_to_offline(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_corpus(workspace)
    assert main(["ingest", "corpus"]) == 0
    capsys.readouterr()

    assert main(["ask", "What does the alpha experiment use?", "--json"]) == 0
    stdout, stderr = capsys.readouterr()
    payload = json.loads(stdout)
    assert payload["mode"] == "offline-extractive"
    assert "No API key found" in stderr


def test_ask_show_context_includes_retrieved_documents(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_corpus(workspace)
    assert main(["ingest", "corpus"]) == 0
    capsys.readouterr()

    assert (
        main(
            ["ask", "What does the alpha experiment use?", "--offline", "--json", "--show-context"]
        )
        == 0
    )
    stdout, _ = capsys.readouterr()
    payload = json.loads(stdout)
    assert payload["context"]
    assert payload["context"][0]["source"]
    assert payload["context"][0]["text"]


def test_ingest_empty_and_unsupported_sources_fail(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = workspace / "empty"
    empty.mkdir()
    assert main(["ingest", str(empty)]) == 2
    _, stderr = capsys.readouterr()
    assert "no supported, non-empty documents" in stderr

    unsupported = workspace / "notes.json"
    unsupported.write_text("{}", encoding="utf-8")
    assert main(["ingest", str(unsupported)]) == 2
    _, stderr = capsys.readouterr()
    assert "unsupported file type" in stderr


def test_ingest_openai_embeddings_requires_key(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_corpus(workspace)
    assert (
        main(
            [
                "ingest",
                "corpus",
                "--embeddings",
                "openai",
                "--index",
                str(workspace / "openai-index"),
            ]
        )
        == 2
    )
    _, stderr = capsys.readouterr()
    assert "CITEWISE_API_KEY" in stderr


def test_eval_json_output(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_corpus(workspace)
    assert main(["ingest", "corpus"]) == 0
    capsys.readouterr()

    eval_path = workspace / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "question": "What does the alpha experiment use?",
                "relevant_sources": ["alpha.md"],
                "expected_terms": ["detector"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["eval", str(eval_path), "--json"]) == 0
    stdout, stderr = capsys.readouterr()
    report = json.loads(stdout)
    assert report["k"] == 5
    assert report["metrics"]["hit_at_k"] == 1.0
    assert report["metrics"]["mrr"] == 1.0
    assert report["questions"][0]["expected_term_coverage"] == 1.0
    assert stderr == ""


def test_version_and_usage_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    stdout, _ = capsys.readouterr()
    assert "citewise" in stdout.lower()

    assert main([]) == 2
    _, stderr = capsys.readouterr()
    assert "usage" in stderr.lower()


def test_invalid_k_is_a_usage_error(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_corpus(workspace)
    assert main(["ingest", "corpus"]) == 0
    capsys.readouterr()

    assert main(["ask", "question", "--offline", "--k", "0"]) == 2
    _, stderr = capsys.readouterr()
    assert "--k must be greater than zero" in stderr


def test_chat_offline_exits_on_exit_command(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_corpus(workspace)
    assert main(["ingest", "corpus"]) == 0
    capsys.readouterr()

    inputs = iter(["What does the alpha experiment use?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert main(["chat", "--offline"]) == 0
    stdout, _ = capsys.readouterr()
    assert "CiteWise chat" in stdout
    assert "detector" in stdout.lower()


def test_ask_non_latin_question_returns_json_without_crashing(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_corpus(workspace)
    assert main(["ingest", "corpus"]) == 0
    capsys.readouterr()

    assert main(["ask", "望远镜的主镜是什么材料做的？", "--offline", "--json"]) == 0
    stdout, stderr = capsys.readouterr()
    payload = json.loads(stdout)
    assert payload["mode"] == "offline-extractive"
    assert payload["text"]
    assert payload["sources"] == []
    assert stderr == ""


def test_answer_renders_on_a_legacy_console_encoding(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = workspace / "corpus"
    corpus.mkdir()
    (corpus / "emoji.md").write_text(
        "# Emoji Doc\n\nThe telescope mirror is cleaned with CO2 snow. \U0001f600 "
        "The dome opens at dusk.\n",
        encoding="utf-8",
    )
    assert main(["ingest", "corpus"]) == 0

    stdout_buffer = io.BytesIO()
    stderr_buffer = io.BytesIO()
    stdout = io.TextIOWrapper(stdout_buffer, encoding="gbk", errors="strict", line_buffering=True)
    stderr = io.TextIOWrapper(stderr_buffer, encoding="gbk", errors="strict", line_buffering=True)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    assert main(["ask", "How is the mirror cleaned?", "--offline"]) == 0
    stdout.flush()
    stderr.flush()

    rendered = stdout_buffer.getvalue().decode("gbk")
    assert "Offline extractive answer" in rendered
    assert "Sources" in rendered


def test_ingest_fails_cleanly_when_index_path_is_a_file(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_corpus(workspace)
    blocker = workspace / "blocked-index"
    blocker.write_text("not a directory", encoding="utf-8")

    assert main(["ingest", "corpus", "--index", str(blocker)]) == 2
    _, stderr = capsys.readouterr()
    assert "could not write the index" in stderr
    assert "--index" in stderr
