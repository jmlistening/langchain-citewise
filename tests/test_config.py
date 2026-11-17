from __future__ import annotations

import os
from pathlib import Path

from citewise.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    get_api_key,
    get_base_url,
    get_embedding_model,
    get_model,
    load_dotenv,
)


def _clear_environment(monkeypatch) -> None:
    for name in (
        "CITEWISE_API_KEY",
        "OPENAI_API_KEY",
        "CITEWISE_BASE_URL",
        "CITEWISE_MODEL",
        "CITEWISE_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_load_dotenv_reads_only_citewise_keys_and_does_not_override(
    tmp_path: Path, monkeypatch
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.setenv("CITEWISE_MODEL", "already-set")

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# a comment",
                'CITEWISE_API_KEY="test-quoted-key"',
                "OPENAI_API_KEY=ignored-by-the-loader",
                "CITEWISE_MODEL=from-file",
                "export CITEWISE_BASE_URL=http://localhost:11434/v1",
                "UNRELATED_VALUE=ignored",
                "",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_dotenv(env_file)

    assert loaded == ["CITEWISE_API_KEY", "CITEWISE_BASE_URL"]
    assert os.environ["CITEWISE_API_KEY"] == "test-quoted-key"
    assert os.environ["CITEWISE_MODEL"] == "already-set"
    assert os.environ["CITEWISE_BASE_URL"] == "http://localhost:11434/v1"
    assert "OPENAI_API_KEY" not in os.environ
    assert "UNRELATED_VALUE" not in os.environ

    assert get_api_key() == "test-quoted-key"
    assert get_base_url() == "http://localhost:11434/v1"
    assert get_model() == "already-set"

    # load_dotenv writes directly to os.environ, outside monkeypatch's undo log.
    os.environ.pop("CITEWISE_API_KEY", None)
    os.environ.pop("CITEWISE_BASE_URL", None)


def test_load_dotenv_missing_file(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "missing.env") == []


def test_environment_defaults_and_key_precedence(monkeypatch) -> None:
    _clear_environment(monkeypatch)

    assert get_api_key() is None
    assert get_base_url() is None
    assert get_model() == DEFAULT_MODEL
    assert get_embedding_model() == DEFAULT_EMBEDDING_MODEL

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert get_api_key() == "openai-key"

    monkeypatch.setenv("CITEWISE_API_KEY", "citewise-key")
    assert get_api_key() == "citewise-key"

    monkeypatch.setenv("CITEWISE_MODEL", "")
    assert get_model() == DEFAULT_MODEL
