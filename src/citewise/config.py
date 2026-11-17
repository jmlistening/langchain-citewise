"""Configuration, environment handling, and shared errors for CiteWise."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_INDEX_DIR = Path(".citewise")
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 4
DEFAULT_EVAL_K = 5
DEFAULT_RRF_K = 60
DEFAULT_FETCH_K = 16
HASHING_DIM = 512

API_KEY_ENV_VARS = ("CITEWISE_API_KEY", "OPENAI_API_KEY")


class CiteWiseError(Exception):
    """An expected error with a user-facing message and a process exit code.

    Exit code 2 covers usage, configuration, and index problems; 1 covers
    runtime failures such as an API request that could not be completed.
    """

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class UsageError(CiteWiseError):
    """Invalid input, configuration, or index state (exit code 2)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=2)


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load ``CITEWISE_*`` keys from a ``.env`` file without overriding the environment.

    The parser is deliberately tiny: ``KEY=VALUE`` lines, optional ``export``
    prefix, ``#`` comments, and optional matching single or double quotes. Only
    keys starting with ``CITEWISE_`` are read, existing environment variables
    win, and values are never logged. Returns the names of the keys that were
    actually set (not their values).
    """

    env_path = Path(path) if path is not None else Path.cwd() / ".env"
    if not env_path.is_file():
        return []

    loaded: list[str] = []
    try:
        raw_text = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key.startswith("CITEWISE_"):
            continue
        if key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded


def get_api_key() -> str | None:
    """Return the first configured API key, or ``None`` when unset/blank."""

    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def get_base_url() -> str | None:
    """Return the configured OpenAI-compatible base URL, if any."""

    value = os.environ.get("CITEWISE_BASE_URL", "").strip()
    return value or None


def get_model() -> str:
    """Return the configured chat model name."""

    return os.environ.get("CITEWISE_MODEL", "").strip() or DEFAULT_MODEL


def get_embedding_model() -> str:
    """Return the configured OpenAI embedding model name."""

    return os.environ.get("CITEWISE_EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
