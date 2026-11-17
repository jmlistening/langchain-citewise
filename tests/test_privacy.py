from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    ".citewise",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}

# Build the sensitive prefixes from pieces so this test file itself never
# contains a literal secret-like token that a simple grep would flag.
_WINDOWS_USER_PATH = "C:" + chr(92) + "Users"
_GITHUB_TOKEN_PREFIX = "ghp" + "_"
_GITHUB_PAT_PREFIX = "github" + "_pat" + "_"
_SECRET_KEY_RE = re.compile(r"\b" + "sk" + r"-[A-Za-z0-9_-]{8,}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ABSOLUTE_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\")


def _shipped_files() -> Iterator[Path]:
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        yield path


def test_no_env_file_in_repository_root() -> None:
    assert not (REPO_ROOT / ".env").exists()


def test_gitignore_covers_secrets_environment_and_index() -> None:
    lines = {
        line.strip() for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert ".env" in lines
    assert ".venv/" in lines
    assert ".citewise/" in lines


def test_no_secret_like_or_personal_content_is_shipped() -> None:
    hits: list[str] = []
    for path in _shipped_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        relative = path.relative_to(REPO_ROOT)
        for label, needle in (
            ("windows user path", _WINDOWS_USER_PATH),
            ("github token prefix", _GITHUB_TOKEN_PREFIX),
            ("github pat prefix", _GITHUB_PAT_PREFIX),
        ):
            if needle in text:
                hits.append(f"{relative}: {label}")
        if _SECRET_KEY_RE.search(text):
            hits.append(f"{relative}: key-like string")
        if _EMAIL_RE.search(text):
            hits.append(f"{relative}: email-like string")

    assert hits == []


def test_no_absolute_windows_paths_in_shipped_examples() -> None:
    examples = REPO_ROOT / "examples"
    assert examples.is_dir()
    hits = []
    for path in examples.rglob("*"):
        if path.is_file() and _ABSOLUTE_WINDOWS_PATH_RE.search(
            path.read_text(encoding="utf-8", errors="replace")
        ):
            hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []


def test_readme_language_switches_exist() -> None:
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (REPO_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert "[简体中文](README.zh-CN.md)" in english
    assert "[English](README.md)" in chinese
