"""Rich terminal rendering helpers (never used in JSON output mode)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .answer import AnswerResult


def _console(stderr: bool = False) -> Console:
    return Console(stderr=stderr, highlight=False)


def notice(message: str) -> None:
    """Print a diagnostic notice to stderr."""

    _console(stderr=True).print(Text(message, style="yellow"))


def print_json(payload: dict[str, Any]) -> None:
    """Write machine-readable JSON to stdout only.

    ASCII escaping keeps the output valid on consoles with legacy code pages;
    JSON parsers decode it transparently.
    """

    print(json.dumps(payload, ensure_ascii=True, indent=2))


def print_banner(message: str) -> None:
    """Print a short styled banner to stdout."""

    _console().print(Text(message, style="bold cyan"))


def print_ingest_summary(manifest: dict[str, Any], index_dir: Path) -> None:
    """Render the ingest summary."""

    console = _console()
    embedding = manifest.get("embedding", {})
    provider = embedding.get("provider", "unknown")
    dim = embedding.get("dim", "?")
    model = embedding.get("model")
    detail = f"{provider} (dim {dim}"
    if model:
        detail += f", model {model}"
    detail += ")"

    console.print(
        Text(
            f"Indexed {manifest.get('chunk_count', 0)} chunks from "
            f"{manifest.get('file_count', 0)} files into {index_dir}",
            style="bold green",
        )
    )
    console.print(Text(f"Embedding: {detail}"))
    console.print(Text(f"Chunks:    {manifest.get('chunk_count', 0)}"))
    console.print(Text('Next:      citewise ask "your question" --offline'))


def _print_sources(result: AnswerResult) -> None:
    console = _console()
    if not result.sources:
        console.print(Text("Sources: none", style="dim"))
        return

    console.print(Text("Sources", style="bold"))
    for source in result.sources:
        console.print(
            Text.assemble(
                (f"[{source.n}] ", "cyan"),
                (source.source, "magenta"),
                (" - ", "dim"),
                (source.title, "bold"),
                (f"  (rrf {source.rrf:.4f})", "dim"),
            )
        )
        limit = max(20, console.width - 6)
        snippet = source.snippet
        if len(snippet) > limit:
            snippet = snippet[: limit - 3].rstrip() + "..."
        console.print(Text(f"    {snippet}", style="dim"))


def print_answer(
    result: AnswerResult,
    *,
    context_documents: list[Any] | None = None,
) -> None:
    """Render an answer, its sources, warnings, and optionally raw context."""

    console = _console()
    title = "Offline extractive answer" if result.mode == "offline-extractive" else "LLM answer"
    console.print(Panel(Text(result.text), title=title, border_style="cyan"))
    _print_sources(result)

    for warning in result.warnings:
        notice(f"warning: {warning}")

    if context_documents is not None:
        console.print(Text("\nRetrieved context", style="bold"))
        for number, document in enumerate(context_documents, start=1):
            source = document.metadata.get("source", "unknown")
            doc_title = document.metadata.get("title", "")
            label = f"[{number}] {doc_title} ({source})"
            console.print(Panel(Text(document.page_content), title=Text(label)))


def print_eval_report(report: dict[str, Any]) -> None:
    """Render the evaluation table and summary."""

    console = _console()
    table = Table(title=f"Retrieval evaluation (k={report['k']})")
    table.add_column("Question", overflow="fold")
    table.add_column("Hit", justify="center")
    table.add_column("First rank", justify="right")
    table.add_column("RR", justify="right")
    table.add_column("Term coverage", justify="right")

    for row in report["questions"]:
        first_rank = row.get("first_relevant_rank")
        coverage = row.get("expected_term_coverage")
        table.add_row(
            row["question"],
            "yes" if row["hit"] else "no",
            str(first_rank) if first_rank is not None else "-",
            f"{row['reciprocal_rank']:.4f}",
            f"{coverage:.4f}" if coverage is not None else "-",
        )

    console.print(table)
    metrics = report["metrics"]
    coverage_text = metrics.get("answer_term_coverage")
    coverage_display = f"{coverage_text:.4f}" if coverage_text is not None else "n/a"
    console.print(
        Text(
            f"hit@{report['k']} = {metrics['hit_at_k']:.4f}   "
            f"MRR = {metrics['mrr']:.4f}   "
            f"answer term coverage = {coverage_display}",
            style="bold green",
        )
    )
