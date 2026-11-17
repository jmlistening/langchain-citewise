"""Command-line interface for CiteWise."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .answer import answer_offline, answer_question
from .config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EVAL_K,
    DEFAULT_INDEX_DIR,
    DEFAULT_TOP_K,
    CiteWiseError,
    UsageError,
    get_api_key,
    get_base_url,
    get_model,
    load_dotenv,
)
from .corpus import chunk_documents, load_documents
from .evaluation import evaluate, load_eval_file
from .output import (
    notice,
    print_answer,
    print_banner,
    print_eval_report,
    print_ingest_summary,
    print_json,
)
from .retriever import HybridRetriever
from .store import IndexContents, build_index, load_index


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``citewise``."""

    parser = argparse.ArgumentParser(
        prog="citewise",
        description=(
            "Citation-grounded RAG over your documents: hybrid retrieval, "
            "offline-first demos, optional OpenAI-compatible models."
        ),
    )
    parser.add_argument("--version", action="version", version=f"citewise {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Chunk and index a file or directory")
    ingest.add_argument("source", help="File or directory to ingest")
    ingest.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_DIR),
        help=f"Index directory (default: {DEFAULT_INDEX_DIR})",
    )
    ingest.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters"
    )
    ingest.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in characters",
    )
    ingest.add_argument(
        "--embeddings",
        choices=("hashing", "openai"),
        default="hashing",
        help="Embedding provider (default: hashing, local and offline)",
    )
    ingest.add_argument(
        "--embedding-model",
        default=None,
        help="OpenAI embedding model (only with --embeddings openai)",
    )
    ingest.set_defaults(handler=cmd_ingest)

    ask = subparsers.add_parser("ask", help="Ask one question against the index")
    ask.add_argument("question", help="Question to answer")
    ask.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_DIR),
        help=f"Index directory (default: {DEFAULT_INDEX_DIR})",
    )
    ask.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Number of passages to retrieve")
    ask.add_argument(
        "--offline",
        action="store_true",
        help="Force the offline extractive answerer (no API key needed)",
    )
    ask.add_argument("--show-context", action="store_true", help="Render all retrieved passages")
    ask.add_argument("--json", dest="json_output", action="store_true", help="Emit JSON to stdout")
    ask.add_argument("--model", default=None, help="Chat model name (overrides CITEWISE_MODEL)")
    ask.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (overrides CITEWISE_BASE_URL)",
    )
    ask.set_defaults(handler=cmd_ask)

    chat = subparsers.add_parser("chat", help="Interactive chat against the index")
    chat.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_DIR),
        help=f"Index directory (default: {DEFAULT_INDEX_DIR})",
    )
    chat.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Number of passages to retrieve")
    chat.add_argument(
        "--offline",
        action="store_true",
        help="Force the offline extractive answerer (no API key needed)",
    )
    chat.add_argument("--model", default=None, help="Chat model name (overrides CITEWISE_MODEL)")
    chat.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (overrides CITEWISE_BASE_URL)",
    )
    chat.set_defaults(handler=cmd_chat)

    eval_parser = subparsers.add_parser(
        "eval", help="Score retrieval against a JSONL question file"
    )
    eval_parser.add_argument("eval_file", help="JSONL file with labeled questions")
    eval_parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_DIR),
        help=f"Index directory (default: {DEFAULT_INDEX_DIR})",
    )
    eval_parser.add_argument("--k", type=int, default=DEFAULT_EVAL_K, help="Retrieval cutoff k")
    eval_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Emit JSON to stdout"
    )
    eval_parser.set_defaults(handler=cmd_eval)

    return parser


def _load_index_or_fail(index_dir: str) -> IndexContents:
    return load_index(Path(index_dir))


def _build_retriever(index: IndexContents, k: int) -> HybridRetriever:
    return HybridRetriever(
        documents=index.documents,
        vector_store=index.vector_store,
        top_k=k,
    )


def _build_llm(args: argparse.Namespace) -> Any:
    api_key = get_api_key()
    if api_key is None:
        raise UsageError(
            "no API key configured; set CITEWISE_API_KEY or OPENAI_API_KEY, or pass --offline."
        )
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise UsageError(
            "online mode requires the 'openai' extra: run "
            "`uv sync --extra openai`, or pass --offline."
        ) from exc

    kwargs: dict[str, Any] = {
        "model": args.model or get_model(),
        "api_key": api_key,
        "temperature": 0.2,
    }
    base_url = args.base_url or get_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return ChatOpenAI(**kwargs)
    except Exception as exc:
        raise CiteWiseError(f"could not configure the chat model: {exc}") from exc


def _resolve_answerer(args: argparse.Namespace) -> tuple[bool, Any | None]:
    """Return ``(offline, llm)``, falling back to offline when no key is set."""

    if args.offline:
        return True, None
    if get_api_key() is None:
        notice(
            "No API key found; using offline extractive mode. Set "
            "CITEWISE_API_KEY or OPENAI_API_KEY for LLM answers."
        )
        return True, None
    return False, _build_llm(args)


def _document_payload(document: Any) -> dict[str, Any]:
    return {
        "source": document.metadata.get("source"),
        "title": document.metadata.get("title"),
        "chunk_id": document.metadata.get("chunk_id"),
        "vector_rank": document.metadata.get("vector_rank"),
        "bm25_rank": document.metadata.get("bm25_rank"),
        "rrf": document.metadata.get("rrf"),
        "text": document.page_content,
    }


def cmd_ingest(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser()

    def warn(message: str) -> None:
        print(f"citewise: warning: {message}", file=sys.stderr)

    documents = load_documents(source, warn=warn)
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if not chunks:
        raise UsageError(
            f"no supported, non-empty documents found under {source}; "
            "supported types: .md, .markdown, .txt, .pdf"
        )

    index_dir = Path(args.index)
    manifest = build_index(
        index_dir,
        chunks,
        provider=args.embeddings,
        model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print_ingest_summary(manifest, index_dir)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    question = args.question.strip()
    if not question:
        raise UsageError("question must not be empty")
    if args.k <= 0:
        raise UsageError("--k must be greater than zero")

    index = _load_index_or_fail(args.index)
    retriever = _build_retriever(index, args.k)
    documents = retriever.invoke(question)
    offline, llm = _resolve_answerer(args)
    result = answer_question(question, documents, offline=offline, llm=llm)

    if args.json_output:
        payload: dict[str, Any] = {
            "question": question,
            "k": args.k,
            **result.to_dict(),
        }
        if args.show_context:
            payload["context"] = [_document_payload(document) for document in documents]
        print_json(payload)
        return 0

    print_answer(
        result,
        context_documents=documents if args.show_context else None,
    )
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    if args.k <= 0:
        raise UsageError("--k must be greater than zero")
    index = _load_index_or_fail(args.index)
    retriever = _build_retriever(index, args.k)
    offline, llm = _resolve_answerer(args)

    mode = "offline extractive" if offline else "LLM"
    print_banner(f"CiteWise chat ({mode}). Type 'exit' or press Ctrl-D to quit.")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        documents = retriever.invoke(question)
        result = answer_question(question, documents, offline=offline, llm=llm)
        print_answer(result)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if args.k <= 0:
        raise UsageError("--k must be greater than zero")
    index = _load_index_or_fail(args.index)
    retriever = _build_retriever(index, args.k)
    questions = load_eval_file(Path(args.eval_file))
    report = evaluate(
        questions,
        retriever,
        k=args.k,
        answer_fn=lambda question, documents: answer_offline(question, documents),
    )
    if args.json_output:
        print_json(report)
    else:
        print_eval_report(report)
    return 0


def _configure_stdio() -> None:
    """Make stdout/stderr tolerate characters outside the console code page.

    Windows consoles often use a legacy code page such as GBK or cp1252. Rich
    renders answers and source snippets to those streams, so a valid answer
    containing an emoji or another non-ASCII character would otherwise abort
    the command with ``UnicodeEncodeError`` after partial output.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (OSError, ValueError):
            continue


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Exit codes: 0 success, 2 usage/config/index errors, 1 runtime/API failures.
    """

    _configure_stdio()
    load_dotenv()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help / --version / argparse usage errors
        code = exc.code
        return code if isinstance(code, int) else 0

    try:
        return int(args.handler(args))
    except CiteWiseError as exc:
        print(f"citewise: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("citewise: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
