"""Command-line interface for docuqa."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .config import Config
from .pipeline import RAGPipeline
from .types import Answer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="docuqa — a lightweight Retrieval-Augmented Generation (RAG) document Q&A assistant.",
)
console = Console()

_OFFLINE_HELP = "Run offline with a deterministic demo embedder and mock LLM (no API key needed)."


def _build_pipeline(offline: bool) -> RAGPipeline:
    config = Config.from_env()
    if not offline and not config.api_key:
        console.print(
            "[red]OPENAI_API_KEY is not set.[/red]\n"
            "Set it via the OPENAI_API_KEY environment variable or a [bold].env[/bold] file, "
            "or pass [bold]--offline[/bold] to try a quick offline demo."
        )
        raise typer.Exit(code=1)
    return RAGPipeline(config, offline=offline)


def _print_sources(answer: Answer) -> None:
    table = Table(title="Sources", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Source")
    table.add_column("Score", justify="right")
    for index, result in enumerate(answer.sources, 1):
        table.add_row(str(index), result.chunk.source, f"{result.score:.3f}")
    console.print(table)


def _print_stats(pipeline: RAGPipeline) -> None:
    table = Table(title="Index stats", show_header=True, header_style="bold")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key, value in pipeline.stats().items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def ingest(
    paths: list[str] = typer.Argument(..., help="Files or directories to index."),
    offline: bool = typer.Option(False, "--offline", help=_OFFLINE_HELP),
) -> None:
    """Load documents, chunk them, embed them, and save the search index."""
    pipeline = _build_pipeline(offline)
    report = pipeline.ingest(paths)
    console.print(
        f"[green]Indexed[/green] {report.chunks} chunks from {report.documents} "
        f"document(s) into {pipeline.config.index_dir}"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Number of chunks to retrieve."),
    offline: bool = typer.Option(False, "--offline", help=_OFFLINE_HELP),
) -> None:
    """Ask a single question against the indexed documents."""
    pipeline = _build_pipeline(offline)
    answer = pipeline.ask(question, top_k=top_k)
    console.print(Panel(answer.answer, title="Answer", border_style="cyan"))
    if answer.sources:
        _print_sources(answer)


@app.command()
def chat(
    offline: bool = typer.Option(False, "--offline", help=_OFFLINE_HELP),
) -> None:
    """Start an interactive Q&A session (type 'exit' to quit)."""
    pipeline = _build_pipeline(offline)
    console.print("[bold]docuqa chat[/bold] — ask questions about your indexed documents.")
    console.print(
        "Type [bold]exit[/bold] or [bold]quit[/bold] to leave, [bold]stats[/bold] for index info."
    )
    while True:
        question = Prompt.ask("[green]You[/green]").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question.lower() == "stats":
            _print_stats(pipeline)
            continue
        answer = pipeline.ask(question)
        console.print(Panel(answer.answer, title="Assistant", border_style="cyan"))
        if answer.sources:
            _print_sources(answer)


@app.command()
def stats() -> None:
    """Show information about the current index."""
    config = Config.from_env()
    pipeline = RAGPipeline(config, offline=True)
    _print_stats(pipeline)


if __name__ == "__main__":
    app()
