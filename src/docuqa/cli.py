"""Command-line interface for docuqa."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .config import Config
from .evaluation import evaluate as run_evaluation
from .evaluation import load_cases
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
    if offline:
        config.embedder = "hashing"
        config.llm = "mock"
    if (config.embedder == "openai" or config.llm in {"openai", "deepseek"}) and not config.api_key:
        console.print(
            "[red]OPENAI_API_KEY is not set.[/red]\n"
            "Set it via OPENAI_API_KEY / a [bold].env[/bold] file, pass [bold]--offline[/bold] "
            "for a quick demo, or switch to a local backend "
            "([bold]DOCUQA_EMBEDDER=local[/bold], [bold]DOCUQA_LLM=ollama[/bold])."
        )
        raise typer.Exit(code=1)
    return RAGPipeline(config)


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
    replaced = f", {report.replaced} replaced" if report.replaced else ""
    console.print(
        f"[green]Indexed[/green] {report.chunks} chunks from {report.documents} "
        f"document(s){replaced} into {pipeline.config.index_dir}"
    )
    if report.dimension_changed:
        console.print(
            "[yellow]Note:[/yellow] the embedding model changed, "
            "so the index was rebuilt from scratch."
        )


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    top_k: int | None = typer.Option(None, "--top-k", help="Number of chunks to retrieve."),
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


@app.command("eval")
def eval_cmd(
    dataset: str = typer.Argument(..., help="Path to a JSON evaluation dataset."),
    top_k: int | None = typer.Option(None, "--top-k", help="Chunks to retrieve per query."),
    offline: bool = typer.Option(False, "--offline", help=_OFFLINE_HELP),
) -> None:
    """Evaluate retrieval quality (Recall@k and MRR) against a dataset."""
    pipeline = _build_pipeline(offline)
    cases = load_cases(dataset)
    report = run_evaluation(pipeline.retriever, cases, top_k or pipeline.config.top_k)

    table = Table(
        title=f"Retrieval evaluation (k={report.k})", show_header=True, header_style="bold"
    )
    table.add_column("Query")
    table.add_column("Hit", justify="center")
    table.add_column("Rank", justify="right")
    table.add_column("Top source")
    for result in report.results:
        top_source = result.retrieved[0] if result.retrieved else "-"
        table.add_row(
            result.query, "yes" if result.hit else "no", str(result.rank or "-"), top_source
        )
    console.print(table)
    console.print(
        f"[bold]Recall@{report.k}[/bold]: {report.recall_at_k:.3f}   "
        f"[bold]MRR[/bold]: {report.mrr:.3f}"
    )


@app.command()
def stats() -> None:
    """Show information about the current index."""
    config = Config.from_env()
    config.embedder = "hashing"
    config.llm = "mock"
    pipeline = RAGPipeline(config)
    _print_stats(pipeline)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface to bind."),
    port: int = typer.Option(8000, "--port", help="Port to bind."),
    offline: bool = typer.Option(False, "--offline", help=_OFFLINE_HELP),
) -> None:
    """Launch the browser-based web UI."""
    try:
        import uvicorn  # noqa: F401
        from .web import create_app
    except ImportError:
        console.print("[red]The web UI requires fastapi and uvicorn.[/red]")
        console.print('Install them with: pip install -e ".[web]"', markup=False)
        raise typer.Exit(code=1)
    pipeline = _build_pipeline(offline)
    app = create_app(pipeline)
    console.print(f"[green]Serving docuqa at[/green] http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    app()
