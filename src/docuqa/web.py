"""FastAPI web application for docuqa.

The HTML/JS/CSS frontend is a single self-contained file (``web/index.html``),
so the whole UI ships without any build step or static-file packaging.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .pipeline import RAGPipeline

_INDEX_FILE = Path(__file__).with_name("web") / "index.html"


class _AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class _IngestRequest(BaseModel):
    path: str


def create_app(pipeline: RAGPipeline) -> FastAPI:
    """Build the FastAPI app around an existing :class:`RAGPipeline`."""
    app = FastAPI(title="docuqa", description="A lightweight RAG document Q&A assistant.")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_FILE.read_text(encoding="utf-8")

    @app.get("/api/stats")
    def stats() -> dict:
        return pipeline.stats()

    @app.post("/api/ask")
    def ask(request: _AskRequest) -> dict:
        answer = pipeline.ask(request.question, top_k=request.top_k)
        return {
            "question": answer.question,
            "answer": answer.answer,
            "sources": [
                {"source": source.chunk.source, "score": source.score, "text": source.chunk.text}
                for source in answer.sources
            ],
        }

    @app.post("/api/ingest")
    def ingest(request: _IngestRequest) -> dict:
        report = pipeline.ingest([request.path])
        return {
            "documents": report.documents,
            "chunks": report.chunks,
            "replaced": report.replaced,
            "rebuilt": report.dimension_changed,
        }

    return app
