"""High-level pipeline that ties loading, retrieval, and generation together."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunker import TextChunker
from .config import Config
from .embedder import HashingEmbedder, OpenAIEmbedder
from .llm import MockLLM, OpenAILLM
from .loader import DocumentLoader
from .retriever import Retriever
from .store import InMemoryVectorStore
from .types import Answer


@dataclass
class IngestReport:
    """Summary of an ingestion run."""

    documents: int
    chunks: int


class RAGPipeline:
    """End-to-end RAG pipeline: ingest documents once, then answer questions."""

    def __init__(self, config: Config, *, offline: bool = False) -> None:
        self.config = config
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        self.store = InMemoryVectorStore.load(config.index_dir)

        if offline:
            self.embedder = HashingEmbedder()
            self.llm = MockLLM()
        else:
            self.embedder = OpenAIEmbedder(config.embedding_model, config.api_key, config.base_url)
            self.llm = OpenAILLM(config.llm_model, config.api_key, config.base_url)

        self.retriever = Retriever(self.embedder, self.store, self.chunker, top_k=config.top_k)

    def ingest(self, paths: list[str | Path]) -> IngestReport:
        """Load, chunk, embed, and persist documents found at ``paths``."""
        documents: list = []
        for path in paths:
            documents.extend(self.loader.load_path(path))
        chunks = self.retriever.index(documents)
        self.store.save(self.config.index_dir)
        return IngestReport(documents=len(documents), chunks=chunks)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        """Answer a single question using the current index."""
        results = self.retriever.search(question, top_k)
        text = self.llm.answer(question, results)
        return Answer(question=question, answer=text, sources=results)

    def stats(self) -> dict:
        """Return a snapshot of the current index and configuration."""
        return {
            "index_dir": str(self.config.index_dir),
            "chunks": len(self.store),
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "top_k": self.config.top_k,
        }
