"""High-level pipeline that ties loading, retrieval, and generation together."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunker import TextChunker
from .config import Config
from .embedder import Embedder, HashingEmbedder, LocalEmbedder, OpenAIEmbedder
from .llm import LLM, MockLLM, OllamaLLM, OpenAILLM
from .loader import DocumentLoader
from .retriever import Retriever
from .store import InMemoryVectorStore
from .types import Answer, Document


@dataclass
class IngestReport:
    """Summary of an ingestion run."""

    documents: int
    chunks: int
    replaced: int = 0
    dimension_changed: bool = False


class RAGPipeline:
    """End-to-end RAG pipeline: ingest documents once, then answer questions.

    The embedder and LLM backends are chosen from :class:`Config`:
    ``openai``/``local``/``hashing`` and ``openai``/``deepseek``/``ollama``/``mock``.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        self.store = InMemoryVectorStore.load(config.index_dir)
        self.embedder = self._build_embedder(config)
        self.llm = self._build_llm(config)
        self.retriever = Retriever(self.embedder, self.store, self.chunker, top_k=config.top_k)

    @staticmethod
    def _build_embedder(config: Config) -> Embedder:
        if config.embedder == "hashing":
            return HashingEmbedder()
        if config.embedder == "local":
            return LocalEmbedder(config.local_embedding_model)
        return OpenAIEmbedder(config.embedding_model, config.api_key, config.base_url)

    @staticmethod
    def _build_llm(config: Config) -> LLM:
        if config.llm == "mock":
            return MockLLM()
        if config.llm == "ollama":
            return OllamaLLM(config.ollama_model, config.ollama_base_url)
        if config.llm == "deepseek":
            # DeepSeek is OpenAI-compatible but has no embeddings endpoint.
            return OpenAILLM(config.deepseek_model, config.api_key, config.deepseek_base_url)
        return OpenAILLM(config.llm_model, config.api_key, config.base_url)

    def ingest(self, paths: list[str | Path]) -> IngestReport:
        """Load documents at ``paths``, then index them (see ``ingest_documents``)."""
        documents: list[Document] = []
        for path in paths:
            documents.extend(self.loader.load_path(path))
        return self.ingest_documents(documents)

    def ingest_documents(self, documents: list[Document]) -> IngestReport:
        """Chunk, embed, and persist ``documents``.

        Re-indexing a source replaces its existing chunks (incremental update)
        instead of duplicating them. If the embedding model changed, the old
        index is rebuilt from scratch.
        """
        sources = {document.source for document in documents}
        existing = self.store.sources()
        replaced = len(sources & existing)
        old_dimension = self.store.dimension
        for source in sources:
            self.store.remove_source(source)

        chunks = self.retriever.index(documents)
        dimension_changed = old_dimension is not None and self.store.dimension != old_dimension

        self.store.save(self.config.index_dir)
        return IngestReport(
            documents=len(documents),
            chunks=chunks,
            replaced=replaced,
            dimension_changed=dimension_changed,
        )

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
            "embedder": self.config.embedder,
            "llm": self.config.llm,
        }
