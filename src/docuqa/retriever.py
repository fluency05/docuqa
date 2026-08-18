"""Retrieval: indexing documents and searching for relevant chunks."""

from __future__ import annotations

from .chunker import TextChunker
from .embedder import Embedder
from .store import VectorStore
from .types import Document, SearchResult


class Retriever:
    """Coordinates chunking, embedding, storage, and querying."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chunker: TextChunker,
        top_k: int = 4,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self.top_k = top_k

    def index(self, documents: list[Document]) -> int:
        """Chunk and embed documents, adding them to the store. Returns chunk count."""
        indexed = 0
        for document in documents:
            chunks = self.chunker.chunk(document)
            if not chunks:
                continue
            vectors = self.embedder.embed([chunk.text for chunk in chunks])
            self.store.add(chunks, vectors)
            indexed += len(chunks)
        return indexed

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(query)
        hits = self.store.search(query_vector, top_k or self.top_k)
        return [SearchResult(chunk=chunk, score=score) for chunk, score in hits]
