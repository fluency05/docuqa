"""Vector storage and similarity search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .types import Chunk


class VectorStore(Protocol):
    """Structural interface every vector store implements."""

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, query: list[float], top_k: int) -> list[tuple[Chunk, float]]: ...

    def save(self, directory: Path) -> None: ...

    def __len__(self) -> int: ...


class InMemoryVectorStore:
    """A simple, dependency-light vector store backed by NumPy.

    Vectors are row-normalized on insert, so similarity search is a single
    matrix-vector dot product (cosine similarity). The index persists as a
    ``vectors.npy`` file plus a ``chunks.json`` metadata sidecar.
    """

    def __init__(
        self, chunks: list[Chunk] | None = None, vectors: np.ndarray | None = None
    ) -> None:
        self._chunks: list[Chunk] = chunks or []
        self._vectors: np.ndarray | None = vectors  # shape (n, d), rows normalized

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        if self.dimension is not None and self.dimension != normalized.shape[1]:
            # The existing index was built with a different embedder, so its
            # vectors are incompatible. Start fresh rather than crash.
            self.clear()
        if self._vectors is None:
            self._vectors = normalized
        else:
            self._vectors = np.vstack([self._vectors, normalized])
        self._chunks.extend(chunks)

    def search(self, query: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        if self._vectors is None or not self._chunks:
            return []
        q = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm != 0:
            q = q / norm
        scores = self._vectors @ q
        top_k = min(top_k, len(self._chunks))
        indices = np.argsort(-scores)[:top_k]
        return [(self._chunks[int(i)], float(scores[int(i)])) for i in indices]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(directory / "vectors.npy", self._vectors)
        payload = [self._chunk_to_dict(chunk) for chunk in self._chunks]
        (directory / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "InMemoryVectorStore":
        vectors_path = directory / "vectors.npy"
        chunks_path = directory / "chunks.json"
        vectors = np.load(vectors_path) if vectors_path.exists() else None
        chunks: list[Chunk] = []
        if chunks_path.exists():
            raw = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = [cls._chunk_from_dict(item) for item in raw]
        return cls(chunks=chunks, vectors=vectors)

    def remove_source(self, source: str) -> int:
        """Remove every chunk whose source equals ``source``. Returns count removed."""
        if self._vectors is None or not self._chunks:
            return 0
        keep = [chunk.source != source for chunk in self._chunks]
        removed = len(self._chunks) - sum(keep)
        if removed:
            keep_mask = np.asarray(keep, dtype=bool)
            self._chunks = [chunk for chunk, keep_it in zip(self._chunks, keep) if keep_it]
            self._vectors = self._vectors[keep_mask] if self._chunks else None
        return removed

    @property
    def dimension(self) -> int | None:
        """Width of the stored vectors, or ``None`` when the store is empty."""
        if self._vectors is None or self._vectors.shape[0] == 0:
            return None
        return int(self._vectors.shape[1])

    def clear(self) -> None:
        """Drop every chunk and vector."""
        self._chunks = []
        self._vectors = None

    def sources(self) -> set[str]:
        """Return the set of source paths currently indexed."""
        return {chunk.source for chunk in self._chunks}

    def __len__(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> dict:
        return {
            "id": chunk.id,
            "text": chunk.text,
            "source": chunk.source,
            "index": chunk.index,
            "metadata": chunk.metadata,
        }

    @staticmethod
    def _chunk_from_dict(data: dict) -> Chunk:
        return Chunk(
            id=data["id"],
            text=data["text"],
            source=data["source"],
            index=data["index"],
            metadata=data.get("metadata", {}),
        )
