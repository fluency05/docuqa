"""Split documents into overlapping, embedding-friendly text chunks."""

from __future__ import annotations

import re

from .types import Chunk, Document

# A paragraph boundary: one or more blank lines.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class TextChunker:
    """Split text into chunks of at most ``chunk_size`` characters.

    Chunking is paragraph-aware: paragraphs are packed together until the next
    one would overflow the budget, and paragraphs longer than ``chunk_size`` are
    broken with a sliding window. Consecutive chunks can share ``overlap``
    characters of context.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split_pieces(document.content)
        raw_chunks = self._pack(pieces)
        final_chunks = self._apply_overlap(raw_chunks)
        return [
            Chunk(
                id=f"{document.source}#{index}",
                text=text,
                source=document.source,
                index=index,
                metadata=dict(document.metadata),
            )
            for index, text in enumerate(final_chunks)
        ]

    @staticmethod
    def _split_pieces(text: str) -> list[str]:
        return [piece.strip() for piece in _PARAGRAPH_SPLIT.split(text) if piece.strip()]

    def _pack(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        buffer = ""
        for piece in pieces:
            if len(piece) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split_long(piece))
                continue
            candidate = f"{buffer} {piece}" if buffer else piece
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                chunks.append(buffer)
                buffer = piece
        if buffer:
            chunks.append(buffer)
        return chunks

    def _split_long(self, text: str) -> list[str]:
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for chunk in chunks[1:]:
            result.append(result[-1][-self.overlap :] + chunk)
        return result
