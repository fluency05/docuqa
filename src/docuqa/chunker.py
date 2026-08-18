"""Split documents into overlapping, embedding-friendly text chunks."""

from __future__ import annotations

import re

from .types import Chunk, Document

# A paragraph boundary: one or more blank lines.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

# Characters that end a sentence in English and Chinese.
_SENTENCE_TERMINATORS = ".!?。！？;；"

# How far back from the hard cut to look for a sentence boundary.
_SENTENCE_LOOKBACK = 100


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
        chunks: list[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                end = self._sentence_boundary(text, start, end)
            chunks.append(text[start:end])
            start = end
        return chunks

    @staticmethod
    def _sentence_boundary(text: str, start: int, end: int) -> int:
        """Return a cut position at (or just after) a sentence terminator.

        Prefer breaking on a sentence boundary near ``end`` so chunks do not cut
        mid-sentence; fall back to ``end`` when none exists in the window.
        """
        window_start = max(start, end - _SENTENCE_LOOKBACK)
        for index in range(end - 1, window_start - 1, -1):
            if text[index] in _SENTENCE_TERMINATORS:
                return index + 1
        return end

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for chunk in chunks[1:]:
            result.append(result[-1][-self.overlap :] + chunk)
        return result
