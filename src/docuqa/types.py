"""Shared data types for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A single loaded document."""

    source: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A contiguous piece of text that can be embedded and retrieved."""

    id: str
    text: str
    source: str
    index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A retrieved chunk together with its similarity score."""

    chunk: Chunk
    score: float


@dataclass
class Answer:
    """The final answer plus the sources used to produce it."""

    question: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
