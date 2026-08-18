"""Text embedders for turning chunks and queries into vectors."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from openai import OpenAI


class Embedder(Protocol):
    """Structural interface every embedder implements."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""


class OpenAIEmbedder:
    """Embed text using any OpenAI-compatible embeddings API."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashingEmbedder:
    """Deterministic, offline embedder for demos and tests.

    It maps tokens to a fixed-size vector via a stable hash. The resulting
    vectors are **not** semantically meaningful — they exist only so the full
    pipeline can run without an API key. Use :class:`OpenAIEmbedder` for real use.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            value = int(digest, 16)
            index = value % self.dim
            sign = 1.0 if (value >> 8) % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
