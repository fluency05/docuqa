"""Language models that turn retrieved context into a cited answer."""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from .types import SearchResult

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the user's question "
    "using ONLY the provided context. If the context does not contain the answer, "
    'say "I could not find the answer in the provided documents." '
    "Cite the sources you used with bracketed numbers like [1], [2]."
)


def build_prompt(question: str, results: list[SearchResult]) -> str:
    """Assemble the retrieval context and question into a single prompt."""
    blocks = [
        f"[{i}] Source: {result.chunk.source}\n{result.chunk.text}"
        for i, result in enumerate(results, 1)
    ]
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (with citations):"


class LLM(Protocol):
    """Structural interface every answer generator implements."""

    def answer(self, question: str, results: list[SearchResult]) -> str: ...


class OpenAILLM:
    """Answer questions with any OpenAI-compatible chat completions API."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def answer(self, question: str, results: list[SearchResult]) -> str:
        prompt = build_prompt(question, results)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()


class MockLLM:
    """Offline stand-in that echoes the top retrieved passage. For demos/tests."""

    def answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "I could not find the answer in the provided documents."
        top = results[0]
        return (
            f"(offline demo) Based on the retrieved context from {top.chunk.source}, "
            f"the most relevant passage is:\n\n{top.chunk.text[:400]}"
        )
