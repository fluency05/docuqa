"""Retrieval evaluation: Recall@k and Mean Reciprocal Rank (MRR)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .retriever import Retriever


@dataclass
class EvalCase:
    """One evaluation example: a query and the sources it should retrieve."""

    query: str
    relevant: list[str]


@dataclass
class EvalResult:
    """Outcome for a single query."""

    query: str
    relevant: list[str]
    retrieved: list[str]
    rank: int | None = None  # 1-based rank of the first relevant hit
    hit: bool = False


@dataclass
class EvalReport:
    """Aggregated evaluation metrics over all queries."""

    k: int
    results: list[EvalResult] = field(default_factory=list)
    recall_at_k: float = 0.0
    mrr: float = 0.0


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load an evaluation dataset from a JSON file.

    Expected shape::

        [{"query": "What is RAG?", "relevant": ["rag-guide.md"]}, ...]
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Eval dataset must be a JSON list of {query, relevant} objects.")
    cases = []
    for item in raw:
        cases.append(EvalCase(query=item["query"], relevant=list(item["relevant"])))
    return cases


def evaluate(retriever: Retriever, cases: list[EvalCase], top_k: int) -> EvalReport:
    """Run retrieval for each case and compute Recall@k and MRR.

    Recall@k is the fraction of queries with at least one relevant source among
    the top-k results; MRR is the mean of ``1 / rank`` of the first relevant hit
    (queries without a hit contribute 0).
    """
    results: list[EvalResult] = []
    for case in cases:
        hits = retriever.search(case.query, top_k)
        rank = None
        for position, result in enumerate(hits, 1):
            if any(_source_matches(result.chunk.source, rel) for rel in case.relevant):
                rank = position
                break
        results.append(
            EvalResult(
                query=case.query,
                relevant=case.relevant,
                retrieved=[result.chunk.source for result in hits],
                rank=rank,
                hit=rank is not None,
            )
        )

    total = len(results)
    recall = sum(1 for result in results if result.hit) / total if total else 0.0
    mrr = sum(1 / result.rank for result in results if result.rank) / total if total else 0.0
    return EvalReport(k=top_k, results=results, recall_at_k=recall, mrr=mrr)


def _source_matches(source: str, relevant: str) -> bool:
    """Match a retrieved source path against a relevant-source label.

    Accepts exact paths, basenames, or relative paths that suffix the retrieved
    source (e.g. ``examples/rag-guide.md`` matches ``.../examples/rag-guide.md``).
    """
    normalized = source.replace("\\", "/")
    label = relevant.replace("\\", "/").strip("/")
    return (
        normalized == label
        or Path(normalized).name == Path(label).name
        or normalized.endswith("/" + label)
    )
