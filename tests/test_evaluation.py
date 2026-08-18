from docuqa.evaluation import EvalCase, _source_matches, evaluate, load_cases
from docuqa.types import Chunk, SearchResult


def _result(source: str, score: float = 0.9) -> SearchResult:
    return SearchResult(chunk=Chunk(id="x", text="t", source=source, index=0), score=score)


class _FakeRetriever:
    def __init__(self, responses: dict) -> None:
        self._responses = responses

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return self._responses.get(query, [])


def test_source_matches_paths():
    assert _source_matches("examples/rag-guide.md", "rag-guide.md")
    assert _source_matches("F:/docuqa/examples/rag-guide.md", "examples/rag-guide.md")
    assert _source_matches("examples/rag-guide.md", "examples/rag-guide.md")
    assert not _source_matches("examples/other.md", "rag-guide.md")


def test_evaluate_computes_recall_and_mrr():
    retriever = _FakeRetriever(
        {
            "q1": [_result("a.md"), _result("b.md"), _result("c.md")],
            "q2": [_result("x.md")],
        }
    )
    cases = [EvalCase("q1", ["b.md"]), EvalCase("q2", ["nope.md"])]
    report = evaluate(retriever, cases, top_k=3)
    assert report.recall_at_k == 0.5
    assert report.mrr == 0.25  # q1 rank 2 -> 0.5; q2 no hit -> 0


def test_evaluate_empty_cases():
    report = evaluate(_FakeRetriever({}), [], top_k=3)
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0


def test_load_cases(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text('[{"query": "q", "relevant": ["a.md", "b.md"]}]', encoding="utf-8")
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].query == "q"
    assert cases[0].relevant == ["a.md", "b.md"]
