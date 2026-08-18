from docuqa.config import Config
from docuqa.pipeline import RAGPipeline


def _offline_config(tmp_path) -> Config:
    return Config(index_dir=tmp_path / "index", embedder="hashing", llm="mock")


def test_offline_end_to_end(tmp_path):
    config = _offline_config(tmp_path)
    doc = tmp_path / "notes.md"
    doc.write_text(
        "Retrieval-Augmented Generation combines retrieval with generation.\n\n"
        "The capital of France is Paris.",
        encoding="utf-8",
    )

    pipeline = RAGPipeline(config)
    report = pipeline.ingest([doc])

    assert report.documents == 1
    assert report.chunks >= 1

    answer = pipeline.ask("What is Retrieval-Augmented Generation?")
    assert answer.question == "What is Retrieval-Augmented Generation?"
    assert answer.answer
    assert answer.sources


def test_ingest_persists_index_and_reloads(tmp_path):
    config = _offline_config(tmp_path)
    doc = tmp_path / "notes.md"
    doc.write_text("The sky is blue.", encoding="utf-8")

    RAGPipeline(config).ingest([doc])

    reloaded = RAGPipeline(config)
    assert reloaded.stats()["chunks"] >= 1
    assert reloaded.ask("What color is the sky?").sources


def test_stats_reports_backends(tmp_path):
    config = _offline_config(tmp_path)
    stats = RAGPipeline(config).stats()
    assert stats["embedder"] == "hashing"
    assert stats["llm"] == "mock"
