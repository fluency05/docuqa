import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from docuqa.config import Config
from docuqa.pipeline import RAGPipeline
from docuqa.web import create_app


def _app(tmp_path):
    config = Config(index_dir=tmp_path / "index", embedder="hashing", llm="mock")
    return create_app(RAGPipeline(config))


def test_web_serves_index(tmp_path):
    client = TestClient(_app(tmp_path))
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_web_end_to_end(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("The capital of France is Paris.", encoding="utf-8")
    client = TestClient(_app(tmp_path))

    response = client.post("/api/ingest", json={"path": str(doc)})
    assert response.status_code == 200
    assert response.json()["documents"] == 1

    response = client.post("/api/ask", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["sources"]

    response = client.get("/api/stats")
    assert response.status_code == 200
    assert response.json()["chunks"] >= 1
