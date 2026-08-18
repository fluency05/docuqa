"""Graceful failures for optional local backends (fastembed / ollama)."""

import sys

import pytest

from docuqa.embedder import LocalEmbedder
from docuqa.llm import OllamaLLM


def test_local_embedder_requires_fastembed(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ImportError, match="fastembed"):
        LocalEmbedder()


def test_ollama_llm_requires_ollama(monkeypatch):
    monkeypatch.setitem(sys.modules, "ollama", None)
    with pytest.raises(ImportError, match="ollama"):
        OllamaLLM()
