# docuqa

> A lightweight, **local-first Retrieval-Augmented Generation (RAG)** document
> Q&A assistant in Python.

Point it at a folder of documents, ask questions in plain language, and get
answers **grounded in your files** — with citations back to the source. Run it
fully on your machine (private), or plug in OpenAI. CLI, web UI, and Python API.

[![CI](https://github.com/fluency05/docuqa/actions/workflows/ci.yml/badge.svg)](https://github.com/fluency05/docuqa/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## ✨ Features

- 📄 **Ingest many formats** — Markdown, plain text, reStructuredText, and PDF.
- 🧩 **Paragraph-aware chunking** — with configurable size and overlap.
- 🔍 **Semantic search** — cosine similarity over embeddings, backed by a
  dependency-light NumPy vector store (no external vector DB required).
- 🔌 **Swappable backends** — OpenAI, fully-local (`fastembed` + `Ollama`), or
  an offline demo backend.
- 🤖 **Cited answers** — every answer lists the source chunks it used.
- 🖥️ **Web UI** — a browser interface built on FastAPI (no frontend build step).
- 💬 **CLI + Python API** — one-shot `ask`, interactive `chat`, and a clean API.
- 🧪 **Tested** — unit tests plus a GitHub Actions CI pipeline.

## 🧱 How it works

```mermaid
flowchart LR
    A[Documents<br/>md / txt / pdf] --> B[Chunker]
    B --> C[Embedder<br/>openai | local]
    C --> D[(Vector store)]
    Q[Question] --> C
    D --> E[Retriever]
    Q --> E
    E --> F[LLM<br/>openai | ollama]
    F --> G[Cited answer]
```

1. **Ingest** — documents are loaded, split into overlapping chunks, embedded,
   and stored in a local vector index.
2. **Ask** — the question is embedded, the most similar chunks are retrieved,
   and an LLM writes an answer citing those chunks.

## 🚀 Quickstart

### 1. Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -e .
```

### 2. Index and ask (offline demo, no API key)

```bash
docuqa ingest examples --offline
docuqa ask "What does the FAQ say about pricing?" --offline
```

### 3. Use a real backend

Copy `.env.example` to `.env` and pick a backend (see below), then:

```bash
docuqa ingest examples
docuqa ask "What is Retrieval-Augmented Generation?"
docuqa chat          # interactive
docuqa web           # browser UI at http://127.0.0.1:8000
```

## 🔌 Backends

`docuqa` selects its embedder and LLM independently via two env vars.

### OpenAI (default)

```bash
# .env
DOCUQA_EMBEDDER=openai
DOCUQA_LLM=openai
OPENAI_API_KEY=sk-...
```

Works with any OpenAI-compatible endpoint (set `OPENAI_BASE_URL`).

### Fully local (privacy-preserving)

```bash
pip install -e ".[local]"
```

```bash
# .env
DOCUQA_EMBEDDER=local          # fastembed, runs via ONNX Runtime (no PyTorch)
DOCUQA_LLM=ollama              # a local LLM served by Ollama
```

- Embeddings: `fastembed` downloads a small model (default `BAAI/bge-small-en-v1.5`)
  and runs it entirely on your machine. Override with `DOCUQA_LOCAL_EMBEDDING_MODEL`.
- LLM: install [Ollama](https://ollama.com), pull a model
  (`ollama pull llama3.2`), and keep the server running. Override with
  `DOCUQA_OLLAMA_MODEL`.

With both set to local backends, **no document or query text ever leaves your
machine** — ideal for private/confidential knowledge bases.

## 🖥️ Web UI

```bash
pip install -e ".[web]"
docuqa web --port 8000
```

Open <http://127.0.0.1:8000>. The single-page UI lets you index a path, ask
questions, and inspect the retrieved sources and index stats — backed by a small
JSON API (`/api/ingest`, `/api/ask`, `/api/stats`).

## 🖥️ CLI reference

```text
docuqa ingest PATH... [--offline]   Load and index files or directories
docuqa ask QUESTION [--top-k N] [--offline]
                                    Answer a single question
docuqa chat [--offline]             Interactive Q&A session
docuqa web [--host H] [--port P]    Launch the web UI
docuqa stats                        Show current index info
```

Run `docuqa --help` for full details.

## 🐍 Python API

```python
from docuqa.config import Config
from docuqa.pipeline import RAGPipeline

pipeline = RAGPipeline(Config.from_env())
pipeline.ingest(["examples"])

answer = pipeline.ask("What regions does AcmeCloud support?")
print(answer.answer)
for result in answer.sources:
    print(result.chunk.source, result.score)
```

## ⚙️ Configuration

All settings are read from environment variables or a `.env` file:

| Variable                      | Default                  | Description                          |
| ----------------------------- | ------------------------ | ------------------------------------ |
| `DOCUQA_EMBEDDER`             | `openai`                 | `openai` / `local` / `hashing`       |
| `DOCUQA_LLM`                  | `openai`                 | `openai` / `ollama` / `mock`         |
| `OPENAI_API_KEY`              | —                        | API key for the OpenAI backend       |
| `OPENAI_BASE_URL`             | OpenAI default           | Any OpenAI-compatible endpoint       |
| `DOCUQA_MODEL`                | `gpt-4o-mini`            | OpenAI chat model                    |
| `DOCUQA_EMBEDDING_MODEL`      | `text-embedding-3-small` | OpenAI embeddings model              |
| `DOCUQA_LOCAL_EMBEDDING_MODEL`| `BAAI/bge-small-en-v1.5` | Local fastembed model                |
| `DOCUQA_OLLAMA_MODEL`         | `llama3.2`               | Local Ollama model                   |
| `DOCUQA_OLLAMA_BASE_URL`      | `http://localhost:11434` | Ollama server address                |
| `DOCUQA_INDEX_DIR`            | `.docuqa`                | Where the index is stored            |
| `DOCUQA_CHUNK_SIZE`           | `500`                    | Max characters per chunk             |
| `DOCUQA_CHUNK_OVERLAP`        | `50`                     | Overlap between consecutive chunks   |
| `DOCUQA_TOP_K`                | `4`                      | Chunks retrieved per question        |

## 🗂️ Project structure

```text
docuqa/
├── src/docuqa/
│   ├── cli.py        # Typer + Rich command-line interface
│   ├── web.py        # FastAPI app (JSON API)
│   ├── web/index.html  # Self-contained single-page UI
│   ├── config.py     # Environment-driven configuration
│   ├── loader.py     # .txt / .md / .rst / .pdf loading
│   ├── chunker.py    # Paragraph-aware text chunking
│   ├── embedder.py   # OpenAI + local (fastembed) + hashing embedders
│   ├── store.py      # NumPy-backed in-memory vector store
│   ├── retriever.py  # Indexing + similarity search
│   ├── llm.py        # Prompt building + OpenAI/Ollama/mock LLMs
│   ├── pipeline.py   # End-to-end orchestration
│   └── types.py      # Shared dataclasses
├── tests/            # pytest unit tests
├── examples/         # Sample documents to index
└── .github/workflows/ci.yml
```

## 🧪 Development

```bash
pip install -e ".[dev,web]"

# Run the test suite
pytest

# Lint
ruff check .
```

## 🗺️ Roadmap

- [x] Local embedding models (`fastembed`) and local LLM (`Ollama`)
- [x] A browser-based web UI
- [ ] Persistent/scalable storage backends (Chroma, FAISS, Qdrant)
- [ ] Streaming answers and chat history
- [ ] Retrieval evaluation metrics (recall, MRR)

## 📄 License

[MIT](LICENSE)
