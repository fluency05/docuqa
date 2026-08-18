# docuqa

> 一个轻量级、**本地优先的检索增强生成（RAG）**文档问答助手，使用 Python 编写。

把它指向一个文档文件夹，用自然语言提问，就能得到**基于你文件内容**的答案——并且**附带出处引用**。可以完全在本机运行（隐私安全），也可以接入 OpenAI。支持命令行、Web 界面和 Python API。

> 🌐 [English](README.md)

[![CI](https://github.com/fluency05/docuqa/actions/workflows/ci.yml/badge.svg)](https://github.com/fluency05/docuqa/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## ✨ 特性

- 📄 **支持多种格式** — Markdown、纯文本、reStructuredText 和 PDF。
- 🧩 **段落感知切分** — 可配置分块大小与重叠。
- 🔍 **语义检索** — 基于向量余弦相似度，底层是零额外依赖的 NumPy 向量库（无需外部向量数据库）。
- 🔌 **可插拔后端** — OpenAI、完全本地（`fastembed` + `Ollama`），或离线演示后端。
- 🤖 **带引用回答** — 每个答案都会列出它所使用的来源分块。
- 🔁 **增量索引** — 重新索引同一来源会替换旧分块，而不是重复叠加。
- 🌐 **中英文支持** — 中英文句子级切分，并支持中文本地嵌入模型。
- 📊 **检索质量评估** — `docuqa eval` 输出 Recall@k 与 MRR。
- 🖥️ **Web 界面** — 基于 FastAPI 的浏览器界面（无需前端构建）。
- 💬 **CLI + Python API** — 一次性 `ask`、交互式 `chat`，以及清晰的 API。
- 🧪 **有测试** — 单元测试 + GitHub Actions CI。

## 🧱 工作原理

```mermaid
flowchart LR
    A[文档<br/>md / txt / pdf] --> B[切分 Chunker]
    B --> C[嵌入 Embedder<br/>openai | local]
    C --> D[(向量库)]
    Q[问题] --> C
    D --> E[检索 Retriever]
    Q --> E
    E --> F[大模型 LLM<br/>openai | ollama]
    F --> G[带引用答案]
```

1. **索引（Ingest）** — 加载文档，切分成带重叠的分块，向量化后存入本地向量索引。
2. **提问（Ask）** — 将问题向量化，检索最相似的分块，再由大模型基于这些分块生成带引用的答案。

## 🚀 快速开始

### 1. 安装

```bash
python -m venv .venv
# Windows：
.venv\Scripts\activate
# macOS / Linux：
source .venv/bin/activate

pip install -e .
```

### 2. 索引并提问（离线演示，无需 API key）

```bash
docuqa ingest examples --offline
docuqa ask "What does the FAQ say about pricing?" --offline
```

### 3. 使用真实后端

复制 `.env.example` 为 `.env`，选择一个后端（见下文），然后：

```bash
docuqa ingest examples
docuqa ask "What is Retrieval-Augmented Generation?"
docuqa chat          # 交互式对话
docuqa web           # 浏览器界面，地址 http://127.0.0.1:8000
```

## 🔌 后端配置

`docuqa` 通过两个环境变量分别选择嵌入（embedder）与大模型（LLM）。

### OpenAI（默认）

```bash
# .env
DOCUQA_EMBEDDER=openai
DOCUQA_LLM=openai
OPENAI_API_KEY=sk-...
```

兼容任意 OpenAI 兼容端点（设置 `OPENAI_BASE_URL`）。

### DeepSeek（仅对话模型）

[DeepSeek](https://api-docs.deepseek.com/) 是 OpenAI 兼容接口，可直接作为 LLM 使用。但它**没有 embedding 接口**，因此需要搭配本地嵌入模型：

```bash
pip install -e ".[local]"
```

```bash
# .env
DOCUQA_LLM=deepseek
OPENAI_API_KEY=sk-你的deepseek-key
DOCUQA_EMBEDDER=local          # DeepSeek 不支持 embedding；用本地（或 OpenAI）
DOCUQA_DEEPSEEK_MODEL=deepseek-chat   # 或 deepseek-reasoner（R1 推理模型）
```

### 完全本地（隐私优先）

```bash
pip install -e ".[local]"
```

```bash
# .env
DOCUQA_EMBEDDER=local          # fastembed，基于 ONNX Runtime（无需 PyTorch）
DOCUQA_LLM=ollama              # 由 Ollama 提供本地大模型
```

- 嵌入：`fastembed` 会下载一个小模型（默认 `BAAI/bge-small-en-v1.5`）并完全在本机运行。可用 `DOCUQA_LOCAL_EMBEDDING_MODEL` 覆盖。
- 大模型：安装 [Ollama](https://ollama.com)，拉取模型（`ollama pull llama3.2`）并保持服务运行。可用 `DOCUQA_OLLAMA_MODEL` 覆盖。

当两者都设为本地后端时，**任何文档或问题内容都不会离开你的机器**——非常适合私密/保密的知识库。

### 🌐 中文文档

切分支持中英文句子边界（在 `。！？;` 和 `.?!;` 处断开）。处理中文文档时，建议使用中文嵌入模型：

```bash
# .env
DOCUQA_EMBEDDER=local
DOCUQA_LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

中文示例文档见 `examples/星辰笔记-产品说明.md`。

## 🖥️ Web 界面

```bash
pip install -e ".[web]"
docuqa web --port 8000
```

打开 <http://127.0.0.1:8000>。单页界面（内置中英文切换）支持：

- **拖拽或多选文件**上传并索引，或
- **按服务器路径索引整个目录**，
- 提问并查看检索来源与索引统计。

底层是简洁的 JSON API（`/api/upload`、`/api/ingest`、`/api/ask`、`/api/stats`）。

## 🖥️ CLI 命令参考

```text
docuqa ingest PATH... [--offline]   加载并索引文件或目录
docuqa ask QUESTION [--top-k N] [--offline]
                                    回答单个问题
docuqa chat [--offline]             交互式问答
docuqa eval DATASET [--top-k N] [--offline]
                                    评估 Recall@k 与 MRR
docuqa web [--host H] [--port P]    启动 Web 界面
docuqa stats                        查看当前索引信息
```

运行 `docuqa --help` 查看完整帮助。

## 📊 检索评估

用一个小型 JSON 数据集评估检索质量（Recall@k 与 MRR）：

```json
[
  { "query": "What is RAG?", "relevant": ["rag-guide.md"] },
  { "query": "How much does AcmeCloud cost?", "relevant": ["acmecloud-faq.md"] }
]
```

```bash
docuqa ingest examples
docuqa eval examples/eval-sample.json --top-k 4
```

`relevant` 里的来源可按完整路径或文件名匹配。命令会输出每个查询的命中表和汇总指标。

> 提示：请用真实嵌入模型（`openai` 或 `local`）做评估——离线哈希嵌入只用于演示，分数会偏低。

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

## ⚙️ 配置项

所有设置都从环境变量或 `.env` 文件读取：

| 变量                          | 默认值                   | 说明                                 |
| ----------------------------- | ------------------------ | ------------------------------------ |
| `DOCUQA_EMBEDDER`             | `openai`                 | `openai` / `local` / `hashing`       |
| `DOCUQA_LLM`                  | `openai`                 | `openai` / `deepseek` / `ollama` / `mock` |
| `OPENAI_API_KEY`              | —                        | OpenAI / DeepSeek 后端的 API key     |
| `OPENAI_BASE_URL`             | OpenAI 默认              | 任意 OpenAI 兼容端点                 |
| `DOCUQA_MODEL`                | `gpt-4o-mini`            | OpenAI 对话模型                      |
| `DOCUQA_EMBEDDING_MODEL`      | `text-embedding-3-small` | OpenAI 嵌入模型                      |
| `DOCUQA_DEEPSEEK_MODEL`       | `deepseek-chat`          | DeepSeek 对话模型（R1 用 `deepseek-reasoner`） |
| `DOCUQA_DEEPSEEK_BASE_URL`    | `https://api.deepseek.com` | DeepSeek API 端点                  |
| `DOCUQA_LOCAL_EMBEDDING_MODEL`| `BAAI/bge-small-en-v1.5` | 本地 fastembed 模型                  |
| `DOCUQA_OLLAMA_MODEL`         | `llama3.2`               | 本地 Ollama 模型                     |
| `DOCUQA_OLLAMA_BASE_URL`      | `http://localhost:11434` | Ollama 服务地址                      |
| `DOCUQA_INDEX_DIR`            | `.docuqa`                | 索引存储位置                         |
| `DOCUQA_CHUNK_SIZE`           | `500`                    | 每个分块的最大字符数                 |
| `DOCUQA_CHUNK_OVERLAP`        | `50`                     | 相邻分块的重叠字符数                 |
| `DOCUQA_TOP_K`                | `4`                      | 每个问题检索的分块数                 |

## 🗂️ 项目结构

```text
docuqa/
├── src/docuqa/
│   ├── cli.py        # Typer + Rich 命令行界面
│   ├── web.py        # FastAPI 应用（JSON API）
│   ├── web/index.html  # 自包含单页界面
│   ├── config.py     # 环境变量驱动的配置
│   ├── loader.py     # .txt / .md / .rst / .pdf 加载
│   ├── chunker.py    # 段落感知的文本切分
│   ├── embedder.py   # OpenAI + 本地（fastembed）+ 哈希嵌入
│   ├── store.py      # 基于 NumPy 的内存向量库
│   ├── retriever.py  # 索引 + 相似度检索
│   ├── llm.py        # 提示词构建 + OpenAI/Ollama/mock 大模型
│   ├── pipeline.py   # 端到端编排
│   └── types.py      # 共享数据结构
├── tests/            # pytest 单元测试
├── examples/         # 示例文档
└── .github/workflows/ci.yml
```

## 🧪 开发

```bash
pip install -e ".[dev,web]"

# 运行测试
pytest

# 代码检查
ruff check .
```

## 🗺️ 路线图

- [x] 本地嵌入模型（`fastembed`）与本地大模型（`Ollama`）
- [x] 浏览器 Web 界面
- [x] 检索评估指标（Recall@k、MRR）
- [x] 增量索引（重新索引即替换，不重复）
- [ ] 持久化/可扩展的向量库后端（Chroma、FAISS、Qdrant）
- [ ] 流式回答与对话历史

## 📄 许可证

[MIT](LICENSE)
