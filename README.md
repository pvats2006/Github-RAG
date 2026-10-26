# GitHub Repository RAG

A production-grade Retrieval-Augmented Generation (RAG) system for public GitHub repositories.

Provide a GitHub repository URL, and the system automatically clones, chunks, embeds, indexes into FAISS, and runs an interactive Q&A session powered by Groq-hosted LLMs, hybrid RRF retrieval, code-aware reranking, citation validation, grounding validation, and confidence scoring.

---

## Architecture Flow

```text
User enters GitHub Repository URL
        ↓
Validate & normalize URL
        ↓
Generate deterministic repository hash (sha256[:16])
        ↓
Check whether repository is already indexed (data/repositories/<hash>/)
        ↓
If not indexed:
    [1/6] Validating repository...
    [2/6] Cloning repository...
    [3/6] Chunking repository (AST & structural chunkers)...
    [4/6] Generating embeddings (all-MiniLM-L6-v2, 384d, batched)...
    [5/6] Building FAISS index...
    [6/6] Repository ready.
        ↓
Initialize retriever, embedding model, and Groq LLM client (loaded ONCE)
        ↓
Interactive question loop:
    User question
        ↓
    Hybrid RRF retrieval (FAISS semantic + structural search)
        ↓
    Code-aware reranking
        ↓
    Context building (structured numbered sources)
        ↓
    Groq LLM generation (inline citations [1], [2])
        ↓
    Citation validation
        ↓
    Grounding validation + automatic repair
        ↓
    Confidence scoring
        ↓
    Final formatted answer with sources, status, and timing
```

---

## Key Features

- **Single Input Point**: Provide the GitHub URL only — all paths, chunks, metadata, and indexes are automatically managed.
- **Repository Isolation**: Every repository receives its own isolated directory under `data/repositories/<hash>/`.
- **Automatic Caching & Integrity Validation**: Cached indexes are verified and reused instantly on subsequent runs.
- **Hybrid Retrieval**: Combines FAISS dense embeddings with AST/structural lexical search using Reciprocal Rank Fusion (RRF).
- **Code-Aware Reranker**: Boosts functions and methods over whole-file chunks and suppresses irrelevant documentation sections for code queries.
- **Groq Cloud LLM**: High-speed LLM generation using Groq API (`llama-3.3-70b-versatile` by default).
- **Strict Grounding & Citation Validation**: Validates inline citations against retrieved sources and runs grounding checks with automatic answer repair.
- **One-Time Resource Loading**: Embedding models, FAISS vector stores, chunk stores, and LLM clients load once and stay resident across queries.

---

## Prerequisites

- Python 3.10+
- Git
- Groq API Key: [Get a free Groq API key](https://console.groq.com/keys)

---

## Setup & Installation

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/github-rag.git
cd "Github RAG"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Set your `GROQ_API_KEY`:

```bash
export GROQ_API_KEY="gsk_your_groq_api_key_here"
```

Or create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## Running the Application

Start the production CLI:

```bash
python -m app.main
```

### Example Interactive Session

```text
=================================================================
                 CODE REPOSITORY RAG
=================================================================
 GitHub URL → Clone → Chunk → Embed → FAISS → Groq
 Hybrid RRF Retrieval · Code Reranking
 Grounding · Citations · Confidence
=================================================================

Enter GitHub repository URL (or 'exit'): https://github.com/pallets/flask

[1/6] Validating repository...             ✓
[2/6] Cloning repository...                ✓
[3/6] Chunking repository...               ✓ (1,842 chunks from 94 files)
[4/6] Generating embeddings...             ✓ (384d)
[5/6] Building FAISS index...              ✓ (1,842 vectors)
[6/6] Repository ready.                    ✓

=================================================================
  REPOSITORY READY
=================================================================
  Repository : https://github.com/pallets/flask
  Files      : 94
  Chunks     : 1,842
  Embedding  : all-MiniLM-L6-v2 (384d)
  Vector DB  : FAISS
  LLM        : llama-3.3-70b-versatile
=================================================================

Ask your question (type 'exit' to switch repo): How does the application context work?

Searching repository...
Generating answer...

=================================================================
ANSWER
=================================================================
The application context binds the application to the current request or CLI command [1]. It is created via the `AppContext` class [1] and can be pushed and popped using `app_context()` [2].

Sources:
[1] src/flask/ctx.py | AppContext | Lines 150-220
     https://github.com/pallets/flask/blob/main/src/flask/ctx.py#L150-L220
[2] src/flask/app.py | Flask.app_context | Lines 1800-1840
     https://github.com/pallets/flask/blob/main/src/flask/app.py#L1800-L1840

Citations  : PASS
Grounding  : PASS
Confidence : HIGH (92.5%)
Timing     : Retrieval: 0.04s | Generation: 0.72s | Validation: 0.03s | Total: 0.79s
=================================================================
```

---

## Python Programmatic API

```python
from app.repository_rag import RepositoryRAG

# Initialize for any public repository
rag = RepositoryRAG("https://github.com/pallets/flask")
info = rag.initialize()
print(f"Ready: {info['repo_name']} with {info['chunks_indexed']} chunks")

# Ask multiple questions (no model/index reloading)
result = rag.ask("Where is routing configured?")
print(result["answer"])
print(result["citations"])
print(result["confidence"])

rag.close()
```

---

## Project Structure

```text
app/
├── main.py                        ← Production CLI entry point
├── repo_indexer.py                ← Clone / AST chunk / embed / FAISS / manifest
├── repository_rag.py              ← High-level RepositoryRAG API
├── rag_pipeline.py                ← End-to-end RAG orchestrator
│
├── chunking/                      ← Multi-language AST & structural chunkers
│   ├── chunk_schema.py            ← CodeChunk schema
│   ├── chunk_store.py             ← Chunk store
│   ├── python_chunker.py          ← Python AST parser
│   ├── markdown_chunker.py        ← Markdown section chunker
│   ├── html_chunker.py            ← HTML chunker
│   └── config_chunker.py          ← Config & YAML chunker
│
├── embeddings/
│   ├── embedding_model.py         ← all-MiniLM-L6-v2 (SentenceTransformer)
│   └── embed_chunks.py            ← Batch embedding generator
│
├── retrieval/
│   ├── semantic_retriever.py      ← Hybrid RRF retriever
│   ├── code_reranker.py           ← Code-aware reranker
│   ├── vector_store.py            ← FAISS vector store bridge
│   ├── faiss_index.py             ← FAISS index wrapper
│   ├── context_builder.py         ← LLM prompt context builder
│   ├── retrieval_evaluation.py    ← Evaluation benchmark
│   └── large_retrieval_evaluation.py ← Multi-query retrieval benchmark
│
├── llm/
│   └── llm_client.py              ← Groq API abstraction
│
└── validation/
    ├── grounded_validator.py      ← Grounding validator & repair
    ├── confidence.py              ← Confidence evaluator
    └── citation_validator.py      ← Citation validator

data/
└── repositories/
    └── <16-char-hash>/            ← Per-repository isolated storage
        ├── source/                ← Cloned repository files
        ├── chunks.json            ← Extracted code chunks
        ├── repository.index       ← FAISS index
        ├── embedding_metadata.json← Metadata mapping
        └── manifest.json          ← Cache manifest & stats

tests/
└── test_rag_system.py             ← Unit & regression test suite
```

---

## Running Tests

Run the test suite:

```bash
pytest
```

---

## Configuration Reference

| Environment Variable / Setting | Default Value | Description |
|---|---|---|
| `GROQ_API_KEY` | *(Required)* | Groq API Key for LLM generation |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model for Q&A and validation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer embedding model (384d) |
| `TOP_K` | `8` | Number of chunks retrieved per query |
| `MAX_FILE_BYTES` | `1,000,000` (1MB) | Maximum file size for indexing |
| `MAX_CHUNK_LINES` | `120` | Maximum lines per chunk |
