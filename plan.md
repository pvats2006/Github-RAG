# Master Project Plan & Progress Tracker
## GitHub Repository Documentation & Code RAG System

---

## 1. Project Goal & Overview

Build a production-grade, structure-aware **GitHub Repository Documentation + Code Retrieval-Augmented Generation (RAG)** system.

The system is designed to deeply understand entire software repositories by combining AST parsing, relationship graph construction, structure-aware chunking, hybrid dense/sparse retrieval (FAISS + BM25), reranking, and grounded LLM synthesis with strict file-path and line-range citations.

### Key Capabilities
1. **Repository Ingestion**: Clone, scan, filter, and extract file structures from GitHub repositories.
2. **Deep Code Understanding**: Python AST parsing, symbol indexing (classes, functions, methods), import resolution, and cross-file call graph mapping.
3. **Unified Knowledge Graph**: Combine AST, dependency graph, and metadata into a comprehensive repository knowledge representation (`repository_knowledge.json`).
4. **Structure-Aware Chunking**: Semantic boundary chunking for Python (classes, methods, functions, file-level), Markdown (headings, hierarchy), HTML/templates, and JSON/YAML/configs, augmented with relationship graph metadata.
5. **Hybrid Retrieval & Reranking**: Dense embeddings (optimized for Apple Silicon / M-series Mac) + BM25 keyword matching for exact symbol lookups (`evaluate_models`, `ModelTrainer`), fused and reranked for precision.
6. **Grounded Generation & Citations**: LLM responses strictly bound to retrieved context with exact `file_path:start_line-end_line` citations, refusing to hallucinate when evidence is insufficient.
7. **Developer Support Queries**:
   - *"What does this function do?"*
   - *"Where is authentication implemented?"*
   - *"How does data flow from file A to file B?"*
   - *"How do I add middleware?"*
   - *"Why is this function failing?"*
   - *"Which files need to be modified to implement this feature?"*
8. **Evaluation & Serving**: Systematic benchmark evaluation across developer question archetypes, exposed via FastAPI backend and modern UI.

---

## 2. Status Legend

| Symbol | Status | Description |
| :---: | :--- | :--- |
| 🟢 | **Completed** | Task is fully implemented, verified, and operational |
| 🟡 | **In Progress** | Task is currently being actively implemented or tested |
| ⬜ | **Not Started** | Task is planned and queued for implementation |
| 🔴 | **Blocked** | Task cannot proceed due to external dependency or prerequisite failure |
| 🔵 | **Needs Improvement** | Implemented but requires refactoring, optimization, or edge-case handling |
| ⚪ | **Not Applicable** | Task skipped or determined unnecessary |

---

## 3. Master Phase Summary Table

| Phase | Phase Name | Status | Progress | Completed Tasks | Remaining Tasks | Notes / Milestones |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 0** | **Project Setup** | 🟢 | 100% | 8 | 0 | Environment, structure, git, config established |
| **Phase 1** | **Repository Ingestion** | 🟢 | 100% | 14 | 0 | Clone, scan, filter, tree generation implemented |
| **Phase 2** | **Repository Understanding** | 🟢 | 100% | 30 | 0 | AST, symbols, imports, call graph & unified JSON |
| **Phase 3** | **Structure-Aware Chunking** | 🟡 | 15% | 2 | 14 | Python chunker started; multi-format chunkers queued |
| **Phase 4** | **Embeddings** | ⬜ | 0% | 0 | 10 | Model selection, Apple Silicon local inference, caching |
| **Phase 5** | **Vector Database** | ⬜ | 0% | 0 | 8 | FAISS dense index creation, save/load, top-k search |
| **Phase 6** | **BM25 Keyword Search** | ⬜ | 0% | 0 | 10 | Exact symbol & token index for identifiers |
| **Phase 7** | **Hybrid Retrieval** | ⬜ | 0% | 0 | 12 | Dense + BM25 fusion, graph-aware context retrieval |
| **Phase 8** | **Reranking** | ⬜ | 0% | 0 | 8 | Cross-encoder reranking & confidence thresholding |
| **Phase 9** | **Grounded LLM Generation** | ⬜ | 0% | 0 | 12 | Evidence-based prompts, anti-hallucination guardrails |
| **Phase 10** | **Citations** | ⬜ | 0% | 0 | 8 | Exact file path + line range citations |
| **Phase 11** | **Confidence / Safety** | ⬜ | 0% | 0 | 9 | Low-confidence refusal, missing symbol detection |
| **Phase 12** | **Evaluation** | ⬜ | 0% | 0 | 15 | Handcrafted developer dataset, Recall@K, MRR |
| **Phase 13** | **Query API** | ⬜ | 0% | 0 | 10 | FastAPI endpoints (`/ingest`, `/query`, `/health`) |
| **Phase 14** | **User Interface** | ⬜ | 0% | 0 | 10 | Interactive UI with context viewer & citation explorer |
| **Phase 15** | **Testing** | ⬜ | 0% | 0 | 14 | Unit, integration, parser, chunking, retrieval tests |
| **Phase 16** | **Performance / Optimization** | ⬜ | 0% | 0 | 12 | Caching, batching, parallelization, latency profiling |
| **Phase 17** | **Final Demo / Documentation** | ⬜ | 0% | 0 | 14 | README, diagrams, video/screenshots, portfolio release |

---

## 4. Current Test Repository & Knowledge Baseline

- **Current Tested Repository**: `data/repos/ML-Project`
- **Output Artifact**: `data/repository_knowledge.json`
- **Verified Metrics**:
  - **Total Files Scanned & Processed**: `18`
  - **Total Graph Nodes**: `178` (Files, Classes, Functions, Methods, Call sites)
  - **Total Graph Relationships**: `203` (`contains`, `imports`, `resolves_to`, `calls`)

---

## 5. Detailed Phase Plans

---

### Phase 0 — Project Setup

**Goal**: Establish clean modular architecture, Python environment, Git workflows, configuration, and dependencies.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P0-01** | Define modular project directory structure | 🟢 | `app/`, `data/`, `tests/`, `evaluation/` | `ls -la` | Clean separation of concerns |
| **P0-02** | Configure Python virtual environment | 🟢 | `.venv/` | `source .venv/bin/activate` | Python 3.10+ virtual environment |
| **P0-03** | Initialize Git repository | 🟢 | `.git/` | `git status` | Version control initialized |
| **P0-04** | Create `.gitignore` for Python, data, artifacts | 🟢 | `.gitignore` | `git status --ignored` | Exclude venvs, caches, indices, clones |
| **P0-05** | Define core project dependencies | 🟢 | `requirements.txt` | `pip install -r requirements.txt` | GitPython, AST tools, FAISS, rank-bm25, etc. |
| **P0-06** | Configure environment variable management | 🟢 | `.env`, `app/config.py` | `python -c "import dotenv"` | API keys, paths, index configs |
| **P0-07** | Initialize base project README | 🟢 | `README.md` | `cat README.md` | Base project documentation |
| **P0-08** | Setup module package init files | 🟢 | `app/**/__init__.py` | `python -c "import app"` | Enable package imports |

---

### Phase 1 — Repository Ingestion

**Goal**: Clone, scan, validate, and extract file hierarchies from remote GitHub repositories while filtering noise.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P1-01** | Accept GitHub repository URL input | 🟢 | `app/ingestion/github_loader.py` | Unit test URL input | Support HTTPS / SSH GitHub URLs |
| **P1-02** | Validate GitHub repository URL format | 🟢 | `app/ingestion/github_loader.py` | Regex / URL validation | Reject malformed or non-GitHub URLs |
| **P1-03** | Clone repository using GitPython | 🟢 | `app/ingestion/github_loader.py` | `GitLoader.clone_or_load()` | Automated git clone execution |
| **P1-04** | Create repository-specific local storage | 🟢 | `app/ingestion/github_loader.py` | Check `data/repos/<repo_name>` | Isolated directories per repo |
| **P1-05** | Detect and sanitize repository name | 🟢 | `app/ingestion/github_loader.py` | Test repo name parser | Handles trailing `.git` and URL slugs |
| **P1-06** | Handle existing cloned repositories | 🟢 | `app/ingestion/github_loader.py` | Re-run on cloned repo | Skip re-cloning if already present |
| **P1-07** | Robust error handling for clone failures | 🟢 | `app/ingestion/github_loader.py` | Test invalid URL / auth error | Catch GitCommandError & network drops |
| **P1-08** | Scan repository directory tree | 🟢 | `app/ingestion/file_scanner.py` | `FileScanner.scan()` | Recursive file system traversal |
| **P1-09** | Detect and categorize supported file extensions | 🟢 | `app/ingestion/file_scanner.py` | Test `.py`, `.md`, `.html`, `.json`, etc. | Language mapping based on suffix |
| **P1-10** | Filter `.git` internal directories | 🟢 | `app/ingestion/file_scanner.py` | Scan check on `.git` | Exclude git internals |
| **P1-11** | Filter virtual environments (`.venv`, `venv`, `env`) | 🟢 | `app/ingestion/file_scanner.py` | Scan check on `venv` | Ignore python environment directories |
| **P1-12** | Filter cache, build, and generated directories | 🟢 | `app/ingestion/file_scanner.py` | Scan check on `__pycache__`, `dist` | Ignore bytecode, artifacts, egg-info |
| **P1-13** | Filter large binaries and unwanted media artifacts | 🟢 | `app/ingestion/file_scanner.py` | Size threshold filter | Ignore files > 1MB or binary formats |
| **P1-14** | Generate clean repository folder/file tree structure | 🟢 | `app/ingestion/repository_structure.py` | `RepositoryStructure.build_tree()` | Hierarchy representation for context |

---

### Phase 2 — Repository Understanding

**Goal**: Extract deep AST semantics, build symbol index, resolve internal dependencies, construct relationship graphs, and assemble unified repository knowledge.

#### 2.1 AST / Code Parsing
| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P2-01** | Parse Python source files with `ast` module | 🟢 | `app/ingestion/code_parser.py` | `parse_python_file(path)` | AST traversal via `ast.NodeVisitor` |
| **P2-02** | Detect standard and relative imports | 🟢 | `app/ingestion/code_parser.py` | AST Import / ImportFrom visitor | Extract alias, module, imported symbols |
| **P2-03** | Detect class definitions and base classes | 🟢 | `app/ingestion/code_parser.py` | AST ClassDef visitor | Capture name, bases, docstring, lines |
| **P2-04** | Detect top-level function definitions | 🟢 | `app/ingestion/code_parser.py` | AST FunctionDef / AsyncFunctionDef | Capture args, docstring, decorators, lines |
| **P2-05** | Detect class method definitions | 🟢 | `app/ingestion/code_parser.py` | Nested function visitor | Link methods to parent class context |
| **P2-06** | Record precise start and end line ranges | 🟢 | `app/ingestion/code_parser.py` | `node.lineno`, `node.end_lineno` | Exact source line tracking |
| **P2-07** | Detect function and method call sites | 🟢 | `app/ingestion/code_parser.py` | AST Call visitor | Extract caller function, callee name, line |
| **P2-08** | Capture structural source-code hierarchy | 🟢 | `app/ingestion/code_parser.py` | Class-method nested mapping | Preserve structural enclosure |

#### 2.2 Symbol Index
| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P2-09** | Build repository-wide symbol index | 🟢 | `app/ingestion/symbol_index.py` | `SymbolIndex.build_index()` | Global registry of all symbols |
| **P2-10** | Index all class definitions | 🟢 | `app/ingestion/symbol_index.py` | Class lookup test | Map class names to locations |
| **P2-11** | Index all function definitions | 🟢 | `app/ingestion/symbol_index.py` | Function lookup test | Map function names to locations |
| **P2-12** | Index all method definitions | 🟢 | `app/ingestion/symbol_index.py` | Method lookup test | Map `Class.method` to locations |
| **P2-13** | Store relative file path for every symbol | 🟢 | `app/ingestion/symbol_index.py` | Index schema verification | Canonical relative repo paths |
| **P2-14** | Store line ranges for every symbol | 🟢 | `app/ingestion/symbol_index.py` | Line integrity verification | `line_start` and `line_end` bounds |
| **P2-15** | Store qualified symbol names | 🟢 | `app/ingestion/symbol_index.py` | `module.Class.method` check | Fully qualified identifier paths |
| **P2-16** | Support fast symbol lookup queries | 🟢 | `app/ingestion/symbol_index.py` | `find_symbol("ModelTrainer")` | O(1) dictionary symbol resolution |

#### 2.3 Dependency Resolution
| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P2-17** | Resolve internal module imports | 🟢 | `app/ingestion/dependency_resolver.py` | `resolve_imports()` | Convert module dots to file paths |
| **P2-18** | Map source file to imported target file | 🟢 | `app/ingestion/dependency_resolver.py` | Cross-file map test | Match `src.pipeline` to `src/pipeline.py` |
| **P2-19** | Capture exact imported symbols per statement | 🟢 | `app/ingestion/dependency_resolver.py` | Import symbol list check | Track specific class/function imports |
| **P2-20** | Store line numbers for import statements | 🟢 | `app/ingestion/dependency_resolver.py` | Import line verification | Exact line of `import` statement |

#### 2.4 Relationship Graph & Symbol Resolution
| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P2-21** | Create file, class, function, method, call nodes | 🟢 | `app/ingestion/relationship_graph.py` | Node count validation | 178 nodes verified on ML-Project |
| **P2-22** | Create `contains` structural relationships | 🟢 | `app/ingestion/relationship_graph.py` | File->Class, Class->Method | Hierarchy edges |
| **P2-23** | Create `imports` cross-file relationships | 🟢 | `app/ingestion/relationship_graph.py` | File->TargetFile | Module dependency edges |
| **P2-24** | Create `resolves_to` / `calls` relationships | 🟢 | `app/ingestion/relationship_graph.py` | Call->TargetDefinition | Caller-to-callee linkage |
| **P2-25** | Resolve imported functions and classes across files | 🟢 | `app/ingestion/relationship_graph.py` | Multi-file call resolution | Cross-file symbol resolution |
| **P2-26** | Resolve same-file function/method invocations | 🟢 | `app/ingestion/relationship_graph.py` | Intra-file call resolution | `self.method()` and local calls |
| **P2-27** | Prevent false positives from Python built-ins and third-party libraries | 🟢 | `app/ingestion/relationship_graph.py` | Built-in filter verification | Ignore `print`, `len`, `pandas`, `sklearn` |

#### 2.5 Unified Repository Knowledge
| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P2-28** | Combine scanner, AST, symbol, dependency, and graph data | 🟢 | `app/ingestion/repository_knowledge.py` | `RepositoryKnowledge.build()` | Unified knowledge aggregation |
| **P2-29** | Store raw source code and metadata per file | 🟢 | `app/ingestion/repository_knowledge.py` | Content validation in JSON | Full fidelity code preservation |
| **P2-30** | Export complete `repository_knowledge.json` | 🟢 | `data/repository_knowledge.json` | JSON schema validation | 18 files, 178 nodes, 203 relationships |

---

### Phase 3 — Structure-Aware Chunking

**Goal**: Transform raw code and documentation into rich, structure-aware semantic chunks with graph metadata, line ranges, and deterministic IDs.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P3-01** | Define standard chunk schema and dataclass | 🟢 | `app/chunking/code_chunker.py` | Verify chunk fields & serialization | `chunk_id`, `file_path`, `type`, `lines`, `symbols`, `relations` |
| **P3-02** | Implement Python function-level chunking | 🟡 | `app/chunking/code_chunker.py` | Unit test function chunks | Extract complete standalone function blocks |
| **P3-03** | Implement Python class-level chunking | ⬜ | `app/chunking/code_chunker.py` | Unit test class chunks | Class definition with method signatures & docstrings |
| **P3-04** | Implement Python method-level chunking | ⬜ | `app/chunking/code_chunker.py` | Unit test method chunks | Methods enriched with class context |
| **P3-05** | Implement Python file-level / module-level chunking | ⬜ | `app/chunking/code_chunker.py` | Unit test module chunks | Top-level declarations, imports, constants |
| **P3-06** | Handle oversized Python functions and classes | ⬜ | `app/chunking/code_chunker.py` | Test on 200+ line functions | Logical sub-block splitting without breaking syntax |
| **P3-07** | Implement Markdown heading-based chunker | ⬜ | `app/chunking/markdown_chunker.py` | Test on `README.md` | Preserve heading breadcrumbs (`# > ## > ###`) |
| **P3-08** | Preserve Markdown code blocks and table integrity | ⬜ | `app/chunking/markdown_chunker.py` | Markdown parsing test | Avoid splitting fenced code blocks |
| **P3-09** | Implement HTML template section chunker | ⬜ | `app/chunking/html_chunker.py` | Test on `templates/*.html` | Semantic chunking of template sections |
| **P3-10** | Implement JSON / YAML / Config chunker | ⬜ | `app/chunking/config_chunker.py` | Test on `requirements.txt`, configs | Key-value and section aware chunking |
| **P3-11** | Attach relationship graph metadata to each chunk | ⬜ | `app/chunking/code_chunker.py` | Verify caller/callee metadata | Inject resolved imports, callers, callees into chunk |
| **P3-12** | Generate deterministic chunk IDs | ⬜ | `app/chunking/code_chunker.py` | ID repeatability test | Format: `{file_path}::{symbol_type}::{name}::{line_start}` |
| **P3-13** | Implement chunk deduplication and overlap controls | ⬜ | `app/chunking/code_chunker.py` | Chunk uniqueness test | Prevent duplicate chunks across granularities |
| **P3-14** | Validate syntax correctness and line range accuracy | ⬜ | `app/chunking/code_chunker.py` | `ast.parse()` validation on chunks | Ensure chunks represent valid syntactic units |
| **P3-15** | Implement unified chunking pipeline orchestrator | ⬜ | `app/chunking/chunk_manager.py` | Process `repository_knowledge.json` | Route files to appropriate chunkers |
| **P3-16** | Export chunks to structured JSON / JSONL storage | ⬜ | `data/chunks.json` | Validate chunk output file | Ready for FAISS and BM25 ingestion |

---

### Phase 4 — Embeddings

**Goal**: Generate high-quality dense vector representations for code and documentation chunks, optimized for local Mac (Apple Silicon) execution.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P4-01** | Evaluate and select embedding models for code + text | ⬜ | `app/retrieval/embeddings.py` | Benchmark embedding models | e.g., `text-embedding-3-small`, `all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5` |
| **P4-02** | Build abstract `EmbeddingProvider` interface | ⬜ | `app/retrieval/embeddings.py` | Interface abstraction test | Pluggable support for Local (HuggingFace/MPS) & API (OpenAI/Gemini) |
| **P4-03** | Implement local PyTorch / HuggingFace embedding provider | ⬜ | `app/retrieval/embeddings.py` | Test MPS / Apple Silicon acceleration | Device auto-detection: `mps` / `cuda` / `cpu` |
| **P4-04** | Implement API-based embedding provider | ⬜ | `app/retrieval/embeddings.py` | Test remote API embedding call | Fallback or high-throughput API provider |
| **P4-05** | Formulate chunk embedding text representation | ⬜ | `app/retrieval/embeddings.py` | Inspect embedding input strings | Prefix with file path, symbol name, docstrings + code |
| **P4-06** | Implement batched embedding generation | ⬜ | `app/retrieval/embeddings.py` | Batch size = 32/64 processing | Efficient batch inference |
| **P4-07** | Implement embedding persistent cache | ⬜ | `data/cache/embeddings.parquet` | Test cache hit on unchanged chunks | SHA-256 chunk hash caching to avoid recomputation |
| **P4-08** | Handle embedding failures, retries, and rate limits | ⬜ | `app/retrieval/embeddings.py` | Simulate network drop / retry | Exponential backoff and graceful error logging |
| **P4-09** | Validate embedding dimensions and normalization | ⬜ | `app/retrieval/embeddings.py` | Verify unit L2 norm & vector shape | Ensure consistent dimensions for cosine similarity |
| **P4-10** | Store embedding metadata and chunk ID alignment | ⬜ | `data/index/embedding_metadata.json` | Vector-to-chunk ID mapping test | 1-to-1 alignment between index IDs and chunk IDs |

---

### Phase 5 — Vector Database (FAISS & Future Qdrant)

**Goal**: Build, persist, load, and query a high-performance FAISS vector index for dense semantic similarity retrieval.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P5-01** | Design vector index manager interface | ⬜ | `app/retrieval/vector_store.py` | Interface definition test | Decoupled vector store interface |
| **P5-02** | Implement FAISS `IndexFlatIP` (Cosine Similarity) | ⬜ | `app/retrieval/vector_store.py` | FAISS index creation test | Fast exact inner-product search on normalized vectors |
| **P5-03** | Persist FAISS index to disk | ⬜ | `data/index/faiss_index.bin` | `faiss.write_index()` | Save binary index file |
| **P5-04** | Load FAISS index from disk | ⬜ | `app/retrieval/vector_store.py` | `faiss.read_index()` | Instant startup with warm index |
| **P5-05** | Implement metadata index map (ID -> Chunk) | ⬜ | `data/index/chunk_id_map.json` | Map verification test | Fast lookup from vector row index to chunk metadata |
| **P5-06** | Implement top-k vector similarity search | ⬜ | `app/retrieval/vector_store.py` | `vector_store.search(query_vec, k=10)` | Returns ranked chunks with similarity scores |
| **P5-07** | Implement metadata pre/post-filtering | ⬜ | `app/retrieval/vector_store.py` | Filter by `file_path`, `language` | Restrict search by file type or directory |
| **P5-08** | Design architecture adapter for future Qdrant support | ⬜ | `app/retrieval/qdrant_store.py` | Schema compatibility review | Ready for cloud/distributed vector DB migration |

---

### Phase 6 — BM25 Keyword Search

**Goal**: Build a specialized sparse BM25 index tailored for code tokens, CamelCase symbols, snake_case identifiers, and exact file paths.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P6-01** | Build code-specific tokenizer (sub-tokens, snake/camel split) | ⬜ | `app/retrieval/bm25_index.py` | Test on `ModelTrainer`, `evaluate_models` | Splits `CustomException` -> `["custom", "exception", "customexception"]` |
| **P6-02** | Implement BM25 Okapi index builder | ⬜ | `app/retrieval/bm25_index.py` | `BM25Okapi` initialization | Index all code and doc chunks |
| **P6-03** | Index exact symbol names with high field weight | ⬜ | `app/retrieval/bm25_index.py` | Test exact symbol query | Boost class and function identifiers |
| **P6-04** | Index relative file paths and directory names | ⬜ | `app/retrieval/bm25_index.py` | Test path query `src/components/` | Locate chunks by path keywords |
| **P6-05** | Index full source code bodies and docstrings | ⬜ | `app/retrieval/bm25_index.py` | Test keyword query | Search across comments, strings, instructions |
| **P6-06** | Support exact identifier and partial token queries | ⬜ | `app/retrieval/bm25_index.py` | Test `evaluate_models` exact match | Retrieve exact matching function definitions |
| **P6-07** | Persist BM25 inverted index and token statistics to disk | ⬜ | `data/index/bm25_index.pkl` | Save / load test | Fast index serialization |
| **P6-08** | Implement BM25 top-k retrieval with score normalization | ⬜ | `app/retrieval/bm25_index.py` | `bm25.search(query, k=10)` | Return top matching chunks with BM25 scores |
| **P6-09** | Benchmark BM25 recall against exact symbols | ⬜ | `tests/test_bm25.py` | Test `PredictPipeline`, `CustomData` | Verify 100% Recall@1 on exact symbol lookups |
| **P6-10** | Implement incremental BM25 update support | ⬜ | `app/retrieval/bm25_index.py` | Single-file update test | Avoid full re-indexing for single file edits |

---

### Phase 7 — Hybrid Retrieval

**Goal**: Merge dense semantic retrieval and sparse BM25 keyword matching with Reciprocal Rank Fusion (RRF) and graph-aware context expansion.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P7-01** | Implement query preprocessing & intent extraction | ⬜ | `app/retrieval/hybrid_retriever.py` | Test query analyzer | Detect exact symbols vs conceptual questions |
| **P7-02** | Execute parallel Dense + BM25 retrieval | ⬜ | `app/retrieval/hybrid_retriever.py` | Retrieve top-k (e.g. k=20 each) | Fetch candidate pools from both indices |
| **P7-03** | Implement score normalization (Min-Max / Sigmoid) | ⬜ | `app/retrieval/hybrid_retriever.py` | Score distribution test | Bring dense cosine & BM25 to comparable scale |
| **P7-04** | Implement Reciprocal Rank Fusion (RRF) algorithm | ⬜ | `app/retrieval/hybrid_retriever.py` | `RRF(score_dense, score_bm25)` | Robust rank-based merging ($k=60$) |
| **P7-05** | Implement weighted convex combination fusion ($W_{dense} + W_{bm25}$) | ⬜ | `app/retrieval/hybrid_retriever.py` | Tunable weight test | Custom weights based on query type |
| **P7-06** | Deduplicate candidate chunks across retrieval passes | ⬜ | `app/retrieval/hybrid_retriever.py` | Deduplication test | Retain highest combined score per chunk ID |
| **P7-07** | Graph expansion: retrieve imported dependency context | ⬜ | `app/retrieval/graph_context.py` | Test on `ModelTrainer` -> `evaluate_models` | Follow `imports` graph edges to bring utility context |
| **P7-08** | Graph expansion: retrieve caller and callee definitions | ⬜ | `app/retrieval/graph_context.py` | Test caller/callee traversal | Follow `calls` and `resolves_to` edges |
| **P7-09** | Graph expansion: retrieve sibling methods / class container | ⬜ | `app/retrieval/graph_context.py` | Test `contains` traversal | Pull parent class header when method retrieved |
| **P7-10** | Graph expansion: retrieve related documentation chunks | ⬜ | `app/retrieval/graph_context.py` | Test `README.md` linkage | Associate documentation mentions with code chunks |
| **P7-11** | Rank and budget context tokens for graph expansion | ⬜ | `app/retrieval/hybrid_retriever.py` | Context token limiter test | Prevent context window overflow |
| **P7-12** | Unit test hybrid retrieval on multi-file flows | ⬜ | `tests/test_hybrid_retrieval.py` | Test "How is the best model selected?" | Must return `model_trainer.py` + `utils.py` |

---

### Phase 8 — Reranking

**Goal**: Apply cross-encoder reranking to hybrid candidates to refine semantic ordering and filter out irrelevant context.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P8-01** | Evaluate and select cross-encoder reranking model | ⬜ | `app/retrieval/reranker.py` | Benchmark reranker latency & quality | e.g. `bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **P8-02** | Implement local cross-encoder reranker on Apple Silicon | ⬜ | `app/retrieval/reranker.py` | Test PyTorch MPS inference | Fast local batch scoring |
| **P8-03** | Implement API reranker provider (Cohere / Voyage / HuggingFace) | ⬜ | `app/retrieval/reranker.py` | Test API reranking fallback | Optional high-accuracy remote reranker |
| **P8-04** | Pairwise query-chunk relevance scoring | ⬜ | `app/retrieval/reranker.py` | Cross-attention scoring test | Full cross-attention between query and code chunk |
| **P8-05** | Preserve chunk metadata, line numbers, and graph context | ⬜ | `app/retrieval/reranker.py` | Metadata retention test | Ensure no citation metadata is stripped |
| **P8-06** | Dynamic top-N selection and token budget fitting | ⬜ | `app/retrieval/reranker.py` | Budget allocator test | Select top 5-10 chunks fitting LLM prompt budget |
| **P8-07** | Implement confidence threshold filtering | ⬜ | `app/retrieval/reranker.py` | Test irrelevant query rejection | Discard chunks scoring below minimum threshold |
| **P8-08** | Unit test reranking on ambiguous queries | ⬜ | `tests/test_reranker.py` | Benchmark ranking shifts | Verify relevant code promoted to top 3 |

---

### Phase 9 — Grounded LLM Generation

**Goal**: Construct strict evidence-based prompts that enforce grounded answers, prevent hallucinations, and answer diverse developer inquiries.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P9-01** | Implement LLM client abstraction (OpenAI, Gemini, Local) | ⬜ | `app/generation/llm_client.py` | Test LLM generation call | Model-agnostic LLM provider |
| **P9-02** | Design system prompt with strict anti-hallucination rules | ⬜ | `app/generation/prompt_templates.py` | Prompt safety audit | Never invent code, files, or symbols |
| **P9-03** | Implement code evidence formatting for context window | ⬜ | `app/generation/prompt_templates.py` | Verify context formatting | Format: `[File: path | Lines: X-Y | Symbol: Z]\n<code>` |
| **P9-04** | Grounding rule: Prefer source code over README claims | ⬜ | `app/generation/prompt_templates.py` | Test conflicting README vs code | Code is ground truth; note discrepancies |
| **P9-05** | Grounding rule: Explicitly refuse when context is insufficient | ⬜ | `app/generation/prompt_templates.py` | Test out-of-scope query | Output: "Insufficient repository evidence" |
| **P9-06** | Developer question prompt: Explanation & Overview | ⬜ | `app/generation/generator.py` | Test "What does X do?" | Clear, concise architectural summary |
| **P9-07** | Developer question prompt: Architecture & Data Flow | ⬜ | `app/generation/generator.py` | Test "How does data flow from A to B?" | Step-by-step trace through function calls |
| **P9-08** | Developer question prompt: Debugging & Root Cause | ⬜ | `app/generation/generator.py` | Test "Why is X failing?" | Trace exception handlers, validation logic |
| **P9-09** | Developer question prompt: Feature Extension / How-To | ⬜ | `app/generation/generator.py` | Test "Which files to modify to add Y?" | Precise file list & required structural edits |
| **P9-10** | Developer question prompt: Code Navigation & Symbol Location | ⬜ | `app/generation/generator.py` | Test "Where is auth implemented?" | File paths and exact line definitions |
| **P9-11** | Implement streaming response generation | ⬜ | `app/generation/generator.py` | Test token streaming | Fast time-to-first-token in API / UI |
| **P9-12** | Implement conversation history & multi-turn memory | ⬜ | `app/generation/generator.py` | Test multi-turn follow-up question | Maintain context of previously cited symbols |

---

### Phase 10 — Citations

**Goal**: Generate strictly validated, clickable file-path and line-range citations for every claim made in the generated answer.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P10-01** | Standardize citation format: `file_path:start_line-end_line` | ⬜ | `app/generation/citations.py` | Citation schema validation | e.g. `src/components/model_trainer.py:31-86` |
| **P10-02** | Extract inline citation markers from LLM output | ⬜ | `app/generation/citations.py` | Regex citation parser | Match `[src/utils.py:21-39]` |
| **P10-03** | Verify citations against actual retrieved chunk bounds | ⬜ | `app/generation/citations.py` | Citation boundary check | Strip citations that fall outside retrieved ranges |
| **P10-04** | Verify cited line ranges against actual repository source code | ⬜ | `app/generation/citations.py` | Source line verification | Confirm file exists and line range contains cited logic |
| **P10-05** | Link multi-source citations for cross-file answers | ⬜ | `app/generation/citations.py` | Multi-citation test | Multiple citations for composite claims |
| **P10-06** | Generate structured citation metadata payload for UI/API | ⬜ | `app/generation/citations.py` | JSON citation payload check | Returns array of `{file, line_start, line_end, snippet}` |
| **P10-07** | Strip or flag unverified / hallucinated citations | ⬜ | `app/generation/citations.py` | Test hallucinated citation filter | Strict safety filter on citations |
| **P10-08** | Unit test citation accuracy across test suite queries | ⬜ | `tests/test_citations.py` | 100% verified citation test | Benchmark against gold standard citations |

---

### Phase 11 — Confidence & Safety Guardrails

**Goal**: Assess retrieval and generation confidence, detect missing symbols or ambiguities, and safely hedge or refuse low-confidence responses.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P11-01** | Compute composite retrieval confidence score | ⬜ | `app/retrieval/confidence.py` | Score algorithm test | Combine dense similarity, BM25 score & reranker logit |
| **P11-02** | Implement minimum evidence threshold gate | ⬜ | `app/retrieval/confidence.py` | Low score trigger test | If confidence < threshold, trigger refusal path |
| **P11-03** | Implement standard refusal / hedge response generator | ⬜ | `app/generation/generator.py` | Test refusal output | Return clear explanation of missing evidence |
| **P11-04** | Detect queries referencing non-existent files or symbols | ⬜ | `app/retrieval/confidence.py` | Query symbol checker | Check against `SymbolIndex` before LLM generation |
| **P11-05** | Detect ambiguous symbol references and request clarification | ⬜ | `app/retrieval/confidence.py` | Multi-match symbol test | Present matching candidates if symbol appears in 3+ files |
| **P11-06** | Detect unresolved cross-file calls and notify user | ⬜ | `app/retrieval/confidence.py` | Check graph `resolves_to` | Flag when dynamic dispatch prevents exact resolution |
| **P11-07** | Prevent hallucination of third-party library internals as local code | ⬜ | `app/generation/prompt_templates.py` | Test library query | Distinguish repo code from external PyPI packages |
| **P11-08** | Implement prompt injection and malicious code safety filters | ⬜ | `app/generation/safety.py` | Test injection strings | Sanitize user prompts against system override |
| **P11-09** | Log confidence scores and refusal telemetry for monitoring | ⬜ | `app/utils/logger.py` | Inspect confidence logs | Telemetry for threshold tuning |

---

### Phase 12 — Evaluation & Benchmarking

**Goal**: Build a ground-truth developer question benchmark dataset and evaluate retrieval metrics (Recall@K, MRR) and generation metrics (Groundedness, Citation Accuracy).

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P12-01** | Define benchmark dataset schema (`query`, `category`, `ground_truth_files`, `ground_truth_lines`, `ground_truth_answer`) | ⬜ | `evaluation/dataset_schema.py` | Dataset validation test | Standardized benchmark format |
| **P12-02** | Create hand-built dataset: File Understanding (5 queries) | ⬜ | `evaluation/dataset.json` | Dataset audit | "What is the purpose of `application.py`?" |
| **P12-03** | Create hand-built dataset: Function Understanding (5 queries) | ⬜ | `evaluation/dataset.json` | Dataset audit | "What does `initiate_data_ingestion` do?" |
| **P12-04** | Create hand-built dataset: Class Understanding (5 queries) | ⬜ | `evaluation/dataset.json` | Dataset audit | "Explain the `CustomData` class." |
| **P12-05** | Create hand-built dataset: Dependency & Data Flow (5 queries) | ⬜ | `evaluation/dataset.json` | Dataset audit | "How does `PredictPipeline` use `StandardScaler`?" |
| **P12-06** | Create hand-built dataset: Cross-file Navigation (5 queries) | ⬜ | `evaluation/dataset.json` | Dataset audit | "Where is `save_object` defined and called?" |
| **P12-07** | Create hand-built dataset: Debugging & Exceptions (5 queries) | ⬜ | `evaluation/dataset.json` | Dataset audit | "How does `CustomException` capture error details?" |
| **P12-08** | Create hand-built dataset: Feature Implementation (5 queries) | ⬜ | `evaluation/dataset.json` | Dataset audit | "How do I add CatBoost hyperparameter tuning?" |
| **P12-09** | Create hand-built dataset: Exact Symbol Lookup (5 queries) | ⬜ | `evaluation/dataset.json` | Dataset audit | "Find definition of `evaluate_models`" |
| **P12-10** | Implement retrieval evaluation script (Recall@1, Recall@3, Recall@5, Recall@10, MRR) | ⬜ | `evaluation/evaluate_retrieval.py` | `python evaluation/evaluate_retrieval.py` | Compute rank retrieval metrics |
| **P12-11** | Implement generation evaluation script (Groundedness, Citation Accuracy, Completeness) | ⬜ | `evaluation/evaluate_generation.py` | `python evaluation/evaluate_generation.py` | LLM-as-a-judge / exact match verification |
| **P12-12** | Implement hallucination rate calculation | ⬜ | `evaluation/evaluate_generation.py` | Compute unsupported claim % | Target < 2% hallucination rate |
| **P12-13** | Compare Dense vs BM25 vs Hybrid vs Reranked ablation | ⬜ | `evaluation/ablation_study.py` | `python evaluation/ablation_study.py` | Quantitative proof of hybrid retrieval gains |
| **P12-14** | Generate Markdown and CSV evaluation benchmark reports | ⬜ | `evaluation/reports/benchmark_summary.md` | Inspect generated report | Visual benchmark tables |
| **P12-15** | Automate evaluation regression test suite | ⬜ | `tests/test_evaluation_regression.py` | `pytest tests/test_evaluation_regression.py` | Prevent retrieval degradation on code changes |

---

### Phase 13 — Query API Backend

**Goal**: Expose high-performance FastAPI endpoints for ingestion, indexing, question answering, and health monitoring.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P13-01** | Initialize FastAPI application and middleware | ⬜ | `app/api/main.py` | `uvicorn app.api.main:app --reload` | CORS, error handlers, request ID middleware |
| **P13-02** | Define Pydantic request and response schemas | ⬜ | `app/api/schemas.py` | Schema validation tests | `IngestRequest`, `QueryRequest`, `QueryResponse` |
| **P13-03** | Implement `POST /ingest` endpoint | ⬜ | `app/api/routes.py` | Test with repo URL | Ingest, parse, chunk, index pipeline |
| **P13-04** | Implement `POST /query` endpoint | ⬜ | `app/api/routes.py` | Test query payload | Returns answer, citations, chunks, confidence |
| **P13-05** | Implement `POST /query/stream` SSE streaming endpoint | ⬜ | `app/api/routes.py` | Test SSE stream | Server-Sent Events for streaming generation |
| **P13-06** | Implement `GET /health` and `GET /status` endpoints | ⬜ | `app/api/routes.py` | `curl localhost:8000/health` | Index health, memory, active repository status |
| **P13-07** | Implement `GET /repository/structure` endpoint | ⬜ | `app/api/routes.py` | Test tree endpoint | Returns folder hierarchy & symbol stats |
| **P13-08** | Implement `GET /repository/file` endpoint | ⬜ | `app/api/routes.py` | Test file viewer | Return raw file content with highlighted lines |
| **P13-09** | Robust API error handling and status code mapping | ⬜ | `app/api/errors.py` | Test 400, 404, 422, 500 cases | Clean JSON error payloads |
| **P13-10** | OpenAPI / Swagger interactive API documentation | ⬜ | `app/api/main.py` | Check `http://localhost:8000/docs` | Auto-generated interactive API docs |

---

### Phase 14 — User Interface

**Goal**: Build a clean, modern, developer-centric interface for repository exploration, interactive chat, citation preview, and context inspection.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P14-01** | Select frontend stack (Streamlit or React/Vite) | ⬜ | `app/ui/` or `frontend/` | Framework initialization | High responsiveness, code rendering |
| **P14-02** | Build GitHub repository URL input and Ingest panel | ⬜ | UI Ingest Component | Test repo submission | Ingestion progress bar and status indicator |
| **P14-03** | Build interactive repository file-tree navigator | ⬜ | UI Tree Component | Test file tree expansion | Explore files, classes, methods visually |
| **P14-04** | Build conversational chat interface with streaming text | ⬜ | UI Chat Component | Test chat interaction | Message bubbles with Markdown code highlighting |
| **P14-05** | Build interactive Citation Pills & Source Links | ⬜ | UI Citation Component | Click citation pill | Highlights exact line range in code viewer |
| **P14-06** | Build split-pane Code Viewer with syntax highlighting | ⬜ | UI Code Viewer | Test file rendering | View source code with line number jumps |
| **P14-07** | Build Retrieved Context & Knowledge Graph inspector | ⬜ | UI Context Inspector | Expand context drawer | Inspect retrieved chunks, BM25 scores, graph edges |
| **P14-08** | Display confidence badge and retrieval breakdown | ⬜ | UI Confidence Badge | Test high vs low confidence | Visual confidence indicator (🟢 High, 🟡 Med, 🔴 Low) |
| **P14-09** | Implement sample question quick-starter chips | ⬜ | UI Starter Chips | Click sample question | Instant 1-click test queries |
| **P14-10** | Implement dark mode and sleek developer styling | ⬜ | UI Design System | Visual inspection | Modern dark aesthetic with glassmorphism |

---

### Phase 15 — Testing & Quality Assurance

**Goal**: Comprehensive test coverage across all pipeline stages, edge cases, malformed code, and multi-repo configurations.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P15-01** | Unit tests for Git loader and repository scanner | ⬜ | `tests/test_ingestion.py` | `pytest tests/test_ingestion.py` | Clone, load, filter verification |
| **P15-02** | Unit tests for AST parser (classes, functions, calls) | ⬜ | `tests/test_code_parser.py` | `pytest tests/test_code_parser.py` | Verify syntax tree extraction |
| **P15-03** | Unit tests for symbol indexing and lookup | ⬜ | `tests/test_symbol_index.py` | `pytest tests/test_symbol_index.py` | Exact and qualified symbol queries |
| **P15-04** | Unit tests for cross-file dependency resolver | ⬜ | `tests/test_dependency_resolver.py` | `pytest tests/test_dependency_resolver.py` | Relative and absolute import mapping |
| **P15-05** | Unit tests for relationship graph construction | ⬜ | `tests/test_relationship_graph.py` | `pytest tests/test_relationship_graph.py` | Node and edge schema validation |
| **P15-06** | Unit tests for structure-aware chunkers (Python, MD, HTML) | ⬜ | `tests/test_chunking.py` | `pytest tests/test_chunking.py` | Boundary, line range, syntax checks |
| **P15-07** | Unit tests for FAISS vector store and BM25 index | ⬜ | `tests/test_indexing.py` | `pytest tests/test_indexing.py` | Save, load, top-k score consistency |
| **P15-08** | Unit tests for hybrid retrieval fusion and graph expansion | ⬜ | `tests/test_retrieval.py` | `pytest tests/test_retrieval.py` | RRF fusion and graph context traversal |
| **P15-09** | Unit tests for citation validator and range extractor | ⬜ | `tests/test_citations.py` | `pytest tests/test_citations.py` | Exact line number and file verification |
| **P15-10** | Integration tests for FastAPI endpoints | ⬜ | `tests/test_api.py` | `pytest tests/test_api.py` | Test `/ingest`, `/query`, `/health` |
| **P15-11** | End-to-end pipeline test on `data/repos/ML-Project` | ⬜ | `tests/test_e2e.py` | `pytest tests/test_e2e.py` | Ingest -> Chunk -> Index -> Query -> Citation |
| **P15-12** | Test on multi-language and complex repositories | ⬜ | `tests/test_complex_repos.py` | Ingest 2nd repo (e.g. Flask / FastChat) | Multi-package, complex imports |
| **P15-13** | Test edge cases: duplicate symbols across files | ⬜ | `tests/test_edge_cases.py` | Duplicate function name test | Ensure disambiguation by file path |
| **P15-14** | Test edge cases: malformed Python syntax & non-UTF8 files | ⬜ | `tests/test_edge_cases.py` | Corrupt file handling | Graceful recovery without pipeline crash |

---

### Phase 16 — Performance, Caching & Optimization

**Goal**: Optimize latency, memory footprint, vector search speed, and batch embedding throughput for fast developer feedback.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P16-01** | Persistent caching for repository clones and scans | ⬜ | `app/utils/cache.py` | Verify clone cache hit | Avoid re-downloading unchanged repos |
| **P16-02** | Persistent caching for AST parsing and knowledge JSON | ⬜ | `data/cache/knowledge_cache.json` | Verify parse cache hit | Skip AST parse on unchanged file hashes |
| **P16-03** | Parallel file scanning and AST parsing with `multiprocessing` | ⬜ | `app/ingestion/code_parser.py` | Benchmark multi-core parsing | Speed up large repo ingestion 4x-8x |
| **P16-04** | Optimized batch embedding inference on Apple Silicon (MPS) | ⬜ | `app/retrieval/embeddings.py` | Benchmark embeddings/sec | High-throughput batch processing |
| **P16-05** | Persistent embedding cache keyed by chunk SHA-256 | ⬜ | `data/cache/embedding_cache.db` | Check cache reuse | 0 redundant embedding API calls |
| **P16-06** | Optimize FAISS index search parameters | ⬜ | `app/retrieval/vector_store.py` | Benchmark query latency | Sub-millisecond vector similarity search |
| **P16-07** | Optimize BM25 index memory footprint and query time | ⬜ | `app/retrieval/bm25_index.py` | Benchmark BM25 latency | Sub-millisecond sparse lookup |
| **P16-08** | Async pipeline execution for retrieval and LLM streaming | ⬜ | `app/api/routes.py` | Measure time-to-first-token | Asynchronous non-blocking I/O |
| **P16-09** | Benchmark and log full end-to-end query latency | ⬜ | `app/utils/profiler.py` | Profile breakdown table | Latency budget: < 2.5s for full RAG answer |
| **P16-10** | Measure and optimize RAM footprint during ingestion | ⬜ | `app/utils/profiler.py` | Memory profile test | Keep memory usage < 1.5 GB for standard repos |
| **P16-11** | Garbage collection and resource cleanup for large repositories | ⬜ | `app/ingestion/repository_knowledge.py` | Large repo memory test | Explicit memory deallocation |
| **P16-12** | Implement index pre-warming on application boot | ⬜ | `app/api/main.py` | Measure cold start vs warm query | Load FAISS & BM25 into memory on startup |

---

### Phase 17 — Final Demo, Documentation & Portfolio Polish

**Goal**: Create publication-grade documentation, interactive demos, architecture diagrams, and portfolio presentation assets.

| ID | Task | Status | File / Component | Command / Test | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P17-01** | Create comprehensive, high-impact `README.md` | ⬜ | `README.md` | Markdown review | Hero banner, features, architecture, setup |
| **P17-02** | Create complete System Architecture Diagram (Mermaid / PNG) | ⬜ | `docs/architecture_diagram.md` | Diagram rendering test | Ingestion -> AST Graph -> Hybrid RAG -> Citations |
| **P17-03** | Create detailed End-to-End Query Sequence Flow Diagram | ⬜ | `docs/sequence_diagram.md` | Sequence rendering test | User -> API -> Hybrid Search -> Reranker -> LLM |
| **P17-04** | Document complete local installation & setup instructions | ⬜ | `README.md`, `docs/setup.md` | Step-by-step clean install test | macOS (Apple Silicon), Linux, Windows setup |
| **P17-05** | Document environment variables and configuration options | ⬜ | `docs/configuration.md` | Config audit | Detail all keys in `.env.example` |
| **P17-06** | Document API endpoints with curl and Python examples | ⬜ | `docs/api_reference.md` | Test all sample requests | Complete REST API documentation |
| **P17-07** | Curate 10 showcase developer questions & cited answers | ⬜ | `docs/showcase_examples.md` | Verify all 10 examples | Real examples on `ML-Project` |
| **P17-08** | Publish Evaluation Results and Ablation Benchmark tables | ⬜ | `docs/evaluation_results.md` | Check metrics presentation | Publish Recall@K, MRR, groundedness scores |
| **P17-09** | Document system limitations and edge-case boundaries | ⬜ | `docs/limitations.md` | Technical transparency | Dynamic execution, non-Python AST limits |
| **P17-10** | Document future roadmap (Qdrant, Tree-sitter, Multi-repo) | ⬜ | `docs/roadmap.md` | Future vision | Multi-language AST, agentic code execution |
| **P17-11** | Record UI demonstration video / animated GIF | ⬜ | `docs/assets/demo.gif` | Visual quality check | End-to-end ingest & query demo |
| **P17-12** | Capture high-resolution UI screenshots | ⬜ | `docs/assets/` | Screenshot audit | Tree view, citation preview, chat interface |
| **P17-13** | Document codebase file structure with component descriptions | ⬜ | `docs/codebase_structure.md` | Directory map check | Clean explanation of all modules |
| **P17-14** | Package sample test repository & quickstart script | ⬜ | `scripts/run_demo.sh` | `./scripts/run_demo.sh` | One-command demo on `ML-Project` |

---

## 6. Progress Log

| Date | Phase | Task ID | Status | Result / Notes |
| :--- | :---: | :---: | :---: | :--- |
| **2026-08-22** | Phase 0 | P0-01 – P0-08 | 🟢 Completed | Project initialized with clean modular architecture, `.venv`, `.gitignore`, dependencies, and environment configs. |
| **2026-08-22** | Phase 1 | P1-01 – P1-14 | 🟢 Completed | `github_loader.py`, `file_scanner.py`, and `repository_structure.py` implemented. Tested on `data/repos/ML-Project` (18 files). |
| **2026-08-22** | Phase 2 | P2-01 – P2-30 | 🟢 Completed | Full Python AST parser, symbol index, dependency resolver, relationship graph, and unified knowledge builder completed. Verified: 18 files, 178 nodes, 203 relationships in `data/repository_knowledge.json`. |
| **2026-08-22** | Phase 3 | P3-01 – P3-02 | 🟡 In Progress | Standard chunk schema designed; `app/chunking/code_chunker.py` structure-aware chunking pipeline underway. |

---

## 7. Next Immediate Tasks

- [ ] Complete and validate Python structure-aware chunker (`app/chunking/code_chunker.py`) across classes, methods, functions, and file-level blocks.
- [ ] Implement Markdown heading-aware chunker (`app/chunking/markdown_chunker.py`) preserving breadcrumb context.
- [ ] Implement HTML template chunker (`app/chunking/html_chunker.py`) for user interface files.
- [ ] Implement JSON/YAML/Config chunker (`app/chunking/config_chunker.py`) for configuration files.
- [ ] Attach relationship graph metadata (`imports`, `calls`, `dependencies`) to every chunk.
- [ ] Validate deterministic chunk IDs, line number integrity, and syntax correctness.
- [ ] Export chunk collection to `data/chunks.json` for Phase 4 (Embeddings) and Phase 6 (BM25 Indexing).
