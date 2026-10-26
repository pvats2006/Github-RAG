"""
Dynamic GitHub repository indexer for the Code RAG application.

User-facing contract:
    GitHub URL -> clone -> discover files -> chunk -> embed -> FAISS

The index is stored per repository so different repositories never share
the same vector store.

Chunks are stored in CodeChunk-compatible JSON format so that
SemanticRetriever / VectorStore / ChunkStore can load them directly
without any conversion.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

from app.embeddings.embedding_model import EmbeddingModel
from app.retrieval.faiss_index import FAISSIndex

try:
    from git import Repo as _GitRepo
    _GIT_AVAILABLE = True
except ImportError:
    _GitRepo = None  # type: ignore[assignment]
    _GIT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", "dist",
    "build", "target", ".next", ".nuxt", "coverage", ".idea", ".vscode",
    ".github", ".circleci", "vendor", ".cache", "catboost_info",
}

TEXT_EXTENSIONS = {
    ".py", ".pyw",
    ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp",
    ".cs", ".php", ".rb", ".swift", ".kt", ".kts", ".scala",
    ".sh", ".bash", ".zsh",
    ".sql",
    ".html", ".htm", ".css", ".scss", ".sass", ".vue", ".svelte",
    ".md", ".mdx",
    ".txt", ".yaml", ".yml", ".json", ".toml", ".ini",
    ".cfg", ".xml", ".dockerfile",
}

BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg",
    ".mp4", ".avi", ".mov", ".mp3", ".wav",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".class", ".jar",
}

MAX_FILE_BYTES = 1_000_000   # 1 MB
MAX_CHUNK_LINES = 120
OVERLAP_LINES = 12


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------

def normalize_github_url(url: str) -> str:
    """Return a canonical clone URL for a GitHub repository."""
    url = url.strip()
    if not url:
        raise ValueError("GitHub URL cannot be empty.")

    if not re.match(r"^https?://", url, re.I):
        raise ValueError(
            "Please enter a full GitHub URL beginning with https://\n"
            "Example: https://github.com/owner/repository"
        )

    parsed = urlparse(url)
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError(
            "Only github.com repository URLs are supported.\n"
            "Example: https://github.com/owner/repository"
        )

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            "GitHub URL must look like https://github.com/owner/repository"
        )

    owner, repo = parts[0], parts[1]
    repo = re.sub(r"\.git$", "", repo, flags=re.I)
    if not owner or not repo:
        raise ValueError("Invalid GitHub repository URL.")

    return f"https://github.com/{owner}/{repo}.git"


def repo_identity(url: str) -> tuple[str, str]:
    """
    Return (repo_key, repo_name) for a GitHub URL.

    repo_key  — 16-char sha256 hex used as the directory name.
    repo_name — 'owner/repo' display string.
    """
    normalized = normalize_github_url(url)
    parsed = urlparse(normalized)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    owner = parts[-2]
    repo = re.sub(r"\.git$", "", parts[-1], flags=re.I)
    key = hashlib.sha256(f"{owner}/{repo}".encode()).hexdigest()[:16]
    return key, f"{owner}/{repo}"


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def language_for(path: Path) -> str:
    if path.name.lower() == "dockerfile":
        return "dockerfile"
    return {
        ".py": "python",   ".pyw": "python",
        ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".java": "java",   ".go": "go",     ".rs": "rust",
        ".c": "c",         ".h": "c",       ".cc": "cpp",
        ".cpp": "cpp",     ".cxx": "cpp",   ".hpp": "cpp",
        ".cs": "csharp",   ".php": "php",   ".rb": "ruby",
        ".swift": "swift", ".kt": "kotlin", ".kts": "kotlin",
        ".scala": "scala",
        ".sh": "shell",    ".bash": "shell", ".zsh": "shell",
        ".sql": "sql",
        ".html": "html",   ".htm": "html",
        ".css": "css",     ".scss": "scss",  ".sass": "sass",
        ".vue": "vue",     ".svelte": "svelte",
        ".md": "markdown", ".mdx": "markdown",
        ".yaml": "yaml",   ".yml": "yaml",
        ".json": "json",   ".toml": "toml",
        ".ini": "ini",     ".cfg": "config",
        ".xml": "xml",     ".txt": "text",
    }.get(path.suffix.lower(), "text")


# ---------------------------------------------------------------------------
# Safe file reading
# ---------------------------------------------------------------------------

def safe_read(path: Path) -> str | None:
    """Read a text file, respecting size limit and symlink safety."""
    try:
        # Resolve symlinks but check the real path is still inside repo
        real = path.resolve()
        if path.stat().st_size > MAX_FILE_BYTES:
            logger.debug("Skipped large file: %s (exceeds %d bytes)", path, MAX_FILE_BYTES)
            return None
        return real.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# GitHub reference builder
# ---------------------------------------------------------------------------

def _github_ref(repo_name: str, file_path: str, start: int, end: int) -> str:
    safe = file_path.replace(" ", "%20")
    return f"https://github.com/{repo_name}/blob/main/{safe}#L{start}-L{end}"


# ---------------------------------------------------------------------------
# CodeChunk-compatible dict factory
# ---------------------------------------------------------------------------

def _make_chunk_dict(
    repo_name: str,
    rel: Path,
    start: int,
    end: int,
    content: str,
    chunk_type: str,
    symbol: str | None = None,
    parent_symbol: str | None = None,
    imports: list[str] | None = None,
    calls: list[str] | None = None,
) -> dict:
    """
    Build a dict that exactly matches CodeChunk fields so ChunkStore
    can load it with ``CodeChunk(**item)``.
    """
    rel_posix = rel.as_posix()
    digest = hashlib.sha1(
        f"{repo_name}:{rel_posix}:{start}:{end}:{symbol or ''}".encode()
    ).hexdigest()[:20]

    return {
        # --- CodeChunk required fields ---
        "chunk_id":     f"{rel_posix}::{chunk_type}::{digest}",
        "repo":         repo_name,
        "file_path":    rel_posix,
        "file_name":    rel.name,
        "language":     language_for(rel),
        "chunk_type":   chunk_type,
        "symbol":       symbol,
        "line_start":   start,
        "line_end":     end,
        "content":      content,
        # --- CodeChunk optional list fields ---
        "imports":      imports or [],
        "calls":        calls or [],
        "dependencies": [],
        "related_files": [],
        "parent_symbol": parent_symbol,
        # --- metadata dict (github_reference lives here) ---
        "metadata": {
            "github_reference": _github_ref(repo_name, rel_posix, start, end),
        },
    }


# ---------------------------------------------------------------------------
# Python AST chunker (produces CodeChunk-compatible dicts)
# ---------------------------------------------------------------------------

def _extract_imports(tree: ast.Module) -> list[str]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(f"{mod}.{alias.name}" if mod else alias.name)
    return imports


def _extract_calls(func_node: ast.AST) -> list[str]:
    calls = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return list(dict.fromkeys(calls))[:20]  # dedupe, cap


def _python_chunks(repo_name: str, rel: Path, source: str) -> list[dict]:
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    module_imports = _extract_imports(tree)

    # Collect all function/class nodes with their full ranges
    nodes: list[tuple[int, int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            nodes.append((start, end, "function", node.name))
        elif isinstance(node, ast.ClassDef):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            nodes.append((start, end, "class", node.name))

    # Reclassify functions inside classes as methods
    class_ranges = [(s, e, sym) for s, e, typ, sym in nodes if typ == "class"]
    refined = []
    for start, end, typ, sym in nodes:
        if typ == "function":
            parent = next(
                (name for cs, ce, name in class_ranges if cs <= start <= ce),
                None,
            )
            if parent:
                refined.append((start, end, "method", f"{parent}.{sym}", parent))
                continue
        refined.append((start, end, typ, sym, None))

    refined.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    chunks = []

    for start, end, typ, sym, parent in refined:
        start = max(1, start)
        end = min(len(lines), end)
        content = "\n".join(lines[start - 1:end]).strip()
        if not content:
            continue

        # Find the AST node to extract calls
        func_calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.lineno == start and node.name == sym.split(".")[-1]:
                    func_calls = _extract_calls(node)
                    break

        chunks.append(_make_chunk_dict(
            repo_name, rel, start, end, content, typ,
            symbol=sym,
            parent_symbol=parent,
            imports=module_imports if typ in ("file", "function", "method") else [],
            calls=func_calls,
        ))

    # File-level chunk: covers imports and module-level code
    if lines:
        chunks.append(_make_chunk_dict(
            repo_name, rel, 1, len(lines), source.strip(), "file",
            symbol=rel.name,
            imports=module_imports,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Generic structural chunker (JS, TS, Java, Go, Rust, etc.)
# ---------------------------------------------------------------------------

DECL_RE = re.compile(
    r"^\s*(?:public|private|protected|static|final|async|export|default|\s)*"
    r"(?:class|interface|struct|enum|function)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.I,
)
JS_FUNC_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)"
)
PY_LIKE_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)")


def _generic_structural_chunks(repo_name: str, rel: Path, source: str) -> list[dict]:
    lines = source.splitlines()
    chunks = []
    starts: list[tuple[int, str, str]] = []

    for i, line in enumerate(lines, start=1):
        m = PY_LIKE_RE.match(line) or JS_FUNC_RE.match(line) or DECL_RE.match(line)
        if m:
            symbol = m.group(1)
            kind = (
                "class"
                if re.search(r"\b(class|interface|struct|enum)\b", line, re.I)
                else "function"
            )
            starts.append((i, kind, symbol))

    for idx, (start, kind, symbol) in enumerate(starts):
        end = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(lines)
        end = min(end, start + MAX_CHUNK_LINES - 1)
        content = "\n".join(lines[start - 1:end]).strip()
        if content:
            chunks.append(_make_chunk_dict(repo_name, rel, start, end, content, kind, symbol=symbol))

    # Markdown sections become semantic chunks
    if language_for(rel) == "markdown":
        heading_starts = [
            (i, line.lstrip("# ").strip())
            for i, line in enumerate(lines, start=1)
            if re.match(r"^\s{0,3}#{1,6}\s+\S+", line)
        ]
        for idx, (start, title) in enumerate(heading_starts):
            end = (
                heading_starts[idx + 1][0] - 1
                if idx + 1 < len(heading_starts)
                else len(lines)
            )
            content = "\n".join(lines[start - 1:end]).strip()
            if content:
                chunks.append(_make_chunk_dict(
                    repo_name, rel, start, end, content,
                    "markdown_section", symbol=title
                ))

    return chunks


# ---------------------------------------------------------------------------
# Repository chunker
# ---------------------------------------------------------------------------

def chunk_repository(repo_root: Path, repo_name: str) -> list[dict]:
    """
    Walk a cloned repository and produce CodeChunk-compatible dicts.

    Parameters
    ----------
    repo_root : Path
        Root of the cloned repository.
    repo_name : str
        'owner/repo' string used for github references.

    Returns
    -------
    list[dict]
        Deduplicated list of chunk dicts, ready for ChunkStore / JSON.
    """
    chunks: list[dict] = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        # Skip symlinks that escape the repo root
        try:
            path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            logger.warning("Skipped symlink escaping repo root: %s", path)
            continue

        # Skip ignored directories
        if any(part in IGNORED_DIRS for part in path.parts):
            continue

        # Skip binary / non-text files
        if path.suffix.lower() in BINARY_EXTENSIONS:
            continue
        if (
            path.suffix.lower() not in TEXT_EXTENSIONS
            and path.name.lower() != "dockerfile"
        ):
            continue

        source = safe_read(path)
        if not source or not source.strip():
            continue

        rel = path.relative_to(repo_root)
        lang = language_for(rel)

        structural: list[dict] = []
        if lang == "python":
            structural = _python_chunks(repo_name, rel, source)

        if not structural:
            structural = _generic_structural_chunks(repo_name, rel, source)

        if structural:
            chunks.extend(structural)
        else:
            # Fallback: sliding-window text blocks
            lines = source.splitlines()
            for start in range(1, len(lines) + 1, MAX_CHUNK_LINES - OVERLAP_LINES):
                end = min(len(lines), start + MAX_CHUNK_LINES - 1)
                content = "\n".join(lines[start - 1:end]).strip()
                if content:
                    chunks.append(_make_chunk_dict(
                        repo_name, rel, start, end, content,
                        "text_block", symbol=rel.name
                    ))

    # Deduplicate by chunk_id
    unique: dict[str, dict] = {}
    for chunk in chunks:
        unique[chunk["chunk_id"]] = chunk
    return list(unique.values())


# ---------------------------------------------------------------------------
# Repository Indexer
# ---------------------------------------------------------------------------

class RepositoryIndexer:
    """
    Orchestrates: clone → chunk → embed → FAISS index.

    All paths are derived from the GitHub URL; the user never enters them.
    """

    def __init__(
        self,
        base_dir: str = "data/repositories",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        # Embedding model is loaded lazily to avoid startup cost when reusing cache
        self._embedder: EmbeddingModel | None = None

    @property
    def embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = EmbeddingModel(model_name=self.model_name)
        return self._embedder

    def paths(self, repo_key: str) -> dict[str, Path]:
        root = self.base_dir / repo_key
        return {
            "root":     root,
            "source":   root / "source",
            "chunks":   root / "chunks.json",
            "index":    root / "repository.index",
            "metadata": root / "embedding_metadata.json",
            "manifest": root / "manifest.json",
        }

    def _is_cached(self, paths: dict[str, Path], normalized_url: str) -> bool:
        if not paths["manifest"].exists():
            return False
        try:
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            if (
                manifest.get("source_url") != normalized_url
                and manifest.get("repository_url") != normalized_url
            ):
                return False

            required = ["chunks", "index", "metadata"]
            if not all(paths[k].exists() for k in required):
                return False

            # Verify integrity: chunk count == metadata count == index size
            chunks = json.loads(paths["chunks"].read_text(encoding="utf-8"))
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            faiss_index = FAISSIndex.load(str(paths["index"]))

            if (
                len(chunks) != len(metadata)
                or len(chunks) != faiss_index.size()
                or faiss_index.size() == 0
            ):
                logger.warning("Cache validation failed: size mismatch.")
                return False

            return True
        except Exception as exc:
            logger.warning("Cache check failed: %s", exc)
            return False

    def build(self, github_url: str, force: bool = False) -> dict:
        """
        Build (or reuse) the index for a GitHub repository.

        Returns the manifest dict with all paths and metadata.
        """
        if not _GIT_AVAILABLE:
            raise RuntimeError(
                "GitPython is not installed. Run: pip install GitPython"
            )

        # ----------------------------------------------------------------
        # [1/6] Validate URL
        # ----------------------------------------------------------------
        print(f"[1/6] Validating repository...             ", end="\r", flush=True)
        normalized_url = normalize_github_url(github_url)
        repo_key, repo_name = repo_identity(github_url)
        paths = self.paths(repo_key)
        print(f"[1/6] Validating repository...             ✓")

        # ----------------------------------------------------------------
        # Cache check
        # ----------------------------------------------------------------
        if not force and self._is_cached(paths, normalized_url):
            logger.info("Reusing cached index for %s", repo_name)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["_cached"] = True
            return manifest
        elif paths["root"].exists() and not force:
            print("Existing index is incomplete. Rebuilding repository index...")

        # ----------------------------------------------------------------
        # [2/6] Clone
        # ----------------------------------------------------------------
        logger.info("[2/6] Cloning %s", normalized_url)
        print(f"[2/6] Cloning repository...                ", end="\r", flush=True)

        if paths["root"].exists():
            shutil.rmtree(paths["root"])
        paths["source"].mkdir(parents=True, exist_ok=True)

        try:
            _GitRepo.clone_from(normalized_url, paths["source"], depth=1)  # type: ignore[union-attr]
        except Exception as exc:
            if paths["root"].exists():
                shutil.rmtree(paths["root"], ignore_errors=True)
            raise RuntimeError(
                f"Could not clone repository.\n"
                f"Please verify the URL and repository access.\n"
                f"URL: {normalized_url}\n"
                f"Error: {exc}"
            ) from exc

        print(f"[2/6] Cloning repository...                ✓")

        # ----------------------------------------------------------------
        # [3/6] Chunk
        # ----------------------------------------------------------------
        logger.info("[3/6] Chunking repository")
        print(f"[3/6] Chunking repository...               ", end="\r", flush=True)

        chunks = chunk_repository(paths["source"], repo_name)
        if not chunks:
            if paths["root"].exists():
                shutil.rmtree(paths["root"], ignore_errors=True)
            raise RuntimeError(
                "Repository contains no supported source files.\n"
                "The repository may be empty or contain only binary files."
            )

        # Count unique files for display
        unique_files = len({c["file_path"] for c in chunks})

        # Save chunks as JSON (CodeChunk-compatible)
        paths["chunks"].write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[3/6] Chunking repository...               ✓ ({len(chunks):,} chunks from {unique_files:,} files)")

        # ----------------------------------------------------------------
        # [4/6] Embed
        # ----------------------------------------------------------------
        logger.info("[4/6] Generating embeddings")
        print(f"[4/6] Generating embeddings...             ", end="\r", flush=True)

        # Build embedding texts
        texts = [
            f"File: {c['file_path']}\n"
            f"Language: {c['language']}\n"
            f"Type: {c['chunk_type']}\n"
            + (f"Symbol: {c['symbol']}\n" if c.get('symbol') else "")
            + (f"Parent: {c['parent_symbol']}\n" if c.get('parent_symbol') else "")
            + (f"Imports: {', '.join(c['imports'])}\n" if c.get('imports') else "")
            + (f"Calls: {', '.join(c['calls'])}\n" if c.get('calls') else "")
            + f"Content:\n{c['content']}"
            for c in chunks
        ]

        import numpy as np
        vectors = self.embedder.encode_texts(texts, batch_size=32)
        vectors = np.asarray(vectors, dtype="float32")

        dim = int(vectors.shape[1])
        print(f"[4/6] Generating embeddings...             ✓ ({dim}d)")

        # ----------------------------------------------------------------
        # [5/6] FAISS index & metadata
        # ----------------------------------------------------------------
        logger.info("[5/6] Building FAISS index")
        print(f"[5/6] Building FAISS index...              ", end="\r", flush=True)

        faiss_index = FAISSIndex(dim)
        faiss_index.add(vectors)
        faiss_index.save(str(paths["index"]))

        metadata = [
            {
                "vector_index": i,
                "chunk_id":     c["chunk_id"],
                "repo":         c["repo"],
                "file_path":    c["file_path"],
                "file_name":    c["file_name"],
                "language":     c["language"],
                "chunk_type":   c["chunk_type"],
                "symbol":       c["symbol"],
                "line_start":   c["line_start"],
                "line_end":     c["line_end"],
                "github_reference": c["metadata"].get("github_reference"),
            }
            for i, c in enumerate(chunks)
        ]
        paths["metadata"].write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[5/6] Building FAISS index...              ✓ ({faiss_index.size():,} vectors)")

        # ----------------------------------------------------------------
        # [6/6] Manifest
        # ----------------------------------------------------------------
        import datetime
        display_url = (
            normalized_url[:-4]
            if normalized_url.endswith(".git")
            else normalized_url
        )

        manifest = {
            "repository_url":     normalized_url,
            "repository_hash":    repo_key,
            "repo_key":           repo_key,
            "repo_name":          repo_name,
            "source_url":         normalized_url,
            "display_url":        display_url,
            "indexed_at":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "files_indexed":      unique_files,
            "chunk_count":        len(chunks),
            "embedding_model":    self.model_name,
            "embedding_dimension": dim,
            "index_type":         "FAISS",
            "version":            "1.0",
            "index_path":         str(paths["index"]),
            "chunks_path":        str(paths["chunks"]),
            "metadata_path":      str(paths["metadata"]),
        }
        paths["manifest"].write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[6/6] Repository ready.                    ✓")

        manifest["_cached"] = False
        return manifest
