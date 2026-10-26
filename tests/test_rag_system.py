"""
Unit tests for the GitHub RAG system.

Tests:
  - URL parsing and normalization
  - Repository identity (stable hashing)
  - File filtering (ignored dirs, extensions)
  - Chunk schema compatibility with CodeChunk
  - Repository isolation (different URLs → different keys)
  - Cache detection

Run:
    python -m pytest tests/ -v
"""
import hashlib
import json
import sys
from pathlib import Path

# Add project root so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.repo_indexer import (
    normalize_github_url,
    repo_identity,
    language_for,
    chunk_repository,
    IGNORED_DIRS,
    TEXT_EXTENSIONS,
    BINARY_EXTENSIONS,
    _make_chunk_dict,
)


# ---------------------------------------------------------------------------
# URL Parsing
# ---------------------------------------------------------------------------

class TestNormalizeGitHubUrl:

    def test_plain_url(self):
        result = normalize_github_url("https://github.com/owner/repo")
        assert result == "https://github.com/owner/repo.git"

    def test_git_url(self):
        result = normalize_github_url("https://github.com/owner/repo.git")
        assert result == "https://github.com/owner/repo.git"

    def test_trailing_slash(self):
        result = normalize_github_url("https://github.com/owner/repo/")
        assert result == "https://github.com/owner/repo.git"

    def test_www_prefix(self):
        result = normalize_github_url("https://www.github.com/owner/repo")
        assert result == "https://github.com/owner/repo.git"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_github_url("")

    def test_no_https_raises(self):
        with pytest.raises(ValueError, match="https"):
            normalize_github_url("github.com/owner/repo")

    def test_non_github_raises(self):
        with pytest.raises(ValueError, match="github.com"):
            normalize_github_url("https://gitlab.com/owner/repo")

    def test_too_short_path_raises(self):
        with pytest.raises(ValueError):
            normalize_github_url("https://github.com/owner")

    def test_empty_owner_raises(self):
        with pytest.raises(ValueError):
            normalize_github_url("https://github.com//repo")

    def test_case_insensitive_git_suffix(self):
        result = normalize_github_url("https://github.com/owner/Repo.GIT")
        # Should strip .GIT and normalise
        assert result.endswith(".git")
        assert "owner" in result


class TestRepoIdentity:

    def test_same_url_same_key(self):
        k1, _ = repo_identity("https://github.com/owner/repo")
        k2, _ = repo_identity("https://github.com/owner/repo.git")
        assert k1 == k2

    def test_different_url_different_key(self):
        k1, _ = repo_identity("https://github.com/owner/repo-a")
        k2, _ = repo_identity("https://github.com/owner/repo-b")
        assert k1 != k2

    def test_key_length(self):
        key, _ = repo_identity("https://github.com/owner/repo")
        assert len(key) == 16

    def test_repo_name(self):
        _, name = repo_identity("https://github.com/owner/repo")
        assert name == "owner/repo"

    def test_stability(self):
        """Same URL always yields the same key (deterministic)."""
        k1, _ = repo_identity("https://github.com/pallets/flask")
        k2, _ = repo_identity("https://github.com/pallets/flask")
        assert k1 == k2

    def test_isolation(self):
        """Flask and FastAPI have different keys."""
        k_flask, _ = repo_identity("https://github.com/pallets/flask")
        k_fastapi, _ = repo_identity("https://github.com/tiangolo/fastapi")
        assert k_flask != k_fastapi


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestLanguageFor:

    @pytest.mark.parametrize("ext,expected", [
        (".py", "python"),
        (".js", "javascript"),
        (".ts", "typescript"),
        (".java", "java"),
        (".go", "go"),
        (".rs", "rust"),
        (".md", "markdown"),
        (".yaml", "yaml"),
        (".json", "json"),
        (".sh", "shell"),
        (".html", "html"),
        (".css", "css"),
        (".rb", "ruby"),
        (".php", "php"),
    ])
    def test_extension_mapping(self, ext, expected, tmp_path):
        p = tmp_path / f"file{ext}"
        assert language_for(p) == expected

    def test_dockerfile(self, tmp_path):
        p = tmp_path / "Dockerfile"
        assert language_for(p) == "dockerfile"


# ---------------------------------------------------------------------------
# File filtering
# ---------------------------------------------------------------------------

class TestFileFiltering:

    def test_ignored_dirs_present(self):
        for d in [".git", "__pycache__", "node_modules", "venv", ".venv",
                  "dist", "build", "target", ".github"]:
            assert d in IGNORED_DIRS, f"'{d}' not in IGNORED_DIRS"

    def test_source_extensions_present(self):
        for ext in [".py", ".js", ".ts", ".java", ".go", ".rs", ".md",
                    ".yaml", ".json", ".html", ".css"]:
            assert ext in TEXT_EXTENSIONS, f"'{ext}' not in TEXT_EXTENSIONS"

    def test_binary_extensions_excluded(self):
        for ext in [".pyc", ".so", ".dll", ".exe", ".jpg", ".png",
                    ".zip", ".tar", ".gz"]:
            assert ext in BINARY_EXTENSIONS, f"'{ext}' not in BINARY_EXTENSIONS"
            assert ext not in TEXT_EXTENSIONS, f"'{ext}' should NOT be in TEXT_EXTENSIONS"


# ---------------------------------------------------------------------------
# Chunk schema compatibility
# ---------------------------------------------------------------------------

class TestChunkSchema:
    """Verify that _make_chunk_dict() produces CodeChunk-compatible dicts."""

    REQUIRED_FIELDS = [
        "chunk_id", "repo", "file_path", "file_name", "language",
        "chunk_type", "symbol", "line_start", "line_end", "content",
        "imports", "calls", "dependencies", "related_files",
        "parent_symbol", "metadata",
    ]

    def _make(self, tmp_path: Path) -> dict:
        rel = Path("src/utils.py")
        return _make_chunk_dict(
            repo_name="owner/repo",
            rel=rel,
            start=1,
            end=10,
            content="def foo(): pass",
            chunk_type="function",
            symbol="foo",
        )

    def test_all_required_fields_present(self, tmp_path):
        chunk = self._make(tmp_path)
        for field in self.REQUIRED_FIELDS:
            assert field in chunk, f"Missing required field: {field}"

    def test_imports_is_list(self, tmp_path):
        chunk = self._make(tmp_path)
        assert isinstance(chunk["imports"], list)

    def test_calls_is_list(self, tmp_path):
        chunk = self._make(tmp_path)
        assert isinstance(chunk["calls"], list)

    def test_dependencies_is_list(self, tmp_path):
        chunk = self._make(tmp_path)
        assert isinstance(chunk["dependencies"], list)

    def test_related_files_is_list(self, tmp_path):
        chunk = self._make(tmp_path)
        assert isinstance(chunk["related_files"], list)

    def test_metadata_has_github_reference(self, tmp_path):
        chunk = self._make(tmp_path)
        assert "github_reference" in chunk["metadata"]
        assert "github.com" in chunk["metadata"]["github_reference"]

    def test_chunk_id_is_deterministic(self, tmp_path):
        rel = Path("src/utils.py")
        c1 = _make_chunk_dict("owner/repo", rel, 1, 10, "def foo(): pass", "function", "foo")
        c2 = _make_chunk_dict("owner/repo", rel, 1, 10, "def foo(): pass", "function", "foo")
        assert c1["chunk_id"] == c2["chunk_id"]

    def test_different_repos_different_chunk_ids(self, tmp_path):
        rel = Path("src/utils.py")
        c1 = _make_chunk_dict("owner/repo-a", rel, 1, 10, "def foo(): pass", "function", "foo")
        c2 = _make_chunk_dict("owner/repo-b", rel, 1, 10, "def foo(): pass", "function", "foo")
        assert c1["chunk_id"] != c2["chunk_id"]

    def test_repo_field_set_correctly(self, tmp_path):
        rel = Path("src/utils.py")
        chunk = _make_chunk_dict("owner/my-repo", rel, 1, 10, "x = 1", "file", "utils.py")
        assert chunk["repo"] == "owner/my-repo"


# ---------------------------------------------------------------------------
# chunk_repository integration (uses real files)
# ---------------------------------------------------------------------------

class TestChunkRepository:

    def test_python_file_produces_chunks(self, tmp_path):
        (tmp_path / "main.py").write_text(
            "def hello():\n    return 'world'\n\nclass Foo:\n    def bar(self):\n        pass\n"
        )
        chunks = chunk_repository(tmp_path, "owner/repo")
        assert len(chunks) > 0

    def test_chunks_have_correct_repo(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        chunks = chunk_repository(tmp_path, "owner/my-repo")
        for c in chunks:
            assert c["repo"] == "owner/my-repo"

    def test_ignored_dirs_skipped(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.pyc").write_bytes(b"garbage")
        (tmp_path / "real.py").write_text("def foo(): pass\n")
        chunks = chunk_repository(tmp_path, "owner/repo")
        paths = [c["file_path"] for c in chunks]
        assert not any("__pycache__" in p for p in paths)

    def test_empty_repo_returns_empty(self, tmp_path):
        chunks = chunk_repository(tmp_path, "owner/repo")
        assert chunks == []

    def test_deduplication(self, tmp_path):
        (tmp_path / "dup.py").write_text("def foo(): pass\n")
        chunks = chunk_repository(tmp_path, "owner/repo")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_symbols_extracted_from_python(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class MyClass:\n    def my_method(self):\n        pass\n"
        )
        chunks = chunk_repository(tmp_path, "owner/repo")
        symbols = {c["symbol"] for c in chunks if c.get("symbol")}
        assert "MyClass" in symbols
        assert "MyClass.my_method" in symbols

    def test_markdown_chunks_created(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "# Introduction\nHello world.\n\n# Usage\nRun it.\n"
        )
        chunks = chunk_repository(tmp_path, "owner/repo")
        md_chunks = [c for c in chunks if c["chunk_type"] == "markdown_section"]
        assert len(md_chunks) >= 2

    def test_node_modules_excluded(self, tmp_path):
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {};")
        (tmp_path / "app.js").write_text("function main() {}")
        chunks = chunk_repository(tmp_path, "owner/repo")
        paths = [c["file_path"] for c in chunks]
        assert not any("node_modules" in p for p in paths)
        assert any("app.js" in p for p in paths)


# ---------------------------------------------------------------------------
# Repository isolation
# ---------------------------------------------------------------------------

class TestRepositoryIsolation:

    def test_different_urls_produce_different_dirs(self):
        k1, _ = repo_identity("https://github.com/owner/repo-a")
        k2, _ = repo_identity("https://github.com/owner/repo-b")
        assert k1 != k2, "Different repos must produce different storage directories"

    def test_same_url_normalizations_same_key(self):
        variants = [
            "https://github.com/pallets/flask",
            "https://github.com/pallets/flask.git",
            "https://github.com/pallets/flask/",
            "https://www.github.com/pallets/flask",
        ]
        keys = [repo_identity(v)[0] for v in variants]
        assert len(set(keys)) == 1, "All URL variants should produce the same key"


# ---------------------------------------------------------------------------
# GroqClient Tests
# ---------------------------------------------------------------------------

from app.llm.llm_client import GroqClient, DEFAULT_GROQ_MODEL
from unittest.mock import MagicMock, patch


class TestGroqClient:

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        client = GroqClient(api_key=None)
        with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
            client.generate("hello")

    def test_empty_prompt_raises(self):
        client = GroqClient(api_key="mock_key")
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            client.generate("")

    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        client = GroqClient(api_key="mock_key")
        assert client.model == DEFAULT_GROQ_MODEL

    def test_custom_model_from_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
        client = GroqClient(api_key="mock_key")
        assert client.model == "llama-3.1-8b-instant"

    @patch("groq.Groq")
    def test_successful_generate(self, mock_groq_class):
        mock_instance = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Model training is in src/train.py [1]."
        mock_response = MagicMock(choices=[mock_choice])
        mock_instance.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_instance

        client = GroqClient(api_key="mock_key")
        result = client.generate("Where is training?")
        assert result == "Model training is in src/train.py [1]."


# ---------------------------------------------------------------------------
# Cache Validation Tests
# ---------------------------------------------------------------------------

from app.repo_indexer import RepositoryIndexer


class TestCacheValidation:

    def test_missing_manifest_is_not_cached(self, tmp_path):
        indexer = RepositoryIndexer(base_dir=str(tmp_path))
        paths = indexer.paths("testkey")
        assert not indexer._is_cached(paths, "https://github.com/owner/repo.git")

    def test_incomplete_artifacts_is_not_cached(self, tmp_path):
        indexer = RepositoryIndexer(base_dir=str(tmp_path))
        paths = indexer.paths("testkey")
        paths["root"].mkdir(parents=True, exist_ok=True)
        paths["manifest"].write_text(json.dumps({"source_url": "https://github.com/owner/repo.git"}))
        # missing chunks, index, metadata
        assert not indexer._is_cached(paths, "https://github.com/owner/repo.git")


# ---------------------------------------------------------------------------
# RAG Pipeline & RepositoryRAG Tests
# ---------------------------------------------------------------------------

from app.rag_pipeline import RAGPipeline
from app.repository_rag import RepositoryRAG


class TestRAGPipelineUnits:

    def test_validate_citations_pass(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        result = pipeline.validate_citations("Answer with citation [1] and [2].", number_of_sources=3)
        assert result["valid"] is True
        assert result["citations_found"] == [1, 2]
        assert result["invalid_citations"] == []

    def test_validate_citations_fail_out_of_range(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        result = pipeline.validate_citations("Answer with invalid citation [5].", number_of_sources=3)
        assert result["valid"] is False
        assert result["invalid_citations"] == [5]

    def test_build_numbered_context(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        chunks = [
            {
                "file_path": "src/train.py",
                "symbol": "train_model",
                "chunk_type": "function",
                "line_start": 10,
                "line_end": 25,
                "content": "def train_model(): pass",
            }
        ]
        context = pipeline.build_numbered_context(chunks)
        assert "[1]" in context
        assert "File: src/train.py" in context
        assert "Symbol: train_model" in context
        assert "Lines: 10-25" in context
        assert "def train_model(): pass" in context


class TestRepositoryRAGAPI:

    def test_uninitialized_ask_raises(self):
        rag = RepositoryRAG("https://github.com/owner/repo")
        with pytest.raises(RuntimeError, match="Repository not initialized"):
            rag.ask("Where is main?")

    def test_empty_question_raises(self):
        rag = RepositoryRAG("https://github.com/owner/repo")
        rag._pipeline = MagicMock()
        with pytest.raises(ValueError, match="Question cannot be empty"):
            rag.ask("")

