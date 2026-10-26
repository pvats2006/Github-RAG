"""
High-level RepositoryRAG API for automated repository indexing and querying.

Usage:
    rag = RepositoryRAG("https://github.com/owner/repo")
    info = rag.initialize()       # Clones / chunks / embeds / indexes (once)
    result = rag.ask("Where is model training implemented?")
    print(result["answer"])
    rag.close()

The caller never manually enters FAISS paths, metadata paths, or chunk paths.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.repo_indexer import RepositoryIndexer, repo_identity, normalize_github_url
from app.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 8


class RepositoryRAG:
    """
    End-to-end Code Repository RAG pipeline.

    Wraps RepositoryIndexer + RAGPipeline behind a simple API:
        initialize() -> clone / index / embed / load (once)
        ask(q)       -> retrieve / rerank / generate / validate / confidence
    """

    def __init__(
        self,
        repo_url: str,
        base_dir: str = "data/repositories",
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: Optional[str] = None,
        top_k: int = TOP_K,
    ):
        self.repo_url = repo_url
        self.base_dir = base_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k

        self._manifest: Optional[Dict[str, Any]] = None
        self._pipeline: Optional[RAGPipeline] = None

    # ------------------------------------------------------------------
    # INITIALIZE / PREPARE
    # ------------------------------------------------------------------

    def initialize(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Clone, chunk, embed, and index the repository (if not already indexed).
        Loads the FAISS index, metadata, chunks, and retriever ONCE for all queries.

        Returns summary information dict.
        """
        indexer = RepositoryIndexer(
            base_dir=self.base_dir,
            model_name=self.embedding_model,
        )

        repo_key, _ = repo_identity(self.repo_url)
        paths = indexer.paths(repo_key)
        normalized = normalize_github_url(self.repo_url)
        already_cached = indexer._is_cached(paths, normalized) and not force_reindex

        manifest = indexer.build(self.repo_url, force=force_reindex)
        self._manifest = manifest

        # Initialize the pipeline once paths are resolved
        self._pipeline = RAGPipeline(
            index_path=manifest["index_path"],
            metadata_path=manifest["metadata_path"],
            chunks_path=manifest["chunks_path"],
            top_k=self.top_k,
            model=self.llm_model,
        )

        return {
            "repository_url":     manifest.get("display_url", manifest.get("source_url")),
            "repository_id":      manifest.get("repository_hash", manifest.get("repo_key")),
            "repo_name":          manifest.get("repo_name", ""),
            "files_indexed":      manifest.get("files_indexed", 0),
            "chunks_indexed":     manifest.get("chunk_count", 0),
            "embedding_dimension": manifest.get("embedding_dimension", 384),
            "embedding_model":    manifest.get("embedding_model", self.embedding_model),
            "llm_model":          self.llm_model or self._pipeline.llm.model,
            "cached":             already_cached,
        }

    def prepare(self, force_reindex: bool = False) -> Dict[str, Any]:
        """Alias for initialize()."""
        return self.initialize(force_reindex=force_reindex)

    # ------------------------------------------------------------------
    # ASK
    # ------------------------------------------------------------------

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Execute full RAG pipeline for a single question using the loaded index and retriever.
        Never reloads models or indexes on repeated calls.

        Returns structured result dict.
        """
        if self._pipeline is None:
            raise RuntimeError("Repository not initialized. Call initialize() first.")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        return self._pipeline.answer(question)

    # ------------------------------------------------------------------
    # CLOSE / CLEANUP
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release pipeline resources."""
        self._pipeline = None
        self._manifest = None

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        return self._manifest
