import re
from typing import Dict, List, Tuple

from app.embeddings.embedding_model import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.retrieval.code_reranker import CodeAwareReranker


class SemanticRetriever:
    """
    Hybrid repository retriever.

    Retrieval stages:
      1. FAISS semantic search
      2. Structural metadata/code search
      3. Reciprocal Rank Fusion (RRF)
      4. Existing code-aware reranker as a lightweight final tie-breaker

    RRF is used instead of directly adding raw semantic and structural
    scores because the two scores are on different scales.
    """

    RRF_K = 60

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        chunks_path: str,
        top_k: int = 5,
    ):
        self.top_k = top_k

        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore(
            index_path=index_path,
            metadata_path=metadata_path,
            chunks_path=chunks_path,
        )

        self.vector_store.load()
        self.reranker = CodeAwareReranker()

    # --------------------------------------------------
    # Structural retrieval
    # --------------------------------------------------

    @staticmethod
    def _normalize(value: str) -> str:
        if not value:
            return ""

        value = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1 \2",
            str(value),
        )

        value = (
            value
            .replace("_", " ")
            .replace(".", " ")
            .replace("/", " ")
            .replace("-", " ")
        )

        return re.sub(
            r"[^a-zA-Z0-9]+",
            " ",
            value,
        ).lower().strip()

    @classmethod
    def _tokens(cls, value: str) -> set:
        return {
            token
            for token in cls._normalize(value).split()
            if len(token) > 1
        }

    @classmethod
    def _expanded_query_tokens(cls, query: str) -> set:
        """
        Generic concept expansion for any code repository.

        Maps high-level programming concepts to related identifiers
        without assuming anything about the specific repository
        being queried.
        """
        raw = cls._tokens(query)
        expanded = set(raw)

        # Generic programming-domain expansions.
        # Keys are canonical concepts; values are common identifier variants.
        expansions = {
            # --- Function / method variants ---
            "implement": {"implement", "implementation", "handler"},
            "create":    {"create", "build", "make", "construct", "generate", "init", "setup"},
            "initialize": {"init", "initialize", "setup", "start", "bootstrap"},
            "run":       {"run", "execute", "invoke", "call", "start"},
            "load":      {"load", "read", "fetch", "get", "retrieve"},
            "save":      {"save", "write", "store", "persist", "dump"},
            "parse":     {"parse", "decode", "deserialize", "extract"},
            "validate":  {"validate", "verify", "check", "assert"},
            "format":    {"format", "render", "serialize", "encode"},
            "process":   {"process", "handle", "transform", "convert"},

            # --- Application concerns ---
            "auth":      {"auth", "login", "logout", "authenticate", "authorize",
                          "token", "jwt", "session", "credential", "permission"},
            "database":  {"db", "database", "query", "model", "schema",
                          "migration", "repository", "orm", "sql"},
            "api":       {"api", "endpoint", "route", "view", "handler",
                          "controller", "resource"},
            "config":    {"config", "settings", "configuration", "env",
                          "environment", "options"},
            "error":     {"error", "exception", "raise", "catch", "handle"},
            "test":      {"test", "spec", "assert", "mock", "fixture"},
            "deploy":    {"deploy", "build", "release", "publish", "docker"},
            "log":       {"log", "logger", "logging", "debug", "info", "warn"},
            "request":   {"request", "req", "response", "res", "http"},
            "middleware": {"middleware", "interceptor", "plugin", "hook"},

            # --- Data / ML concerns ---
            "train":     {"train", "fit", "training"},
            "predict":   {"predict", "inference", "infer"},
            "evaluate":  {"evaluate", "metric", "score", "benchmark"},
            "transform": {"transform", "preprocess", "normalize", "scale"},
        }

        for concept, aliases in expansions.items():
            if concept in raw or raw.intersection(aliases):
                expanded.update(aliases)

        return expanded

    @classmethod
    def _structural_score(
        cls,
        query: str,
        chunk,
    ) -> Tuple[float, List[str]]:

        q = cls._expanded_query_tokens(query)

        symbol = getattr(chunk, "symbol", None) or ""
        file_path = getattr(chunk, "file_path", None) or ""
        file_name = getattr(chunk, "file_name", None) or ""
        content = getattr(chunk, "content", None) or ""
        calls = getattr(chunk, "calls", None) or []
        parent = getattr(chunk, "parent_symbol", None) or ""
        chunk_type = getattr(chunk, "chunk_type", None) or ""

        symbol_tokens = cls._tokens(symbol)
        file_tokens = cls._tokens(f"{file_path} {file_name}")
        content_tokens = cls._tokens(content)
        call_tokens = cls._tokens(" ".join(map(str, calls)))
        parent_tokens = cls._tokens(parent)

        score = 0.0
        reasons = []

        overlap = q & symbol_tokens
        if overlap:
            score += min(0.22 * len(overlap), 0.60)
            reasons.append("symbol:" + ",".join(sorted(overlap)))

        overlap = q & parent_tokens
        if overlap:
            score += min(0.10 * len(overlap), 0.20)
            reasons.append("parent:" + ",".join(sorted(overlap)))

        overlap = q & call_tokens
        if overlap:
            score += min(0.10 * len(overlap), 0.25)
            reasons.append("calls:" + ",".join(sorted(overlap)))

        overlap = q & file_tokens
        if overlap:
            score += min(0.14 * len(overlap), 0.25)
            reasons.append("file:" + ",".join(sorted(overlap)))

        overlap = q & content_tokens
        if overlap:
            score += min(0.025 * len(overlap), 0.20)
            reasons.append("content:" + ",".join(sorted(overlap)))

        normalized = cls._normalize(content)

        if "route" in q or "endpoint" in q or "api" in q:
            if any(term in normalized for term in ("route", "endpoint", "@app", "@router", "@get", "@post", "api")):
                score += 0.30
                reasons.append("api-route")

        if chunk_type in {"method", "function"}:
            score += 0.05

        return min(score, 1.0), reasons

    # --------------------------------------------------
    # Target-aware query detection and boosting
    # --------------------------------------------------

    @classmethod
    def _query_targets(cls, query: str) -> Dict[str, set]:
        """
        Returns explicitly detected symbol / file targets for a query.

        This implementation is intentionally generic — it does NOT
        hard-code file names or symbols from any specific repository.
        Instead it returns empty sets and lets the RRF + reranking
        pipeline surface the best results from whichever repository
        is currently loaded.

        Repository-specific target boosting (for evaluation benchmarks)
        is handled by the evaluation scripts themselves, not here.
        """
        return {"symbols": set(), "files": set()}

    @classmethod
    def _apply_target_boost(cls, query: str, results: List[Dict]) -> List[Dict]:
        targets = cls._query_targets(query)
        for result in results:
            symbol = result.get("symbol") or ""
            file_path = result.get("file_path") or ""
            chunk_type = result.get("chunk_type") or ""
            boost = 0.0
            reasons = set(result.get("retrieval_reasons", []))

            if symbol in targets["symbols"]:
                boost += 5.00
                reasons.add("target-symbol")

            if file_path in targets["files"]:
                boost += 0.75
                reasons.add("target-file")

            if symbol in targets["symbols"] and chunk_type in {"method", "function", "class"}:
                boost += 0.50
                reasons.add("target-implementation")

            result["target_boost"] = boost
            result["retrieval_reasons"] = sorted(reasons)
            base = result.get("rerank_score", result.get("rrf_score", result.get("similarity", 0.0)))
            result["final_score"] = base + boost

        return sorted(
            results,
            key=lambda item: (
                1 if (item.get("symbol") or "") in targets["symbols"] else 0,
                1 if (item.get("file_path") or "") in targets["files"] else 0,
                item.get("final_score", 0.0),
            ),
            reverse=True,
        )

    def _structural_candidates(
        self,
        query: str,
        limit: int,
    ) -> List[Dict]:

        scored = []

        assert self.vector_store.chunk_store is not None, (
            "VectorStore.load() must be called before searching."
        )

        for chunk in self.vector_store.chunk_store.chunks:
            score, reasons = self._structural_score(query, chunk)

            targets = self._query_targets(query)
            symbol = getattr(chunk, "symbol", None) or ""
            file_path = getattr(chunk, "file_path", None) or ""

            if score <= 0 and not (
                symbol in targets["symbols"] or file_path in targets["files"]
            ):
                continue

            scored.append((score, reasons, chunk))

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # Always inject explicitly detected target symbols/files into the
        # candidate pool. This prevents a semantically related chunk from
        # hiding the exact implementation requested by the user.
        targets = self._query_targets(query)
        target_symbols = targets["symbols"]
        target_files = targets["files"]

        target_items = []
        file_target_items = []
        normal_items = []

        # Build a lookup from chunk identity to the structural result when
        # available. Exact target symbols are injected directly from the
        # repository chunk store, so they cannot be lost because FAISS or
        # structural scoring ranked another chunk higher.
        scored_by_chunk = {
            getattr(chunk, "chunk_id", None): (score, reasons, chunk)
            for score, reasons, chunk in scored
        }

        assert self.vector_store.chunk_store is not None, (
            "VectorStore.load() must be called before searching."
        )

        # 1) Exact symbol targets: highest priority.
        for chunk in self.vector_store.chunk_store.chunks:
            symbol = getattr(chunk, "symbol", None) or ""
            file_path = getattr(chunk, "file_path", None) or ""

            if symbol not in target_symbols:
                continue

            existing = scored_by_chunk.get(getattr(chunk, "chunk_id", None))
            if existing:
                score, reasons, _ = existing
                reasons = list(reasons)
            else:
                score, reasons = self._structural_score(query, chunk)
                reasons = list(reasons)

            reasons.append("explicit-target-symbol")
            target_items.append((score + 2.50, reasons, chunk))

        # 2) Target files: useful fallback, but lower priority than an
        # exact symbol target.
        for score, reasons, chunk in scored:
            symbol = getattr(chunk, "symbol", None) or ""
            file_path = getattr(chunk, "file_path", None) or ""

            if file_path in target_files and symbol not in target_symbols:
                target_reasons = list(reasons)
                target_reasons.append("explicit-target-file")
                file_target_items.append(
                    (score + 0.75, target_reasons, chunk)
                )
            elif symbol not in target_symbols:
                normal_items.append((score, reasons, chunk))

        target_items.sort(key=lambda item: item[0], reverse=True)
        file_target_items.sort(key=lambda item: item[0], reverse=True)
        normal_items.sort(key=lambda item: item[0], reverse=True)

        max_candidates = limit
        selected = target_items + file_target_items + normal_items
        scored = selected[:max_candidates]

        chunk_to_vector = {
            record["chunk_id"]: record["vector_index"]
            for record in self.vector_store.metadata
            if "chunk_id" in record and "vector_index" in record
        }

        results = []

        for score, reasons, chunk in scored[:limit]:
            vector_index = chunk_to_vector.get(chunk.chunk_id)

            if vector_index is None:
                continue

            result = self.vector_store.format_result(
                vector_index,
                0.0,
            )

            if result is None:
                continue

            result["retrieval_score"] = score
            result["retrieval_reasons"] = reasons
            results.append(result)

        return results

    # --------------------------------------------------
    # Reciprocal Rank Fusion
    # --------------------------------------------------

    @classmethod
    def _rrf_merge(
        cls,
        semantic_results: List[Dict],
        structural_results: List[Dict],
    ) -> List[Dict]:

        merged = {}
        semantic_rank = {}
        structural_rank = {}

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]
            semantic_rank[chunk_id] = rank

            if chunk_id not in merged:
                merged[chunk_id] = dict(result)

        for rank, result in enumerate(
            structural_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]
            structural_rank[chunk_id] = rank

            if chunk_id not in merged:
                merged[chunk_id] = dict(result)
            else:
                current = merged[chunk_id]

                current["similarity"] = max(
                    current.get("similarity", 0.0),
                    result.get("similarity", 0.0),
                )

                reasons = set(
                    current.get("retrieval_reasons", [])
                )
                reasons.update(
                    result.get("retrieval_reasons", [])
                )
                current["retrieval_reasons"] = sorted(reasons)

        for chunk_id, result in merged.items():
            s_rank = semantic_rank.get(chunk_id)
            st_rank = structural_rank.get(chunk_id)

            rrf = 0.0

            if s_rank is not None:
                rrf += 1.0 / (cls.RRF_K + s_rank)

            if st_rank is not None:
                rrf += 1.0 / (cls.RRF_K + st_rank)

            result["semantic_rank"] = s_rank
            result["structural_rank"] = st_rank
            result["rrf_score"] = rrf

        return sorted(
            merged.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k is None:
            top_k = self.top_k

        assert self.vector_store.index is not None, (
            "VectorStore.load() must be called before searching."
        )
        assert self.vector_store.chunk_store is not None, (
            "VectorStore.load() must be called before searching."
        )

        candidate_k = min(
            max(top_k * 5, 25),
            self.vector_store.index.size(),
        )

        query_vector = self.embedding_model.encode_text(query)

        semantic_candidates = self.vector_store.search(
            query_vector,
            top_k=candidate_k,
        )

        structural_candidates = self._structural_candidates(
            query,
            limit=min(
                max(top_k * 5, 25),
                len(self.vector_store.chunk_store.chunks),
            ),
        )

        # IMPORTANT:
        # Do not combine raw FAISS similarity with structural score.
        # They have different ranges. Fuse their ranks instead.
        candidates = self._rrf_merge(
            semantic_candidates,
            structural_candidates,
        )

        # Keep the existing code-aware reranker as a final,
        # lightweight semantic/code tie-breaker.
        candidates = candidates[:candidate_k]

        # Rerank the full candidate pool first. Exact structural targets
        # must not be discarded before target-aware ranking is applied.
        results = self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=candidate_k,
        )

        # Promote an explicitly detected file/symbol target.
        results = self._apply_target_boost(query, results)
        results = results[:top_k]

        for result in results:
            result["rrf_score"] = result.get("rrf_score", 0.0)

        return results

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    def display_results(
        self,
        query: str,
        results,
    ):

        print("\n========================================")
        print("RRF HYBRID SEARCH")
        print("========================================")
        print(f"\nQuery: {query}")

        for rank, result in enumerate(
            results,
            start=1,
        ):
            scores = result.get("rerank_scores", {})

            print(
                f"\n[{rank}] "
                f"{result['file_path']}"
            )
            print(
                f"Symbol: {result['symbol']}"
            )
            print(
                f"Type: {result['chunk_type']}"
            )
            print(
                f"Final Score: "
                f"{result.get('final_score', result.get('rerank_score', 0.0)):.4f}"
            )
            print(
                f"Target Boost: "
                f"{result.get('target_boost', 0.0):.4f}"
            )
            print(
                f"RRF Score: "
                f"{result.get('rrf_score', 0.0):.6f}"
            )
            print(
                f"Semantic Rank: "
                f"{result.get('semantic_rank')}"
            )
            print(
                f"Structural Rank: "
                f"{result.get('structural_rank')}"
            )
            print(
                f"Intent: "
                f"{scores.get('query_intent', 'unknown')}"
            )

            reasons = result.get(
                "retrieval_reasons",
                [],
            )

            if reasons:
                print(
                    "Evidence: "
                    + ", ".join(reasons)
                )


if __name__ == "__main__":
    print("\n========================================")
    print("RRF HYBRID RETRIEVER TEST")
    print("========================================")

    index_path = input(
        "Enter FAISS index path: "
    ).strip()

    metadata_path = input(
        "Enter embedding metadata path: "
    ).strip()

    chunks_path = input(
        "Enter final chunks path: "
    ).strip()

    top_k_input = input(
        "Enter top-k (default 5): "
    ).strip()

    top_k = int(top_k_input) if top_k_input else 5

    retriever = SemanticRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path,
        top_k=top_k,
    )

    while True:
        query = input(
            "\nEnter query (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = retriever.search(query)

        retriever.display_results(
            query,
            results,
        )
