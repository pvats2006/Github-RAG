import re
from typing import Dict, List


class CodeAwareReranker:
    """
    Code-aware reranker.

    FAISS finds semantic candidates.
    This class decides which candidate is more useful
    for a code-related question.
    """

    CHUNK_TYPE_WEIGHTS = {
        "method": 1.00,
        "function": 0.95,
        "class": 0.85,
        "file": 0.55,
        "html_section": 0.45,
        "html_file": 0.35,
        "markdown_section": 0.20,
        "text_block": 0.15,
        "text_file": 0.15,
    }

    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were",
        "be", "been", "being", "how", "what", "where",
        "when", "why", "which", "who", "does", "do",
        "did", "can", "could", "would", "should", "will",
        "this", "that", "these", "those", "it", "its",
        "in", "on", "at", "to", "of", "for", "from",
        "with", "by", "and", "or", "as", "about",
        "into", "through",
        "project", "repository", "application", "code",
        "file", "files",
    }

    CODE_INTENT_WORDS = {
        "implement", "implementation", "implemented",
        "function", "functions", "method", "methods",
        "logic", "handle", "handles", "handling",
        "process", "processing", "train", "training",
        "trained", "predict", "prediction", "predicting",
        "transform", "transformation", "evaluate",
        "evaluation", "select", "selection", "save", "load",
        "validate", "validation", "call", "calls", "create",
        "creates", "calculate", "configure", "initialize",
        "initialization", "defined", "definition",
    }

    DOCUMENTATION_INTENT_WORDS = {
        "feature", "features", "overview", "installation",
        "install", "usage", "author", "description", "purpose",
    }

    NORMALIZATION = {
        "implemented": "implement",
        "implementation": "implement",
        "implementing": "implement",
        "training": "train",
        "trained": "train",
        "trainer": "train",
        "predictions": "predict",
        "prediction": "predict",
        "predicting": "predict",
        "transformations": "transform",
        "transformation": "transform",
        "evaluating": "evaluate",
        "evaluation": "evaluate",
        "selected": "select",
        "selection": "select",
        "handled": "handle",
        "handling": "handle",
        "functions": "function",
        "methods": "method",
        "calls": "call",
        "features": "feature",
        "defined": "define",
        "definition": "define",
    }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*",
            text.lower(),
        )

    @classmethod
    def normalize_token(cls, token: str) -> str:
        return cls.NORMALIZATION.get(
            token.lower(),
            token.lower(),
        )

    @classmethod
    def query_tokens(cls, query: str) -> set:
        tokens = {
            cls.normalize_token(token)
            for token in cls.tokenize(query)
        }
        return tokens - cls.STOP_WORDS

    @classmethod
    def content_tokens(cls, text: str) -> set:
        return {
            cls.normalize_token(token)
            for token in cls.tokenize(text)
        }

    @classmethod
    def detect_query_intent(cls, query: str) -> str:
        tokens = cls.query_tokens(query)
        if tokens & cls.CODE_INTENT_WORDS:
            return "code"
        if tokens & cls.DOCUMENTATION_INTENT_WORDS:
            return "documentation"
        return "general"

    @classmethod
    def identifier_score(cls, query: str, result: Dict) -> float:
        symbol = result.get("symbol")
        if not symbol:
            return 0.0

        symbol_lower = symbol.lower()
        query_lower = query.lower()

        if symbol_lower in query_lower:
            return 1.0

        symbol_parts = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*",
            symbol_lower,
        )
        query_tokens = cls.query_tokens(query)

        meaningful_parts = {
            cls.normalize_token(part)
            for part in symbol_parts
            if part not in cls.STOP_WORDS
        }

        if not meaningful_parts:
            return 0.0

        overlap = meaningful_parts & query_tokens

        return min(
            len(overlap) / len(meaningful_parts),
            1.0,
        )

    @classmethod
    def symbol_score(cls, query: str, result: Dict) -> float:
        score = cls.identifier_score(query, result)

        symbol = result.get("symbol")
        if not symbol:
            return score

        query_tokens = cls.query_tokens(query)

        strong_terms = {
            "train", "predict", "transform", "evaluate",
            "save", "load", "data", "model", "exception",
        }

        symbol_tokens = cls.content_tokens(
            symbol.replace(".", " ")
        )

        overlap = symbol_tokens & query_tokens
        strong_overlap = overlap & strong_terms

        if strong_overlap:
            score += 0.25

        return min(score, 1.0)

    @classmethod
    def content_score(cls, query: str, result: Dict) -> float:
        query_tokens = cls.query_tokens(query)
        content_tokens = cls.content_tokens(
            result.get("content", "")
        )

        if not query_tokens:
            return 0.0

        return len(
            query_tokens & content_tokens
        ) / len(query_tokens)

    @classmethod
    def action_score(cls, query: str, result: Dict) -> float:
        query_tokens = cls.query_tokens(query)
        actions = query_tokens & cls.CODE_INTENT_WORDS

        if not actions:
            return 0.0

        content_tokens = cls.content_tokens(
            result.get("content", "")
        )

        return len(
            actions & content_tokens
        ) / len(actions)

    # NEW:
    # Directly checks whether the action asked by the user
    # appears in the candidate's symbol name.
    #
    # Example:
    #   "How are models evaluated?"
    #       query action = evaluate
    #       symbol = evaluate_models
    #       -> strong match
    #
    # This is intentionally separate from semantic similarity.
    @classmethod
    def action_symbol_score(
        cls,
        query: str,
        result: Dict,
    ) -> float:

        query_tokens = cls.query_tokens(query)
        actions = query_tokens & cls.CODE_INTENT_WORDS

        symbol = result.get("symbol")
        if not actions or not symbol:
            return 0.0

        symbol_tokens = cls.content_tokens(
            symbol.replace(".", " ")
        )

        matched_actions = actions & symbol_tokens

        if not matched_actions:
            return 0.0

        # Exact normalized action ↔ symbol alignment.
        return len(matched_actions) / len(actions)

    @classmethod
    def code_content_score(cls, result: Dict) -> float:
        content = result.get("content", "").lower()

        if not content:
            return 0.0

        signals = [
            "def ", "class ", "return ", "import ",
            "from ", "self.", "try:", "except ",
        ]

        matches = sum(
            1 for signal in signals if signal in content
        )

        return min(matches / 4.0, 1.0)

    @classmethod
    def file_score(cls, query: str, result: Dict) -> float:
        query_tokens = cls.query_tokens(query)
        file_tokens = cls.content_tokens(
            result.get("file_path", "")
            .replace("/", " ")
            .replace(".", " ")
        )

        if not query_tokens or not file_tokens:
            return 0.0

        return min(
            len(query_tokens & file_tokens)
            / max(len(file_tokens), 1),
            1.0,
        )

    @classmethod
    def chunk_type_score(
        cls,
        result: Dict,
        intent: str,
    ) -> float:

        chunk_type = result.get("chunk_type") or ""

        if intent == "code":
            if chunk_type in {"method", "function"}:
                return 1.0
            if chunk_type == "class":
                return 0.85
            if chunk_type == "file":
                return 0.30
            if chunk_type == "markdown_section":
                return 0.05
            return 0.10

        if intent == "documentation":
            if chunk_type == "markdown_section":
                return 1.0
            if chunk_type == "file":
                return 0.50
            if chunk_type in {"method", "function"}:
                return 0.35

        return cls.CHUNK_TYPE_WEIGHTS.get(
            chunk_type,
            0.10,
        )

    @classmethod
    def intent_bonus(
        cls,
        intent: str,
        result: Dict,
    ) -> float:

        chunk_type = result.get("chunk_type") or ""

        if intent == "code":
            if chunk_type in {"method", "function"}:
                return 1.0
            if chunk_type == "class":
                return 0.70
            if chunk_type == "file":
                return 0.15
            return 0.0

        if intent == "documentation":
            if chunk_type == "markdown_section":
                return 1.0
            return 0.20

        return 0.20

    @classmethod
    def calculate_score(
        cls,
        query: str,
        result: Dict,
    ):
        intent = cls.detect_query_intent(query)

        semantic = float(
            result.get("similarity", 0.0)
        )

        symbol = cls.symbol_score(query, result)
        identifier = cls.identifier_score(query, result)
        content = cls.content_score(query, result)
        action = cls.action_score(query, result)
        action_symbol = cls.action_symbol_score(
            query,
            result,
        )
        code_content = cls.code_content_score(result)
        file_score = cls.file_score(query, result)
        chunk_type = cls.chunk_type_score(
            result,
            intent,
        )
        intent_score = cls.intent_bonus(
            intent,
            result,
        )

        # The new action_symbol signal gets meaningful weight.
        # Semantic similarity remains important, but an exact
        # code-action/symbol match can outrank a merely similar
        # generic method.
        final_score = (
            semantic * 0.25
            + identifier * 0.10
            + symbol * 0.10
            + action_symbol * 0.20
            + content * 0.10
            + action * 0.05
            + code_content * 0.05
            + file_score * 0.05
            + chunk_type * 0.05
            + intent_score * 0.05
        )

        # ------------------------------------------------------------------
        # File-chunk penalty
        #
        # When the query explicitly asks for a *function* or *method*,
        # the whole-file chunk for that file will score high on both
        # semantic similarity (it contains all the function names) and
        # content overlap. We apply a penalty so that individual
        # function / method chunks rank above their parent file chunk.
        #
        # Only applied when:
        #   - chunk_type is "file"
        #   - the query contains "function" or "method" OR has a code-action verb
        #   - intent is "code"
        # ------------------------------------------------------------------
        chunk_type_val = result.get("chunk_type", "")
        if chunk_type_val == "file" and intent == "code":
            q_lower = query.lower()
            has_function_intent = any(
                w in q_lower
                for w in ("function", "method", "which function", "which method",
                           "what function", "what method")
            )
            has_action_intent = bool(
                cls.query_tokens(query) & cls.CODE_INTENT_WORDS
            )
            if has_function_intent or has_action_intent:
                final_score *= 0.60  # penalise file chunks when asking for a function

        # ------------------------------------------------------------------
        # Markdown section penalty
        #
        # Markdown README sections should not outrank code chunks for
        # code-intent queries (e.g. "where are HOG features extracted?"
        # should not return the README "Features" section).
        # ------------------------------------------------------------------
        if chunk_type_val in ("markdown_section", "text_block", "text_file") and intent == "code":
            final_score *= 0.55

        return {
            "semantic": semantic,
            "identifier": identifier,
            "symbol": symbol,
            "action_symbol": action_symbol,
            "content": content,
            "action": action,
            "code_content": code_content,
            "file": file_score,
            "chunk_type": chunk_type,
            "intent": intent_score,
            "query_intent": intent,
            "final": final_score,
        }

    @classmethod
    def rerank(
        cls,
        query: str,
        results: List[Dict],
        top_k: int = 5,
    ):
        reranked = []

        for result in results:
            scores = cls.calculate_score(
                query,
                result,
            )

            enriched = dict(result)
            enriched["rerank_scores"] = scores
            enriched["rerank_score"] = scores["final"]

            reranked.append(enriched)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]

    @staticmethod
    def print_results(results: List[Dict]):
        print("\n========================================")
        print("CODE-AWARE RERANKED RESULTS")
        print("========================================")

        for rank, result in enumerate(results, start=1):
            scores = result["rerank_scores"]

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
                f"Final: {scores['final']:.4f}"
            )
            print(
                f"Semantic: {scores['semantic']:.4f}"
            )
            print(
                f"Identifier: {scores['identifier']:.4f}"
            )
            print(
                f"Symbol: {scores['symbol']:.4f}"
            )
            print(
                f"Action-Symbol: "
                f"{scores['action_symbol']:.4f}"
            )
            print(
                f"Content: {scores['content']:.4f}"
            )
            print(
                f"Action: {scores['action']:.4f}"
            )
            print(
                f"Intent: {scores['query_intent']}"
            )
