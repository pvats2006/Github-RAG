import json
from pathlib import Path
from typing import Any, Dict, List

from app.retrieval.semantic_retriever import SemanticRetriever


class ContextBuilder:
    """
    Builds clean LLM-ready context from retrieved repository chunks.
    """

    def __init__(
        self,
        chunks_path: str,
        max_chunks: int = 5,
        max_chars_per_chunk: int = 12000,
    ):
        self.chunks_path = Path(chunks_path)
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk

        self.chunks = self._load_chunks()

    def _load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_path}"
            )

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                "final_chunks.json must contain a list."
            )

        return data

    def _get_content(self, chunk: Dict[str, Any]) -> str:
        content = str(
            chunk.get("content", "")
        ).strip()

        if len(content) > self.max_chars_per_chunk:
            content = (
                content[:self.max_chars_per_chunk]
                + "\n... [content truncated]"
            )

        return content

    def _get_github_reference(
        self,
        chunk: Dict[str, Any],
    ) -> str:

        metadata = chunk.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        return str(
            chunk.get("github_reference")
            or metadata.get("github_reference")
            or ""
        )

    def build_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:

        output = [
            "========================================",
            "REPOSITORY CONTEXT",
            "========================================",
            "",
        ]

        seen = set()
        context_number = 1

        for chunk in retrieved_chunks[:self.max_chunks]:

            if not isinstance(chunk, dict):
                continue

            chunk_id = str(
                chunk.get("chunk_id")
                or chunk.get("id")
                or ""
            )

            if chunk_id and chunk_id in seen:
                continue

            if chunk_id:
                seen.add(chunk_id)

            file_path = str(
                chunk.get("file_path", "unknown")
            )

            file_name = str(
                chunk.get("file_name", "unknown")
            )

            chunk_type = str(
                chunk.get("chunk_type", "unknown")
            )

            symbol = chunk.get("symbol")

            if symbol is None:
                symbol = "None"

            line_start = chunk.get("line_start")
            line_end = chunk.get("line_end")

            content = self._get_content(chunk)

            github = self._get_github_reference(chunk)

            output.append(
                f"[CONTEXT {context_number}]"
            )

            output.append(
                f"File: {file_path}"
            )

            output.append(
                f"File Name: {file_name}"
            )

            output.append(
                f"Chunk Type: {chunk_type}"
            )

            output.append(
                f"Symbol: {symbol}"
            )

            if (
                line_start is not None
                and line_end is not None
            ):
                output.append(
                    f"Lines: {line_start}-{line_end}"
                )

            output.append("")
            output.append("Code:")
            output.append("----------------------------------------")
            output.append(content)
            output.append("----------------------------------------")

            if github:
                output.append("")
                output.append(
                    f"GitHub: {github}"
                )

            output.append("")
            output.append("")

            context_number += 1

        if context_number == 1:
            output.append(
                "No relevant repository chunks found."
            )

        return "\n".join(output).strip()

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:

        context = self.build_context(
            retrieved_chunks
        )

        return f"""
You are a code repository assistant.

Answer the user's question using ONLY the
repository context below.

Rules:
1. Do not invent repository behavior.
2. Do not invent code.
3. Use the actual file paths and symbols.
4. Explain the answer clearly.
5. Mention the most relevant file and symbol.
6. If the context is insufficient, say so.
7. Prefer direct evidence from the code.

USER QUESTION:
{query}

{context}

ANSWER:
""".strip()


class RetrievalContextPipeline:
    """
    Connects:

    Query
       ↓
    SemanticRetriever
       ↓
    ContextBuilder
       ↓
    LLM-ready prompt
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        chunks_path: str,
        top_k: int = 5,
    ):

        self.chunks_path = chunks_path
        self.top_k = top_k

        self.retriever = SemanticRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            chunks_path=chunks_path,
            top_k=top_k,
        )

        self.context_builder = ContextBuilder(
            chunks_path=chunks_path,
            max_chunks=top_k,
        )

    def _extract_chunks(
        self,
        results: Any,
    ) -> List[Dict[str, Any]]:

        if results is None:
            return []

        if isinstance(results, dict):

            for key in (
                "results",
                "chunks",
                "documents",
                "retrieved_chunks",
            ):
                if key in results:
                    results = results[key]
                    break

        if not isinstance(results, list):
            return []

        chunks = []

        for result in results:

            if not isinstance(result, dict):
                continue

            # Retriever may return:
            # {"chunk": {...}, "score": 0.8}
            if isinstance(result.get("chunk"), dict):

                chunk = dict(
                    result["chunk"]
                )

                if "score" in result:
                    chunk["retrieval_score"] = (
                        result["score"]
                    )

                chunks.append(chunk)

            # Retriever may already return chunk dict
            elif (
                "file_path" in result
                or "chunk_id" in result
                or "content" in result
            ):
                chunks.append(result)

        return chunks[:self.top_k]

    def retrieve(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:

        # Support common retriever method names.
        if hasattr(self.retriever, "search"):
            results = self.retriever.search(
                query
            )

        elif hasattr(self.retriever, "retrieve"):
            retrieve_fn = getattr(self.retriever, "retrieve")
            results = retrieve_fn(
                query
            )

        elif hasattr(
            self.retriever,
            "get_relevant_chunks",
        ):
            get_chunks_fn = getattr(self.retriever, "get_relevant_chunks")
            results = get_chunks_fn(
                query
            )

        else:
            raise AttributeError(
                "SemanticRetriever does not expose "
                "retrieve(), search(), or "
                "get_relevant_chunks()."
            )

        return self._extract_chunks(results)

    def build(
        self,
        query: str,
    ):

        chunks = self.retrieve(query)

        context = (
            self.context_builder.build_context(
                chunks
            )
        )

        prompt = (
            self.context_builder.build_prompt(
                query,
                chunks,
            )
        )

        return {
            "query": query,
            "chunks": chunks,
            "context": context,
            "prompt": prompt,
        }


def main():

    print()
    print("=" * 40)
    print("RETRIEVAL → CONTEXT PIPELINE")
    print("=" * 40)

    index_path = input(
        "Enter FAISS index path: "
    ).strip()

    metadata_path = input(
        "Enter embedding metadata path: "
    ).strip()

    chunks_path = input(
        "Enter final chunks path: "
    ).strip()

    pipeline = RetrievalContextPipeline(
        index_path=index_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path,
        top_k=5,
    )

    print()
    print("=" * 40)
    print("QUERY")
    print("=" * 40)

    query = input(
        "Enter query: "
    ).strip()

    result = pipeline.build(query)

    chunks = result["chunks"]

    print()
    print("=" * 40)
    print("RETRIEVED CHUNKS")
    print("=" * 40)

    print(
        f"Retrieved: {len(chunks)}"
    )

    for i, chunk in enumerate(
        chunks,
        start=1,
    ):

        print()
        print(
            f"[{i}] "
            f"{chunk.get('file_path')} "
            f"| "
            f"{chunk.get('symbol')}"
        )

        if "retrieval_score" in chunk:
            print(
                f"Score: "
                f"{chunk['retrieval_score']:.4f}"
            )

    print()
    print("=" * 40)
    print("GENERATED CONTEXT")
    print("=" * 40)

    print(result["context"])

    print()
    print("=" * 40)
    print("LLM PROMPT")
    print("=" * 40)

    print(result["prompt"])


if __name__ == "__main__":
    main()