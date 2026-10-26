"""
RAG Pipeline orchestrator connecting Retrieval, Context Building, Groq Generation,
Citation Validation, Grounding Validation, Answer Repair, and Confidence Scoring.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from app.retrieval.context_builder import RetrievalContextPipeline
from app.llm.llm_client import GroqClient, LLMClient
from app.validation.grounded_validator import GroundedAnswerValidator
from app.validation.confidence import ConfidenceEvaluator


class RAGPipeline:

    MAX_CONTEXT_CHARS = 3500

    @staticmethod
    def _truncate_prompt_value(value: str, max_chars: int = 1800) -> str:
        if value is None:
            return ""

        text = str(value).strip()
        if len(text) <= max_chars:
            return text

        return text[: max_chars - 120].rstrip() + "\n\n... [truncated for prompt-size limits]"

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        chunks_path: str,
        top_k: int = 5,
        model: Optional[str] = None,
        llm: Optional[GroqClient] = None,
    ):
        # --------------------------------------------
        # RETRIEVAL SYSTEM
        # --------------------------------------------
        self.retrieval_context = RetrievalContextPipeline(
            index_path=index_path,
            metadata_path=metadata_path,
            chunks_path=chunks_path,
            top_k=top_k,
        )

        # --------------------------------------------
        # ANSWER GENERATION (Groq Cloud LLM)
        # --------------------------------------------
        self.llm = llm or GroqClient(
            model=model,
            temperature=0.1,
        )

        # --------------------------------------------
        # GROUNDING VALIDATOR
        # --------------------------------------------
        self.validator = GroundedAnswerValidator(
            model=model,
            llm=self.llm,
        )

        # --------------------------------------------
        # CONFIDENCE EVALUATOR
        # --------------------------------------------
        self.confidence_evaluator = ConfidenceEvaluator()

    # --------------------------------------------------
    # BUILD NUMBERED SOURCES FOR LLM
    # --------------------------------------------------

    def build_numbered_context(self, chunks: List[Dict[str, Any]]) -> str:
        numbered_context = []

        for index, chunk in enumerate(chunks, start=1):
            file_path = chunk.get("file_path", "Unknown")
            symbol = chunk.get("symbol")
            chunk_type = chunk.get("chunk_type")
            start_line = chunk.get("line_start") or chunk.get("start_line")
            end_line = chunk.get("line_end") or chunk.get("end_line")
            content = self._truncate_prompt_value(
                chunk.get("content", "").strip(),
                max_chars=1800,
            )

            source = f"[{index}]\n"
            source += f"File: {file_path}\n"

            if symbol:
                source += f"Symbol: {symbol}\n"

            if chunk_type:
                source += f"Type: {chunk_type}\n"

            if start_line is not None and end_line is not None:
                source += f"Lines: {start_line}-{end_line}\n"

            source += f"\nCode:\n{content}"
            numbered_context.append(source)

        context = "\n\n".join(numbered_context)
        if len(context) > self.MAX_CONTEXT_CHARS:
            return context[: self.MAX_CONTEXT_CHARS - 120].rstrip() + "\n\n... [truncated for prompt-size limits]"
        return context

    # --------------------------------------------------
    # CITATION-AWARE PROMPT
    # --------------------------------------------------

    def build_citation_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        context = self.build_numbered_context(chunks)

        prompt = f"""
You are a code repository assistant.

Answer the user's question using ONLY the repository context provided below.

IMPORTANT RULES:
1. Answer questions using only the supplied repository context.
2. Do not invent:
   - files
   - classes
   - functions
   - variables
   - behavior
   - dependencies
   - architecture
3. Do not use outside knowledge about this repository.
4. If the supplied context does not contain enough information, say that the repository context is insufficient.
5. Every factual repository claim should have a citation.
6. Use citations such as: [1], [2], [3].
7. Citations must use ONLY the source numbers provided.
8. Never create citations that do not exist in the supplied context.
9. Keep the answer clear, direct, and concise.

USER QUESTION:
{query}

REPOSITORY CONTEXT:
{context}

Now answer the question using ONLY the repository context and include inline citations.
"""
        return prompt.strip()

    # --------------------------------------------------
    # CITATION NUMBER VALIDATION
    # --------------------------------------------------

    def validate_citations(self, answer: str, number_of_sources: int) -> Dict[str, Any]:
        citations = re.findall(r"\[(\d+)\]", answer)
        invalid = []

        for citation in citations:
            number = int(citation)
            if number < 1 or number > number_of_sources:
                invalid.append(number)

        citations_found = sorted(set(int(c) for c in citations))
        return {
            "valid": len(invalid) == 0,
            "citations_found": citations_found,
            "invalid_citations": sorted(set(invalid)),
        }

    # --------------------------------------------------
    # BUILD SOURCE INFORMATION
    # --------------------------------------------------

    def build_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations = []
        for index, chunk in enumerate(chunks, start=1):
            file_path = chunk.get("file_path")
            if not file_path:
                continue

            metadata = chunk.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            github_url = (
                chunk.get("github_reference")
                or chunk.get("github_url")
                or metadata.get("github_reference")
            )

            citations.append({
                "number": index,
                "file": file_path,
                "symbol": chunk.get("symbol"),
                "chunk_type": chunk.get("chunk_type"),
                "start_line": chunk.get("line_start") or chunk.get("start_line"),
                "end_line": chunk.get("line_end") or chunk.get("end_line"),
                "github": github_url,
            })
        return citations

    # --------------------------------------------------
    # FORMAT SOURCES FOR DISPLAY
    # --------------------------------------------------

    def format_citations(self, citations: List[Dict[str, Any]]) -> str:
        if not citations:
            return "No sources available."

        lines = []
        for citation in citations:
            symbol_str = f" | {citation['symbol']}" if citation.get("symbol") else ""
            lines_str = (
                f" | Lines {citation['start_line']}-{citation['end_line']}"
                if citation.get("start_line") is not None and citation.get("end_line") is not None
                else ""
            )
            lines.append(f"[{citation['number']}] {citation['file']}{symbol_str}{lines_str}")
            if citation.get("github"):
                lines.append(f"    GitHub: {citation['github']}")
        return "\n".join(lines)

    # --------------------------------------------------
    # AUTOMATIC ANSWER REPAIR
    # --------------------------------------------------

    def repair_answer(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]],
        grounding_validation: Dict[str, Any],
    ) -> str:
        unsupported_claims = grounding_validation.get("unsupported_claims", [])
        context = self.build_numbered_context(chunks)
        claims_text = "\n".join(f"- {claim}" for claim in unsupported_claims)

        prompt = f"""Repair this code repository answer. Use ONLY the context below.

QUESTION: {query}

ORIGINAL ANSWER: {answer}

UNSUPPORTED CLAIMS:
{claims_text if claims_text else "- None specified"}

CONTEXT:
{context}

Rules: Keep supported claims. Remove or correct unsupported claims. Preserve valid citations [1], [2]. Do not invent facts. Return ONLY the repaired answer."""
        return self.llm.generate(prompt, max_tokens=600).strip()

    # --------------------------------------------------
    # MAIN RAG ANSWER PIPELINE
    # --------------------------------------------------

    def answer(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        t_total_start = time.perf_counter()

        # Step 1: Retrieve chunks
        t0 = time.perf_counter()
        retrieval_result = self.retrieval_context.build(query)
        chunks = retrieval_result["chunks"]
        retrieval_time = time.perf_counter() - t0

        # Step 2: Build prompt & Generate answer
        t1 = time.perf_counter()
        prompt = self.build_citation_prompt(query, chunks)
        answer = self.llm.generate(prompt)
        generation_time = time.perf_counter() - t1

        # Step 3: Validate citations
        t2 = time.perf_counter()
        t_citation_start = time.perf_counter()
        citation_validation = self.validate_citations(answer, len(chunks))
        citation_validation_time = time.perf_counter() - t_citation_start
        citations = self.build_citations(chunks)

        # Step 4: Grounding validation (fast model, no repair — saves ~35-40s)
        t_grounding_start = time.perf_counter()
        grounding_validation = self.validator.validate(
            question=query,
            answer=answer,
            chunks=chunks,
        )
        grounding_time = time.perf_counter() - t_grounding_start

        repair_attempted = False
        repair_passed = False
        repair_time = 0.0

        # Step 5: Confidence evaluation
        t_confidence_start = time.perf_counter()
        confidence = self.confidence_evaluator.evaluate(
            chunks=chunks,
            citation_validation=citation_validation,
            grounding_validation=grounding_validation,
            repair_attempted=repair_attempted,
            repair_passed=repair_passed,
        )
        confidence_time = time.perf_counter() - t_confidence_start
        validation_time = time.perf_counter() - t2
        total_time = time.perf_counter() - t_total_start

        return {
            "query": query,
            "chunks": chunks,
            "context": retrieval_result["context"],
            "answer": answer,
            "citations": citations,
            "citation_validation": citation_validation,
            "grounding_validation": grounding_validation,
            "repair_attempted": repair_attempted,
            "repair_passed": repair_passed,
            "confidence": confidence,
            "timing": {
                "retrieval": round(retrieval_time, 3),
                "generation": round(generation_time, 3),
                "validation": round(validation_time, 3),
                "total": round(total_time, 3),
                "validation_breakdown": {
                    "citation": round(citation_validation_time, 3),
                    "grounding": round(grounding_time, 3),
                    "repair": round(repair_time, 3),
                    "confidence": round(confidence_time, 3),
                },
            },
        }
