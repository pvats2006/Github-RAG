import re
from typing import Optional

from app.llm.llm_client import GroqClient, LLMClient


class GroundedAnswerValidator:

    def __init__(
        self,
        model: Optional[str] = None,
        llm: Optional[GroqClient] = None,
    ):
        self.llm = llm or GroqClient(
            model=model,
            temperature=0.0,
        )

    @staticmethod
    def _is_insufficient_context_answer(answer: str) -> bool:
        if not answer:
            return False

        normalized = answer.strip().lower()
        if not normalized:
            return False

        insufficient_phrases = (
            "insufficient context",
            "not enough information",
            "insufficient repository context",
            "context is insufficient",
            "not enough context",
            "cannot determine",
            "can't determine",
            "unable to determine",
            "needs more context",
            "need more context",
            "not enough evidence",
            "cannot answer from the provided context",
            "do not have enough information",
            "not enough repository context",
        )
        return any(phrase in normalized for phrase in insufficient_phrases)

    @staticmethod
    def _link_citation_numbers(answer: str):
        return sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer)})

    @staticmethod
    def _select_relevant_chunks(answer: str, chunks: list):
        if not chunks:
            return []

        cited_numbers = GroundedAnswerValidator._link_citation_numbers(answer)
        if cited_numbers:
            relevant = []
            for index in cited_numbers:
                if 1 <= index <= len(chunks):
                    relevant.append(chunks[index - 1])
            if relevant:
                return relevant

        tokens = re.findall(r"[a-zA-Z0-9_./-]+", answer.lower())
        if not tokens:
            return chunks[:2]

        scored = []
        for index, chunk in enumerate(chunks):
            text = " ".join([
                str(chunk.get("file_path", "")),
                str(chunk.get("symbol", "")),
                str(chunk.get("content", "")),
            ]).lower()
            score = sum(1 for token in tokens if token in text)
            if score > 0:
                scored.append((score, index))

        if not scored:
            return chunks[:2]

        scored.sort(key=lambda item: item[0], reverse=True)
        selected_indexes = [index for _, index in scored[: min(2, len(scored))]]
        return [chunks[index] for index in selected_indexes]

    @staticmethod
    def _truncate_content(content: str, max_chars: int = 400) -> str:
        """Truncate chunk content to reduce LLM prompt size."""
        if not content or len(content) <= max_chars:
            return content
        return content[:max_chars].rstrip() + "\n... [truncated]"

    @staticmethod
    def _build_context(chunks: list) -> str:
        context_parts = []
        for index, chunk in enumerate(chunks, start=1):
            file_path = chunk.get("file_path", "Unknown")
            symbol = chunk.get("symbol")
            content = GroundedAnswerValidator._truncate_content(
                chunk.get("content", "")
            )
            header = f"[{index}] {file_path}"
            if symbol:
                header += f" | {symbol}"
            context_parts.append(f"{header}\n{content}")
        return "\n\n".join(context_parts)

    def validate(
        self,
        question: str,
        answer: str,
        chunks: list,
    ):
        if not answer or not answer.strip():
            return {
                "grounding": "FAIL",
                "passed": False,
                "unsupported_claims": ["Empty answer"],
                "reason": "The answer is empty.",
                "raw": answer,
                "fast_path": False,
            }

        if self._is_insufficient_context_answer(answer):
            return {
                "grounding": "PASS",
                "passed": True,
                "unsupported_claims": [],
                "reason": "Answer explicitly reports insufficient repository context and does not assert unsupported repository facts.",
                "raw": answer,
                "fast_path": True,
            }

        relevant_chunks = self._select_relevant_chunks(answer, chunks)
        context = self._build_context(relevant_chunks)

        prompt = f"""You are a strict code-grounding validator. Check if the answer is supported by the sources.

QUESTION: {question}

ANSWER: {answer}

SOURCES:
{context}

Rules: A claim is SUPPORTED only if the sources contain evidence. Do not use outside knowledge. Citation numbers alone do not prove support.

Return ONLY:
GROUNDING: PASS or FAIL
UNSUPPORTED_CLAIMS:
- claim (or "None")
REASON: short explanation
"""

        try:
            result = self.llm.generate(prompt, max_tokens=200)
        except TypeError:
            result = self.llm.generate(prompt)
        parsed = self.parse_result(result)
        parsed["fast_path"] = False
        return parsed

    def parse_result(
        self,
        result: str,
    ):

        result_upper = result.upper()

        grounding_match = re.search(
            r"GROUNDING:\s*(PASS|FAIL)",
            result_upper,
        )

        if grounding_match:
            grounding = (
                grounding_match.group(1)
            )
        else:
            grounding = "FAIL"

        unsupported = []

        match = re.search(
            r"UNSUPPORTED_CLAIMS:\s*(.*?)(?:\nREASON:|\Z)",
            result,
            re.IGNORECASE | re.DOTALL,
        )

        if match:

            block = match.group(1).strip()

            for line in block.splitlines():

                line = line.strip()

                if line.startswith("-"):
                    claim = line[1:].strip()

                    if (
                        claim
                        and claim.lower()
                        != "none"
                    ):
                        unsupported.append(
                            claim
                        )

        reason_match = re.search(
            r"REASON:\s*(.*)",
            result,
            re.IGNORECASE | re.DOTALL,
        )

        reason = ""

        if reason_match:
            reason = (
                reason_match.group(1)
                .strip()
            )

        return {
            "grounding": grounding,
            "passed": grounding == "PASS",
            "unsupported_claims": unsupported,
            "reason": reason,
            "raw": result,
        }