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

    def validate(
        self,
        question: str,
        answer: str,
        chunks: list,
    ):

        context_parts = []

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            file_path = chunk.get(
                "file_path",
                "Unknown",
            )

            symbol = chunk.get(
                "symbol"
            )

            content = chunk.get(
                "content",
                "",
            )

            context_parts.append(
                f"""
SOURCE [{index}]
File: {file_path}
Symbol: {symbol}

Code:
{content}
"""
            )

        context = "\n".join(
            context_parts
        )

        prompt = f"""
You are a strict code-grounding validator.

Your job is to check whether the answer
is supported by the repository code provided.

USER QUESTION:
{question}

ANSWER:
{answer}

REPOSITORY SOURCES:
{context}

Rules:

1. Check every important factual claim.
2. A claim is SUPPORTED only if the provided
   repository sources actually contain evidence
   for it.
3. Do not use general programming knowledge.
4. Do not assume behavior that is not visible
   in the provided code.
5. Pay special attention to claims about:
   - algorithms
   - model selection
   - thresholds
   - databases
   - APIs
   - files
   - functions
   - cross-validation
   - training/testing
6. If a claim is not supported, mark it
   UNSUPPORTED.
7. Valid citation numbers alone do NOT mean
   that the claim is supported.

Return ONLY this format:

GROUNDING: PASS
or
GROUNDING: FAIL

UNSUPPORTED_CLAIMS:
- claim 1
- claim 2

REASON:
short explanation

If all important claims are supported,
return GROUNDING: PASS and write:
UNSUPPORTED_CLAIMS:
- None
"""

        result = self.llm.generate(
            prompt
        )

        return self.parse_result(
            result
        )

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