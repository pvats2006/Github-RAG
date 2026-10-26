from typing import List, Dict, Any
import math


class ConfidenceEvaluator:
    """
    Evidence-based heuristic confidence.

    This is an estimate, NOT a calibrated probability.
    """

    def __init__(
        self,
        high_threshold: float = 0.75,
        medium_threshold: float = 0.50,
        minimum_retrieval_confidence: float = 0.30,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.minimum_retrieval_confidence = minimum_retrieval_confidence

    def get_similarity_scores(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[float]:

        scores = []

        for chunk in chunks:
            score = chunk.get(
                "score",
                chunk.get("similarity", 0.0),
            )

            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                scores.append(0.0)

        return scores

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def calibrate_similarity(self, similarity: float) -> float:
        # Raw FAISS similarity is not a probability.
        return self._sigmoid(
            (similarity - 0.30) / 0.08
        )

    def calculate_retrieval_confidence(
        self,
        chunks: List[Dict[str, Any]],
    ) -> float:

        if not chunks:
            return 0.0

        scores = self.get_similarity_scores(chunks)

        if not scores:
            return 0.0

        top = self.calibrate_similarity(scores[0])

        supporting = sum(
            self.calibrate_similarity(score)
            for score in scores[:5]
        ) / min(len(scores), 5)

        if len(scores) > 1:
            margin = max(0.0, scores[0] - scores[1])
        else:
            margin = 0.0

        margin_signal = self._sigmoid(
            (margin - 0.05) / 0.04
        )

        return max(
            0.0,
            min(
                0.55 * top
                + 0.30 * supporting
                + 0.15 * margin_signal,
                1.0,
            ),
        )

    def evaluate(
        self,
        chunks: List[Dict[str, Any]],
        citation_validation: Dict[str, Any],
        grounding_validation: Dict[str, Any],
        repair_attempted: bool = False,
        repair_passed: bool = False,
    ) -> Dict[str, Any]:

        retrieval = self.calculate_retrieval_confidence(chunks)

        citation_passed = bool(
            citation_validation.get("valid", False)
        )

        grounding_passed = bool(
            grounding_validation.get("passed", False)
        )

        # Evidence quality gets more weight than raw similarity.
        score = (
            0.35 * retrieval
            + 0.20 * float(citation_passed)
            + 0.35 * float(grounding_passed)
            + 0.10 * float(repair_passed)
        )

        if repair_attempted and not repair_passed:
            score -= 0.15

        score = max(0.0, min(score, 1.0))

        context_sufficient = (
            bool(chunks)
            and retrieval >= self.minimum_retrieval_confidence
            and citation_passed
            and grounding_passed
        )

        if not context_sufficient:
            level = "LOW"
        elif score >= self.high_threshold:
            level = "HIGH"
        elif score >= self.medium_threshold:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "retrieval_confidence": round(retrieval, 4),
            "final_confidence": round(score, 4),
            "confidence_percent": round(score * 100, 2),
            "level": level,
            "context_sufficient": context_sufficient,
            "citation_passed": citation_passed,
            "grounding_passed": grounding_passed,
            "repair_attempted": repair_attempted,
            "repair_passed": repair_passed,
        }
