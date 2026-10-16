from dataclasses import dataclass

from .schemas import RetrievedChunk


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    signals: dict[str, float | int]
    reason: str | None = None


def assess_evidence(results: list[RetrievedChunk], min_score: float = 0.08, min_passages: int = 1) -> EvidenceAssessment:
    usable = [item for item in results if item.score >= min_score]
    top_score = max((item.score for item in results), default=0.0)
    sufficient = len(usable) >= min_passages
    reason = None if sufficient else "Retrieved passages did not meet the configured evidence threshold."
    return EvidenceAssessment(sufficient, {"top_score": top_score, "usable_passages": len(usable), "required_passages": min_passages}, reason)
