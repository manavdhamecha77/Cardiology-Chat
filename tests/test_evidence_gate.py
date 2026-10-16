from cardio_rag.evidence_gate import assess_evidence
from cardio_rag.schemas import Chunk, RetrievedChunk


def test_empty_evidence_is_insufficient():
    assessment = assess_evidence([], min_score=0.2)
    assert not assessment.sufficient
    assert assessment.reason


def test_retry_is_bounded_and_low_evidence_abstains():
    from cardio_rag.config import Config
    from cardio_rag.pipeline import QAPipeline
    from cardio_rag.retrieval import LexicalIndex

    chunk = Chunk("c:1", "c", "Basic", "Org", "url", "date", "license", "s", "General educational information about heart anatomy and health.")
    record = QAPipeline(LexicalIndex([chunk]), Config(min_score=0.99)).answer("What is an unrelated planetary orbit?", "ea_rag")
    assert record.retry_count == 1
    assert record.abstained
