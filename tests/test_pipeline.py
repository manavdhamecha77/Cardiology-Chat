from cardio_rag.config import Config
from cardio_rag.pipeline import QAPipeline
from cardio_rag.retrieval import LexicalIndex
from cardio_rag.schemas import Chunk


def test_fallback_is_labeled_and_cited():
    chunk = Chunk("guide:1", "guide", "Guide", "Org", "url", "date", "license", "Basics", "Heart failure is a condition in which the heart cannot pump blood as well as it should.")
    record = QAPipeline(LexicalIndex([chunk]), Config(min_score=0.0)).answer("What is heart failure?", "baseline")
    assert "Extractive demo response" in record.answer
    assert record.citations == ["guide:1"]
    assert record.citation_valid


def test_adaptive_pipeline_retries_once():
    chunk = Chunk("guide:1", "guide", "Guide", "Org", "url", "date", "license", "Basics", "Heart failure is a condition in which the heart cannot pump blood as well as it should.")
    record = QAPipeline(LexicalIndex([chunk]), Config(min_score=0.99)).answer("What is heart failure?", "ea_rag")
    assert record.retry_count == 1
