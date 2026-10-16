from cardio_rag.citations import validate_citations
from cardio_rag.schemas import Chunk, RetrievedChunk


def test_missing_or_unretrieved_citation_fails():
    chunk = Chunk("a:1", "a", "Guide", "Org", "url", "date", "license", "s", "A sufficiently long educational passage about heart health.")
    retrieved = [RetrievedChunk(chunk, 0.5)]
    assert not validate_citations("A claim without a citation.", retrieved)[0]
    assert not validate_citations("A claim [b:9].", retrieved)[0]
    assert validate_citations("A claim supported here [a:1].", retrieved)[0]
