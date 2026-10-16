from cardio_rag.retrieval import LexicalIndex
from cardio_rag.schemas import Chunk


def test_empty_index_and_relevant_rank():
    assert LexicalIndex([]).retrieve("heart") == []
    generic = Chunk("a:1", "a", "Generic", "Org", "url", "date", "license", "s", "general public education material")
    target = Chunk("b:1", "b", "Target", "Org", "url", "date", "license", "s", "Atrial fibrillation is an irregular heart rhythm")
    assert LexicalIndex([generic, target]).retrieve("atrial fibrillation") [0].chunk.chunk_id == "b:1"
