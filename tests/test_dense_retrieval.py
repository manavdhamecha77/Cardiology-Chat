import numpy as np

from cardio_rag.retrieval import DenseIndex
from cardio_rag.schemas import Chunk


class TinyEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0, 0.0] if "atrial" in lowered else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_dense_index_ranks_by_cosine_and_caches(tmp_path):
    chunks = [
        Chunk("a:1", "a", "Atrial", "Org", "url", "date", "license", "s", "Atrial fibrillation is an irregular rhythm."),
        Chunk("b:1", "b", "General", "Org", "url", "date", "license", "s", "General heart health information."),
    ]
    index = DenseIndex(chunks, TinyEncoder(), "test-encoder", cache_dir=tmp_path)
    assert index.retrieve("atrial fibrillation", 1)[0].chunk.chunk_id == "a:1"
    assert index.cache_key
    assert list(tmp_path.glob("dense-*.npz"))


def test_dense_index_handles_empty_corpus():
    index = DenseIndex([], TinyEncoder(), "test-encoder")
    assert index.retrieve("anything") == []
