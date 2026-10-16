import math
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

from .schemas import Chunk, RetrievedChunk

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class LexicalIndex:
    """Deterministic BM25-style local index; embeddings can replace this later."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self.lengths = [len(item) for item in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if chunks else 0.0
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if not self.chunks or k < 1:
            return []
        terms = set(tokenize(query))
        if not terms:
            return []
        scored: list[RetrievedChunk] = []
        total = len(self.chunks)
        for chunk, tokens, length in zip(self.chunks, self.tokens, self.lengths):
            counts = Counter(tokens)
            score = 0.0
            for term in terms:
                frequency = counts[term]
                if not frequency:
                    continue
                inverse_frequency = math.log(1 + (total - self.document_frequency[term] + 0.5) / (self.document_frequency[term] + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / max(self.average_length, 1))
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator
            normalized = score / (score + 2.0) if score > 0 else 0.0
            scored.append(RetrievedChunk(chunk, normalized))
        return sorted(scored, key=lambda result: (-result.score, result.chunk.chunk_id))[:k]


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> object: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, revision: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Dense retrieval requires sentence-transformers; install it from requirements.txt.") from exc
        try:
            self.model = SentenceTransformer(model_name, device="cpu", revision=revision)
        except Exception as exc:
            raise RuntimeError(f"Could not load embedding model '{model_name}': {exc}") from exc

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class DenseIndex:
    """Exact cosine retrieval over normalized sentence-transformer vectors."""

    def __init__(
        self,
        chunks: list[Chunk],
        encoder: TextEncoder,
        model_name: str,
        revision: str | None = None,
        cache_dir: Path | None = None,
    ):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Dense retrieval requires numpy; install it from requirements.txt.") from exc
        self.chunks = chunks
        self.encoder = encoder
        self.model_name = model_name
        self.revision = revision
        identity = json.dumps(
            {"model": model_name, "revision": revision, "chunks": [(item.chunk_id, item.text) for item in chunks]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.cache_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"dense-{self.cache_key}.npz" if cache_dir else None
        vectors = None
        if cache_path and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    if str(cached["cache_key"].item()) == self.cache_key:
                        vectors = cached["embeddings"]
            except (OSError, ValueError, KeyError):
                vectors = None
            if vectors is not None and (vectors.ndim != 2 or vectors.shape[0] != len(chunks)):
                vectors = None
        if vectors is None:
            if not chunks:
                vectors = np.empty((0, 0), dtype=np.float32)
            else:
                vectors = np.asarray(encoder.encode([item.text for item in chunks]), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
                raise ValueError("Embedding model returned an unexpected vector shape")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, cache_key=self.cache_key, embeddings=vectors)
        self.embeddings = vectors

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if not self.chunks or k < 1 or not query.strip():
            return []
        import numpy as np

        query_vector = np.asarray(self.encoder.encode([query]), dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[0] != 1 or query_vector.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Embedding model returned an incompatible query vector")
        query_vector = query_vector[0] / max(float(np.linalg.norm(query_vector[0])), 1e-12)
        scores = self.embeddings @ query_vector
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))[:k]
        return [RetrievedChunk(self.chunks[i], max(-1.0, min(1.0, float(scores[i])))) for i in ranked]


def create_index(chunks: list[Chunk], config) -> LexicalIndex | DenseIndex:
    if config.retrieval_backend == "lexical":
        return LexicalIndex(chunks)
    encoder = SentenceTransformerEncoder(config.embedding_model, config.embedding_revision)
    return DenseIndex(chunks, encoder, config.embedding_model, config.embedding_revision, config.embedding_cache_dir)
