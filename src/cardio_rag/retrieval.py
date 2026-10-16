import math
import re
from collections import Counter

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
