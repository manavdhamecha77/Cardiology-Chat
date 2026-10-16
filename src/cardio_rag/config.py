from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    corpus_path: Path = Path("data/processed/chunks.jsonl")
    output_dir: Path = Path("artifacts")
    top_k: int = 4
    min_score: float = 0.08
    min_passages: int = 1
    chunk_words: int = 550
    overlap_words: int = 80
    seed: int = 17

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.min_passages < 1:
            raise ValueError("top_k and min_passages must be positive")
        if self.chunk_words < 1 or not 0 <= self.overlap_words < self.chunk_words:
            raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
