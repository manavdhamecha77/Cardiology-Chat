import json
import hashlib
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pipeline import QAPipeline


def load_questions(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed question at {path}:{line_number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("question"), str) or not record["question"].strip():
                raise ValueError(f"Question at {path}:{line_number} must have non-empty question text")
            records.append(record)
    return records


def evaluate(pipeline: QAPipeline, questions: list[dict[str, Any]], output_dir: Path, run_id: str = "local") -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    for question in questions:
        for variant in ("baseline", "ea_rag", "ea_rag_no_abstention", "ea_rag_no_adaptation"):
            records.append(pipeline.answer(question["question"], variant, question.get("question_id", ""), question.get("reference_answer"), question.get("relevant_chunk_ids", [])).to_dict())
    raw_path = output_dir / f"{run_id}.jsonl"
    raw_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    metrics = summarize(records, (time.perf_counter() - started) * 1000)
    (output_dir / f"{run_id}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config = pipeline.config
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(config.corpus_path),
        "corpus_sha256": hashlib.sha256(config.corpus_path.read_bytes()).hexdigest() if config.corpus_path.exists() else None,
        "source_manifest": "data/sources/manifest.json",
        "source_manifest_sha256": hashlib.sha256(Path("data/sources/manifest.json").read_bytes()).hexdigest() if Path("data/sources/manifest.json").exists() else None,
        "questions_sha256": hashlib.sha256(json.dumps(questions, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "questions_count": len(questions),
        "systems": ["baseline", "ea_rag", "ea_rag_no_abstention", "ea_rag_no_adaptation"],
        "generator": pipeline.generator.provider if pipeline.generator else "configuration_error",
        "model": pipeline.generator.model if pipeline.generator else None,
        "retrieval": "BM25-style lexical ranking",
        "config": {"top_k": config.top_k, "min_score": config.min_score, "min_passages": config.min_passages, "chunk_words": config.chunk_words, "overlap_words": config.overlap_words, "seed": config.seed},
    }
    (output_dir / f"{run_id}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics


def summarize(records: list[dict[str, Any]], total_latency_ms: float = 0.0) -> dict[str, Any]:
    metrics: dict[str, Any] = {"record_count": len(records), "total_latency_ms": total_latency_ms, "systems": {}}
    for variant in ("baseline", "ea_rag", "ea_rag_no_abstention", "ea_rag_no_adaptation"):
        rows = [row for row in records if row["variant"] == variant]
        citation_rows = [row for row in rows if not row["abstained"]]
        labeled = [row for row in rows if row["relevant_chunk_ids"]]
        correct = [row for row in rows if row["reference_answer"] is not None]
        metrics["systems"][variant] = {
            "count": len(rows),
            "abstention_rate": sum(row["abstained"] for row in rows) / len(rows) if rows else None,
            "citation_validity": sum(row["citation_valid"] for row in citation_rows) / len(citation_rows) if citation_rows else None,
            "citation_validity_denominator": len(citation_rows),
            "hit_at_k": sum(bool(set(row["retrieved_chunk_ids"]) & set(row["relevant_chunk_ids"])) for row in labeled) / len(labeled) if labeled else None,
            "hit_at_k_denominator": len(labeled),
            "reference_substring_accuracy": sum(row["reference_answer"].casefold() in row["answer"].casefold() for row in correct) / len(correct) if correct else None,
            "reference_answer_denominator": len(correct),
            "mean_latency_ms": statistics.mean(row["latency_ms"] for row in rows) if rows else None,
            "p50_latency_ms": statistics.median(row["latency_ms"] for row in rows) if rows else None,
            "retry_rate": sum(row["retry_count"] > 0 for row in rows) / len(rows) if rows else None,
            "unsupported_answer_rate": None,
            "evidence_support_rate": None,
        }
    return metrics
