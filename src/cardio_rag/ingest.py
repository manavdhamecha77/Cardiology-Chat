import json
from pathlib import Path

from .chunking import chunk_text
from .schemas import Chunk

REQUIRED_FIELDS = {"document_id", "title", "publisher", "url", "accessed_date", "license_note", "path"}


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read source manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("source manifest must contain a JSON list")
    seen: set[str] = set()
    base = manifest_path.parent
    documents = []
    for position, record in enumerate(raw, start=1):
        if not isinstance(record, dict) or REQUIRED_FIELDS - record.keys():
            raise ValueError(f"manifest record {position} is missing required fields")
        if not all(isinstance(record[key], str) and record[key].strip() for key in REQUIRED_FIELDS):
            raise ValueError(f"manifest record {position} has empty or non-text fields")
        if record["document_id"] in seen:
            raise ValueError(f"duplicate document_id: {record['document_id']}")
        seen.add(record["document_id"])
        documents.append({**record, "_path": str((base / record["path"]).resolve())})
    return documents


def ingest(manifest_path: Path, output_path: Path, chunk_words: int = 550, overlap_words: int = 80) -> int:
    manifest = load_manifest(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as target:
        for record in manifest:
            source_path = Path(record.pop("_path"))
            try:
                text = source_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Cannot read source {source_path}: {exc}") from exc
            metadata = {key: record[key] for key in REQUIRED_FIELDS - {"path"}}
            metadata["published_date"] = record.get("published_date")
            chunks = chunk_text(text, metadata, chunk_words, overlap_words)
            for chunk in chunks:
                target.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")
                count += 1
    return count


def read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    chunks.append(Chunk(**record))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"Malformed chunk record at {path}:{line_number}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read corpus {path}: {exc}") from exc
    return chunks
