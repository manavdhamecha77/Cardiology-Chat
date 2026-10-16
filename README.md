# Evidence-Aware Adaptive RAG for Cardiology QA

A local research prototype comparing fixed top-k retrieval with evidence-gated retrieval, one bounded retry, and abstention for cardiology educational questions.

## Current implementation

- Source manifest validation and ingestion from local UTF-8 text or Markdown files.
- Stable heading-aware chunks with source metadata and JSONL persistence.
- Deterministic local BM25-style lexical retrieval. Sentence-transformer embeddings and a persistent vector database are not implemented yet.
- Fixed top-k baseline and EA-RAG with keyword intent routing, query expansion, a configurable evidence threshold, one retry, and abstention.
- A deterministic extractive generator, structural citation validation, JSONL evaluation records, aggregate JSON metrics, and run manifests.
- Ablations without abstention and without adaptation/retry, plus a minimal Streamlit interface.

No source corpus or benchmark questions are bundled. Add permitted public educational source files and document each one's provenance and usage terms in `data/sources/manifest.json`. Do not use patient records. Do not report experiment results until you have supplied a corpus/questions and run the evaluation.

## Setup (PowerShell)

Install the pinned dependencies manually into the existing root virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

The package uses only the Python standard library for ingestion, chunking, retrieval, and evaluation. `streamlit` runs the demo and `pytest` is available for the included unit tests.

## Add sources and build the corpus

Create one local `.txt` or `.md` file per public source. The manifest is a JSON array with required provenance fields; paths are relative to the manifest file:

```json
[
  {
    "document_id": "organization-topic-2026",
    "title": "Exact title of the source",
    "publisher": "Publishing organization",
    "url": "https://publisher.example/source",
    "published_date": "YYYY-MM-DD",
    "accessed_date": "YYYY-MM-DD",
    "license_note": "License or terms permitting this use",
    "path": "organization-topic.md"
  }
]
```

Record the exact source and its version/date in your manifest and local file. The tool does not fetch URLs or verify licenses. Then run:

```powershell
python -m cardio_rag.cli ingest
```

Chunk defaults are 550 whitespace-delimited words with 80 words of overlap. The generated `data/processed/chunks.jsonl` is ignored by Git; preserve your source files and manifest to reproduce it.

## Ask and evaluate

```powershell
python -m cardio_rag.cli ask "What is the definition of atrial fibrillation?" --variant ea_rag
python -m cardio_rag.cli evaluate --questions data/evaluation/dev.jsonl --run-id pilot
```

Question JSONL records require a non-empty `question`; optional fields include `question_id`, `reference_answer`, and `relevant_chunk_ids`. The evaluator runs baseline, EA-RAG, no-abstention, and no-adaptation variants on the same records. It writes per-question `artifacts/pilot.jsonl`, aggregate `artifacts/pilot_metrics.json`, and configuration `artifacts/pilot_manifest.json`. It calculates citation structure validity, Hit@k when relevance labels exist, a simple reference-answer substring rate when answers are provided, abstention, and latency. Substring match is not a clinical correctness metric. Unsupported-answer and semantic citation support rates require human review and are not inferred by this prototype.

Run the interface after building a corpus:

```powershell
python -m streamlit run src/cardio_rag/app.py
```

## Safety and research limits

This is an educational research prototype, not a medical device, diagnostic service, or substitute for professional care. The UI warns users not to enter personal health information and the pipeline returns a safety notice for recognized first-person symptom queries. The notice uses simple keyword matching and is not a triage system. Never use outputs for clinical decisions.

The retrieval score is a normalized lexical ranking signal, not calibrated confidence. Query intent and evidence sufficiency use transparent heuristics. The extractive fallback copies source sentences; citations are checked structurally and are not proof of semantic support. There is no language-model provider, embedding model, PDF/HTML parser, automatic contradiction detector, reviewed benchmark, or measured research result yet.

## Research reference

Yang, Z. et al. “Cardiology-Chat: A Multi-LLMs Powered System for Cardiac Diagnostic Reasoning and Clinical Support.” *IEEE Journal of Translational Engineering in Health and Medicine*, 2026. DOI: [10.1109/JTEHM.2026.3668755](https://doi.org/10.1109/JTEHM.2026.3668755). This repository does not reproduce its training or evaluation and does not claim established novelty or clinical performance.
