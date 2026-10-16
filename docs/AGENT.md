# Agent Instructions

## Mission
Implement the project described in `ARCHITECTURE.md` and the staged plan in `IMPLEMENTATION.md`. The goal is a reproducible, small-scope research prototype comparing fixed top-k retrieval-augmented generation with an evidence-aware adaptive retrieval and abstention method for cardiology question answering.

## Read first
1. `ARCHITECTURE.md`
2. `IMPLEMENTATION.md`
3. `README.md`

Treat these files as the project contract. If requirements conflict, prioritize safety, reproducibility, and the architecture's explicit scope.

## Required working style
- Inspect the repository before changing it.
- Start with a short implementation plan and list assumptions.
- Implement a working end-to-end vertical slice early; do not spend the entire schedule on infrastructure.
- Keep modules small, typed where practical, and covered by tests.
- Prefer deterministic and local functionality; make external model APIs optional.
- Never claim a feature or experiment ran unless it actually ran.
- Never invent benchmark results, citations, documents, or test output.
- Keep a `TODO` list and update documentation when interfaces change.
- If credentials, internet access, or compute are unavailable, implement a mock/extractive fallback and clearly label its limitations.

## Research integrity
- The baseline reference is *Cardiology-Chat: A Multi-LLMs Powered System for Cardiac Diagnostic Reasoning and Clinical Support* (DOI: 10.1109/JTEHM.2026.3668755).
- Do not copy the paper's prose. Describe the baseline in original wording and cite it in any resulting manuscript.
- The proposed method is a hypothesis. Do not claim novelty is established until related work is checked.
- Do not fabricate experimental results, human evaluation, clinician validation, data provenance, or statistical significance.
- Clearly separate baseline paper results from results produced by this repository.
- Preserve dataset licenses and source attribution. Do not include private or identifiable health information.
- Do not call this system a clinically validated diagnostic tool.

## Safety constraints
- This is an educational/research prototype, not a medical device or clinical decision system.
- Use public educational cardiology material only.
- Do not ask for or store real patient records.
- Do not provide personalized diagnosis or treatment. For personal symptoms, show a safety message advising contact with a qualified healthcare professional or local emergency service where appropriate.
- No automatic treatment recommendations.
- Never embed API keys in source code, notebooks, logs, docs, or commits. Use environment variables and `.env.example`.

## Required implementation
1. Source manifest and ingestion pipeline with metadata/provenance.
2. Stable chunking and vector indexing.
3. Baseline fixed top-k RAG.
4. EA-RAG query intent/adaptation, evidence sufficiency gate, one bounded retry, and abstention.
5. Citation formatting and structural validation.
6. Shared evaluation runner and machine-readable outputs.
7. Metrics for answer correctness when answer keys exist, retrieval quality where labels exist, citation validity, support rate on a manually reviewed subset, abstention/coverage, and latency.
8. Ablations isolating abstention and query adaptation/retry.
9. Minimal UI showing answer/abstention, sources, and research-only warning.
10. Unit tests and setup documentation.

## Preferred technology choices
Use Python 3.11+.
- Retrieval: `sentence-transformers` embeddings plus FAISS or Chroma; select the simpler reliable option for the environment.
- Parsing: standard libraries and lightweight PDF/HTML parsing only when needed.
- App: Streamlit.
- Evaluation/data: pandas, scikit-learn where appropriate, pytest.
- Configuration: environment variables and a typed/config object.
- Model provider: small adapter interface; support an API-backed model if configured, and a deterministic extractive fallback for no-credential runs.

Avoid installing large dependencies unless required. Pin versions in `requirements.txt` once a working environment is established. Include a CPU-compatible path.

## Engineering requirements
- Use stable random seeds where applicable.
- Make corpus paths, model names, thresholds, k, and output paths configurable.
- Save a corpus manifest and experiment config with every run.
- Keep raw outputs separate from aggregate metrics.
- Use JSONL for per-question records and CSV/JSON for aggregate metrics.
- Catch and report API/network failures without silently converting them into successful answers.
- Unit-test empty corpus, malformed records, missing citations, low-evidence abstention, retry limit, and deterministic fallback.
- Add clear errors for missing dependencies and missing configuration.
- Do not use a test set to tune prompts, thresholds, or retrieval settings.

## Suggested repository layout
```text
.
├── ARCHITECTURE.md
├── AGENT.md
├── IMPLEMENTATION.md
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── sources/          # source manifest and permitted documents
│   ├── processed/        # generated chunks; normally gitignored
│   └── evaluation/
│       ├── dev.jsonl
│       └── test.jsonl
├── artifacts/            # index, raw runs, reports; normally gitignored
├── src/
│   └── cardio_rag/
│       ├── config.py
│       ├── schemas.py
│       ├── ingest.py
│       ├── chunking.py
│       ├── index.py
│       ├── retrieval.py
│       ├── query_router.py
│       ├── evidence_gate.py
│       ├── generation.py
│       ├── citations.py
│       ├── pipeline.py
│       ├── evaluate.py
│       └── app.py
└── tests/
    ├── test_chunking.py
    ├── test_retrieval.py
    ├── test_evidence_gate.py
    ├── test_citations.py
    └── test_pipeline.py
```

## Definition of done
A clean checkout can be installed and run by following `README.md`; both retrieval variants execute on the same test set; outputs and configuration are saved; tests pass; results are reproducible to the extent allowed by model/API nondeterminism; and all limitations are explicit.

## How to report progress
At each milestone, report:
- files created/modified,
- commands run,
- tests and outcomes,
- blockers,
- actual measured results (if any),
- next concrete step.

Do not present a planned experiment as a completed result.
