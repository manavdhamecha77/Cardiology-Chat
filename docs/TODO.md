# Remaining Work

- Add 5–10 permitted public cardiology sources with verified provenance, dates, and usage terms; do not redistribute material without permission.
- Build and freeze provenance-backed development and test questions with answer notes and relevant chunk labels where supportable.
- Install and run the pinned dense-retrieval dependency; record the exact embedding model revision and verify cache reuse on the local corpus.
- Tune lexical and dense evidence thresholds using development data only, then freeze them before test evaluation.
- Run baseline, EA-RAG, and ablations with one fixed generator and configuration; preserve raw JSONL and run manifests.
- Manually review a documented sample for evidence support, citation entailment, unsupported answers, and unnecessary abstentions.
- Add contradiction handling and PDF/HTML parsing only when selected source formats require them.
- Report measured results and limitations; no benchmark or model result is established yet.
