import argparse
import json
import os
from pathlib import Path

from .config import Config
from .evaluate import evaluate, load_questions
from .ingest import ingest, read_chunks
from .pipeline import QAPipeline
from .retrieval import create_index


def main() -> None:
    parser = argparse.ArgumentParser(prog="cardio-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="chunk local public educational sources")
    ingest_parser.add_argument("--manifest", type=Path, default=Path("data/sources/manifest.json"))
    ingest_parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    ask_parser = commands.add_parser("ask", help="answer one educational question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--variant", choices=("baseline", "ea_rag", "ea_rag_no_abstention", "ea_rag_no_adaptation"), default="ea_rag")
    ask_parser.add_argument("--corpus", type=Path, default=Path("data/processed/chunks.jsonl"))
    ask_parser.add_argument("--retrieval", choices=("lexical", "dense"), default=os.getenv("CARDIO_RAG_RETRIEVAL", "lexical"))
    ask_parser.add_argument("--embedding-model", default=os.getenv("CARDIO_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ask_parser.add_argument("--embedding-revision", default=os.getenv("CARDIO_RAG_EMBEDDING_REVISION"))
    eval_parser = commands.add_parser("evaluate", help="run baseline and EA-RAG on the same JSONL questions")
    eval_parser.add_argument("--questions", type=Path, required=True)
    eval_parser.add_argument("--corpus", type=Path, default=Path("data/processed/chunks.jsonl"))
    eval_parser.add_argument("--output", type=Path, default=Path("artifacts"))
    eval_parser.add_argument("--run-id", default="local")
    eval_parser.add_argument("--retrieval", choices=("lexical", "dense"), default=os.getenv("CARDIO_RAG_RETRIEVAL", "lexical"))
    eval_parser.add_argument("--embedding-model", default=os.getenv("CARDIO_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    eval_parser.add_argument("--embedding-revision", default=os.getenv("CARDIO_RAG_EMBEDDING_REVISION"))
    args = parser.parse_args()
    if args.command == "ingest":
        print(json.dumps({"chunks_written": ingest(args.manifest, args.output)}))
        return
    corpus = read_chunks(args.corpus)
    config = Config(
        corpus_path=args.corpus,
        output_dir=getattr(args, "output", Path("artifacts")),
        retrieval_backend=args.retrieval,
        embedding_model=args.embedding_model,
        embedding_revision=args.embedding_revision,
    )
    pipeline = QAPipeline(create_index(corpus, config), config)
    if args.command == "ask":
        print(json.dumps(pipeline.answer(args.question, args.variant).to_dict(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(evaluate(pipeline, load_questions(args.questions), args.output, args.run_id), indent=2))


if __name__ == "__main__":
    main()
