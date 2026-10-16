from pathlib import Path

import streamlit as st

from cardio_rag.config import Config
from cardio_rag.ingest import read_chunks
from cardio_rag.pipeline import QAPipeline, RESEARCH_NOTICE
from cardio_rag.retrieval import LexicalIndex


@st.cache_resource
def get_pipeline(corpus_path: str) -> QAPipeline:
    return QAPipeline(LexicalIndex(read_chunks(Path(corpus_path))), Config(corpus_path=Path(corpus_path)))


def main() -> None:
    st.set_page_config(page_title="Cardiology Evidence QA", page_icon="♥")
    st.title("Cardiology Evidence QA")
    st.warning(RESEARCH_NOTICE + " Do not enter personal health information.")
    corpus_path = st.text_input("Local chunk corpus", "data/processed/chunks.jsonl")
    question = st.text_area("Educational question")
    variant = st.selectbox("Pipeline", ["ea_rag", "baseline"])
    if st.button("Ask"):
        try:
            record = get_pipeline(corpus_path).answer(question, variant)
            st.write(record.answer)
            st.caption(f"Abstained: {record.abstained} · Intent: {record.intent} · Retries: {record.retry_count} · Citation structure valid: {record.citation_valid}")
            chunks_by_id = {chunk.chunk_id: chunk for chunk in get_pipeline(corpus_path).index.chunks}
            scores_by_id = dict(zip(record.retrieved_chunk_ids, record.retrieval_scores))
            for chunk_id in record.retrieved_chunk_ids:
                chunk = chunks_by_id[chunk_id]
                with st.expander(f"{chunk.title} · {chunk.section} · {scores_by_id[chunk_id]:.3f}"):
                    st.write(chunk.text)
                    st.caption(f"{chunk.chunk_id} · {chunk.publisher} · {chunk.accessed_date} · {chunk.url}")
        except ValueError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
