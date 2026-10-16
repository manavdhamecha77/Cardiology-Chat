import time
import re

from .citations import validate_citations
from .config import Config
from .evidence_gate import assess_evidence
from .generation import GenerationError, generator_from_environment
from .query_router import adapt_query, classify_intent
from .retrieval import DenseIndex, LexicalIndex
from .schemas import RunRecord

RESEARCH_NOTICE = "Research prototype for educational use only; not medical advice or a clinical decision system."
PERSONAL_SYMPTOM_NOTICE = "If these are your symptoms, contact a qualified healthcare professional; for urgent or severe symptoms, contact local emergency services. This prototype cannot assess personal symptoms."


class QAPipeline:
    def __init__(self, index: LexicalIndex | DenseIndex, config: Config | None = None):
        self.index = index
        self.config = config or Config()
        try:
            self.generator = generator_from_environment()
            self.generator_error: str | None = None
        except GenerationError as exc:
            self.generator = None
            self.generator_error = str(exc)

    def answer(self, question: str, variant: str = "baseline", question_id: str = "adhoc", reference_answer: str | None = None, relevant_chunk_ids: list[str] | None = None) -> RunRecord:
        started = time.perf_counter()
        question = question.strip()
        intent = classify_intent(question)
        if not question:
            return RunRecord(question_id, question, variant, "Enter an educational question.", True, intent, abstention_reason="Empty question", latency_ms=(time.perf_counter() - started) * 1000)
        if re.search(r"\b(i|i'm|i am|my|we|my child)\b.*\b(chest pain|shortness of breath|palpitation|faint|dizz|heart attack|symptom)\b", question.lower()):
            return RunRecord(question_id, question, variant, PERSONAL_SYMPTOM_NOTICE, True, intent, abstention_reason="Personal symptom safety response", latency_ms=(time.perf_counter() - started) * 1000)

        retrieved = self.index.retrieve(question, self.config.top_k)
        retry_count = 0
        assessment = assess_evidence(retrieved, self.config.evidence_min_score, self.config.min_passages)
        if variant in {"ea_rag", "ea_rag_no_abstention"} and not assessment.sufficient:
            retry_count = 1
            retry_query = adapt_query(question, intent)
            retry = self.index.retrieve(retry_query, self.config.top_k)
            merged: dict[str, object] = {item.chunk.chunk_id: item for item in retrieved}
            for item in retry:
                if item.chunk.chunk_id not in merged or item.score > merged[item.chunk.chunk_id].score:
                    merged[item.chunk.chunk_id] = item
            retrieved = sorted(merged.values(), key=lambda item: (-item.score, item.chunk.chunk_id))[:self.config.top_k]
            assessment = assess_evidence(retrieved, self.config.evidence_min_score, self.config.min_passages)

        abstain = variant in {"ea_rag", "ea_rag_no_adaptation"} and not assessment.sufficient
        abstention_reason = assessment.reason if abstain else None
        if abstain:
            answer = "The available indexed sources are insufficient to answer this question reliably. " + RESEARCH_NOTICE
            citations: list[str] = []
            citation_valid = False
        else:
            if variant in {"baseline", "ea_rag_no_abstention"}:
                generation_results = retrieved
            else:
                generation_results = [item for item in retrieved if item.score >= self.config.evidence_min_score]
            if self.generator_error:
                return RunRecord(
                    question_id=question_id, question=question, variant=variant,
                    answer=f"Answer generation failed: {self.generator_error}", abstained=True, intent=intent,
                    retrieved_chunk_ids=[item.chunk.chunk_id for item in retrieved],
                    retrieval_scores=[item.score for item in retrieved], retry_count=retry_count,
                    abstention_reason=self.generator_error,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    reference_answer=reference_answer, relevant_chunk_ids=relevant_chunk_ids or [],
                    metadata={"generator_error": self.generator_error},
                )
            try:
                generated = self.generator.generate(question, generation_results)
            except GenerationError as exc:
                return RunRecord(
                    question_id=question_id, question=question, variant=variant,
                    answer=f"Answer generation failed: {exc}", abstained=True, intent=intent,
                    retrieved_chunk_ids=[item.chunk.chunk_id for item in retrieved],
                    retrieval_scores=[item.score for item in retrieved], retry_count=retry_count,
                    abstention_reason=str(exc), latency_ms=(time.perf_counter() - started) * 1000,
                    reference_answer=reference_answer, relevant_chunk_ids=relevant_chunk_ids or [],
                    metadata={"generator_error": str(exc), "provider": self.generator.provider, "model": self.generator.model},
                )
            answer = generated.answer
            citation_valid, citations = validate_citations(answer, generation_results)
            if not citations and variant in {"ea_rag", "ea_rag_no_adaptation"}:
                abstain = True
                abstention_reason = "No citable evidence passages were available."
                answer = "The available indexed sources are insufficient to answer this question reliably. " + RESEARCH_NOTICE

        return RunRecord(
            question_id=question_id, question=question, variant=variant, answer=answer, abstained=abstain,
            intent=intent, retrieved_chunk_ids=[item.chunk.chunk_id for item in retrieved],
            retrieval_scores=[item.score for item in retrieved], citations=citations, citation_valid=citation_valid,
            retry_count=retry_count, abstention_reason=abstention_reason,
            latency_ms=(time.perf_counter() - started) * 1000, reference_answer=reference_answer,
            relevant_chunk_ids=relevant_chunk_ids or [], metadata={
                "evidence": assessment.signals,
                "generator": self.generator.provider if self.generator else "configuration_error",
                "model": self.generator.model if self.generator else None,
            },
        )
