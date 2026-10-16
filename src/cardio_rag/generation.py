from .schemas import RetrievedChunk

FALLBACK_LABEL = "Extractive demo response (no language model configured)."


def extractive_answer(results: list[RetrievedChunk], max_sentences: int = 3) -> tuple[str, list[str]]:
    if not results:
        return "The available sources do not provide enough evidence to answer this question.", []
    sentences: list[tuple[float, str, str]] = []
    for result in results:
        for sentence in result.chunk.text.replace("\n", " ").split(". "):
            cleaned = sentence.strip().rstrip(".")
            if len(cleaned.split()) >= 5:
                sentences.append((result.score, cleaned, result.chunk.chunk_id))
    selected = sorted(sentences, key=lambda item: (-item[0], item[2]))[:max_sentences]
    if not selected:
        selected = [(results[0].score, results[0].chunk.text.strip(), results[0].chunk.chunk_id)]
    answer = " ".join(f"{sentence}. [{chunk_id}]" for _, sentence, chunk_id in selected)
    return f"{FALLBACK_LABEL}\n\n{answer}", [chunk_id for _, _, chunk_id in selected]
