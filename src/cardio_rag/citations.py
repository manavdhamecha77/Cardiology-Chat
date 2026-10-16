import re

from .schemas import RetrievedChunk

CITATION = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


def validate_citations(answer: str, retrieved: list[RetrievedChunk]) -> tuple[bool, list[str]]:
    citation_ids = CITATION.findall(answer)
    allowed = {item.chunk.chunk_id for item in retrieved}
    paragraphs = [part.strip() for part in answer.splitlines() if part.strip() and not part.startswith("Extractive demo response")]
    valid = bool(paragraphs) and all(CITATION.search(part) for part in paragraphs) and bool(citation_ids) and set(citation_ids) <= allowed
    return valid, citation_ids
