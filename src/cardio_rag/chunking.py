import re
from collections.abc import Iterable

from .schemas import Chunk

HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


def chunk_text(text: str, document: dict[str, str], chunk_words: int = 550, overlap_words: int = 80) -> list[Chunk]:
    if chunk_words < 1 or not 0 <= overlap_words < chunk_words:
        raise ValueError("invalid chunk size or overlap")
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Document"
    current_words: list[str] = []
    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            if current_words:
                sections.append((current_heading, current_words))
            current_heading = match.group(2).strip()
            current_words = []
        else:
            current_words.extend(line.split())
    if current_words:
        sections.append((current_heading, current_words))

    chunks: list[Chunk] = []
    for section, words in sections:
        step = chunk_words - overlap_words
        for start in range(0, len(words), step):
            part = words[start : start + chunk_words]
            if not part:
                break
            number = len(chunks) + 1
            chunks.append(Chunk(
                chunk_id=f"{document['document_id']}:{number:04d}",
                document_id=document["document_id"], title=document["title"],
                publisher=document["publisher"], url=document["url"],
                accessed_date=document["accessed_date"], license_note=document["license_note"],
                section=section, text=" ".join(part), published_date=document.get("published_date"),
            ))
            if start + chunk_words >= len(words):
                break
    return chunks


def iter_chunks(texts: Iterable[tuple[str, dict[str, str]]], chunk_words: int, overlap_words: int) -> list[Chunk]:
    return [chunk for text, metadata in texts for chunk in chunk_text(text, metadata, chunk_words, overlap_words)]
