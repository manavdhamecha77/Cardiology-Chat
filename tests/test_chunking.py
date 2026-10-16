from cardio_rag.chunking import chunk_text


def test_chunk_ids_are_stable_and_sections_preserved():
    metadata = {"document_id": "guide", "title": "Guide", "publisher": "Org", "url": "https://example.org", "accessed_date": "2026-10-16", "license_note": "Public"}
    content = "# Overview\n" + "alpha " * 12 + "\n## Risk factors\n" + "beta " * 12
    first = chunk_text(content, metadata, chunk_words=5, overlap_words=1)
    second = chunk_text(content, metadata, chunk_words=5, overlap_words=1)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert {item.section for item in first} == {"Overview", "Risk factors"}
