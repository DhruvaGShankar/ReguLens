import pytest
from src.ingestion.preprocessing import clean_regulatory_text, extract_structural_sections
from src.ingestion.chunking import create_chunks_from_document, process_corpus_into_chunks


@pytest.fixture
def dummy_document():
    return {
        "document_id": "doc_0001_SEBI_HO_CFD_2023_123",
        "circular_number": "SEBI/HO/CFD/2023/123",
        "subject": "Disclosure requirements for listed entities",
        "issue_date": "2023-07-15",
        "effective_date": "2023-08-01",
        "issuing_department": "CFD",
        "circular_type": "CIRCULAR",
        "supersession_status": "in_force",
        "validity_status": "current",
        "source_url": "https://www.sebi.gov.in/test.html",
        "text": """SEBI Circular No. SEBI/HO/CFD/2023/123

1. Objective
The Securities and Exchange Board of India (SEBI) hereby mandates revised disclosure requirements.
All listed entities shall comply with 15% threshold limits under Regulation 30.

2. Scope and Applicability
This circular applies to all mutual funds, research analysts, and stock exchanges.
The net worth requirement is set at ₹50 crore.
"""
    }


def test_clean_regulatory_text():
    raw_text = "Line 1\r\n\r\n\r\n\r\nLine 2   \nSection 1.1 Clause 50%"
    cleaned = clean_regulatory_text(raw_text)
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Section 1.1 Clause 50%" in cleaned


def test_extract_structural_sections(dummy_document):
    sections = extract_structural_sections(dummy_document["text"])
    assert len(sections) >= 2
    sec_titles = [s["section_title"] for s in sections]
    assert any("1. Objective" in t for t in sec_titles)


def test_create_chunks_metadata_preservation(dummy_document):
    chunks = create_chunks_from_document(dummy_document)
    assert len(chunks) >= 1
    
    first_chunk = chunks[0]
    assert first_chunk["document_id"] == dummy_document["document_id"]
    assert first_chunk["circular_number"] == dummy_document["circular_number"]
    assert first_chunk["subject"] == dummy_document["subject"]
    assert first_chunk["source_url"] == dummy_document["source_url"]
    assert first_chunk["issuing_department"] == dummy_document["issuing_department"]
    assert "chunk_id" in first_chunk
    assert "context_header" in first_chunk
    assert first_chunk["chunk_id"].startswith("doc_0001_SEBI_HO_CFD_2023_123_chunk_")


def test_process_corpus_into_chunks(dummy_document):
    corpus = [dummy_document, dummy_document]
    chunks = process_corpus_into_chunks(corpus)
    assert len(chunks) >= 2
