import logging
from typing import List, Dict, Any
from src.ingestion.preprocessing import clean_regulatory_text, extract_structural_sections
from src import config

logger = logging.getLogger(__name__)


def word_count(text: str) -> int:
    return len(text.split())


def create_chunks_from_document(
    doc: Dict[str, Any],
    target_words: int = 450,  # ~600 tokens
    overlap_words: int = 75   # ~100 tokens
) -> List[Dict[str, Any]]:
    """
    Produce section-aware chunks for a single document object, preserving parent metadata.
    """
    doc_id = doc.get("document_id", "doc_unknown")
    circular_number = doc.get("circular_number", "")
    subject = doc.get("subject", "")
    issue_date = doc.get("issue_date", "")
    effective_date = doc.get("effective_date", None)
    department = doc.get("issuing_department", "UNKNOWN")
    circular_type = doc.get("circular_type", "CIRCULAR")
    supersession_status = doc.get("supersession_status", "in_force")
    validity_status = doc.get("validity_status", "current")
    source_url = doc.get("source_url", "")
    full_text = doc.get("text", "")

    cleaned_full_text = clean_regulatory_text(full_text)
    sections = extract_structural_sections(cleaned_full_text)

    chunks = []
    chunk_counter = 1

    for sec in sections:
        sec_title = sec["section_title"]
        sec_text = sec["text"]
        words = sec_text.split()

        if not words:
            continue

        # If section is within target size, process as single chunk
        if len(words) <= target_words + overlap_words:
            sub_chunks = [sec_text]
        else:
            # Sliding window over section text
            sub_chunks = []
            start = 0
            while start < len(words):
                end = start + target_words
                chunk_words = words[start:end]
                sub_chunks.append(" ".join(chunk_words))
                if end >= len(words):
                    break
                start += (target_words - overlap_words)

        for sub_txt in sub_chunks:
            chunk_id = f"{doc_id}_chunk_{chunk_counter:04d}"
            
            context_header = (
                f"[SEBI Circular: {circular_number} | "
                f"Date: {issue_date} | "
                f"Dept: {department} | "
                f"Section: {sec_title}]"
            )

            # Combined text includes context header for retrieval accuracy
            combined_text = f"{context_header}\n{sub_txt}"

            chunk_obj = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "circular_number": circular_number,
                "subject": subject,
                "issue_date": issue_date,
                "effective_date": effective_date,
                "issuing_department": department,
                "circular_type": circular_type,
                "supersession_status": supersession_status,
                "validity_status": validity_status,
                "source_url": source_url,
                "section": sec_title,
                "context_header": context_header,
                "text": combined_text,
                "body_text": sub_txt,
                "word_count": len(sub_txt.split())
            }
            chunks.append(chunk_obj)
            chunk_counter += 1

    return chunks


def process_corpus_into_chunks(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all documents in the corpus list into section-aware chunks.
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = create_chunks_from_document(doc)
        all_chunks.extend(doc_chunks)
    logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} total chunks.")
    return all_chunks
