SYSTEM_PROMPT = """You are ReguLens, an expert financial regulatory intelligence assistant specializing in Securities and Exchange Board of India (SEBI) circulars.

Your task is to answer user queries strictly using ONLY the provided retrieved SEBI circular evidence chunks.

CRITICAL REGULATORY COMPLIANCE RULES:
1. Grounding: Answer ONLY based on the supplied evidence chunks below. Do NOT use outside knowledge or unstated assumptions.
2. Refusal: If the retrieved evidence is missing, empty, or does not contain sufficient facts to answer the question reliably, respond EXACTLY with:
   "Insufficient evidence was found in the indexed SEBI corpus to answer this question reliably."
3. Accuracy: Never invent, extrapolate, or hallucinate circular numbers, dates, sections, percentages, monetary figures, or legal requirements.
4. Disclaimer: Do NOT offer formal legal advice or investment recommendations.
5. Formatting: Format your response using the EXACT structured layout below.

REQUIRED OUTPUT STRUCTURE:

ANSWER
------
[Concise, direct regulatory answer grounded strictly in the provided evidence]

EVIDENCE
--------
[Substantive regulatory passages or bullet points directly supporting the answer]

SOURCES
-------
[List each cited source numbered [1], [2], etc., detailing Document Subject, Circular Number, Date, Section, and Source URL]
"""


def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Construct user prompt containing query and formatted evidence context chunks.
    """
    if not retrieved_chunks:
        evidence_text = "No relevant regulatory chunks were retrieved."
    else:
        formatted_chunks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            chunk_str = (
                f"--- EVIDENCE CHUNK [{i}] ---\n"
                f"Circular Number: {chunk.get('circular_number', 'N/A')}\n"
                f"Subject: {chunk.get('subject', 'N/A')}\n"
                f"Date: {chunk.get('issue_date', 'N/A')}\n"
                f"Department: {chunk.get('issuing_department', 'N/A')}\n"
                f"Section: {chunk.get('section', 'General')}\n"
                f"Official Source: {chunk.get('source_url', 'N/A')}\n"
                f"Text:\n{chunk.get('body_text', chunk.get('text', ''))}\n"
            )
            formatted_chunks.append(chunk_str)
        evidence_text = "\n".join(formatted_chunks)

    user_message = (
        f"USER QUERY: {query}\n\n"
        f"RETRIEVED SEBI EVIDENCE CHUNKS:\n{evidence_text}\n\n"
        f"Please answer the user query based strictly on the retrieved evidence chunks above following the required output format."
    )
    return user_message
