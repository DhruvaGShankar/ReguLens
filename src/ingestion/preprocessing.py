import re
import logging

logger = logging.getLogger(__name__)

# Common section / clause header regex patterns in SEBI circulars
SECTION_HEADER_REGEX = re.compile(
    r"^(?:"
    r"(?:CHAPTER|Chapter|PART|Part|SECTION|Section)\s+[IVXLCDM\d]+"
    r"|(?:\d+\.){1,3}\s+[A-Z0-9]"
    r"|(?:\d+\.)\s+[A-Z]"
    r"|(?:\([a-z0-9]+\))\s+[A-Z]"
    r"|Sub:\s+.*"
    r"|Subject:\s+.*"
    r")",
    re.MULTILINE
)

def clean_regulatory_text(text: str) -> str:
    """
    Clean text while preserving regulatory structure:
    - Normalizes null bytes and trailing whitespace per line.
    - Consolidates 3+ consecutive newlines to 2 newlines.
    - Preserves section headings, bullet points, numbers, percentages, dates, and clause tags.
    """
    if not text:
        return ""

    # Replace carriage returns
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Strip null characters or odd control codes except tab & newline
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)

    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)

    # Consolidate excessive blank lines (more than 2 consecutive blank lines -> 2)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def extract_structural_sections(text: str) -> list[dict]:
    """
    Break regulatory circular text into sections based on detected structural headings.
    Returns a list of dicts with 'section_title' and 'text'.
    """
    text = clean_regulatory_text(text)
    if not text:
        return [{"section_title": "General", "text": ""}]

    lines = text.split("\n")
    sections = []
    current_title = "Preamble / General"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if SECTION_HEADER_REGEX.match(stripped) and len(stripped) < 120:
            if current_lines:
                sec_text = "\n".join(current_lines).strip()
                if sec_text:
                    sections.append({"section_title": current_title, "text": sec_text})
                current_lines = []
            current_title = stripped
        current_lines.append(line)

    if current_lines:
        sec_text = "\n".join(current_lines).strip()
        if sec_text:
            sections.append({"section_title": current_title, "text": sec_text})

    if not sections:
        sections.append({"section_title": "General", "text": text})

    return sections
