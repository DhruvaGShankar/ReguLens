import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.chunking import process_corpus_into_chunks
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting section-aware chunking script...")

    if not config.CORPUS_DOCUMENTS_JSON.exists():
        logger.error(f"Corpus documents JSON not found at {config.CORPUS_DOCUMENTS_JSON}. Run build_corpus.py first.")
        sys.exit(1)

    with open(config.CORPUS_DOCUMENTS_JSON, "r", encoding="utf-8") as f:
        documents = json.load(f)

    logger.info(f"Loaded {len(documents)} documents from {config.CORPUS_DOCUMENTS_JSON}")

    chunks = process_corpus_into_chunks(documents)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, default=str)

    logger.info(f"Saved {len(chunks)} chunks to {config.CHUNKS_JSON}")

    # Log statistics
    word_counts = [c["word_count"] for c in chunks]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0

    logger.info("=== CHUNKING SUMMARY ===")
    logger.info(f"Total Documents: {len(documents)}")
    logger.info(f"Total Chunks Generated: {len(chunks)}")
    logger.info(f"Average Chunks per Document: {len(chunks) / len(documents):.2f}")
    logger.info(f"Chunk Word Count (Min / Avg / Max): {min_words} / {avg_words:.1f} / {max_words}")


if __name__ == "__main__":
    main()
