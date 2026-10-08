import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.bm25 import BM25Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting index construction script...")

    if not config.CHUNKS_JSON.exists():
        logger.error(f"Chunks JSON file not found at {config.CHUNKS_JSON}. Run build_chunks.py first.")
        sys.exit(1)

    with open(config.CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {config.CHUNKS_JSON}")

    # 1. Build FAISS Vector Index
    logger.info("--- BUILDING FAISS DENSE VECTOR INDEX ---")
    vector_store = FAISSVectorStore()
    vector_store.build_index(chunks)

    # 2. Build BM25 Lexical Index
    logger.info("--- BUILDING BM25 LEXICAL INDEX ---")
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(chunks)

    logger.info("=== INDEXING SUMMARY ===")
    logger.info(f"Total Indexed Chunks: {len(chunks)}")
    logger.info(f"FAISS Index Path: {config.FAISS_INDEX_PATH}")
    logger.info(f"BM25 Index Path: {config.BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
