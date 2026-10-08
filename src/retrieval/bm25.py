import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src import config

logger = logging.getLogger(__name__)


def regulatory_tokenizer(text: str) -> List[str]:
    """
    Tokenize regulatory text while preserving:
    - Circular numbers (e.g., SEBI/HO/CFD/2023/123 -> sebi/ho/cfd/2023/123)
    - Monetary amounts & percentages (e.g., ₹50, 15%)
    - Alphanumeric clause identifiers (e.g., 30A, 12(1)(a))
    """
    if not text:
        return []
    
    # Lowercase text
    text_lower = text.lower()
    
    # Extract alphanumeric tokens plus / - % ₹
    tokens = re.findall(r"[a-z0-9/%\-₹]+", text_lower)
    return tokens


class BM25Retriever:
    def __init__(self, index_path: Path = config.BM25_INDEX_PATH):
        self.index_path = index_path
        self.bm25: BM25Okapi = None
        self.chunks_map: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Tokenize chunk texts and build BM25Okapi index.
        """
        logger.info(f"Tokenizing {len(chunks)} chunks for BM25 indexing...")
        corpus_tokens = [regulatory_tokenizer(c["text"]) for c in chunks]
        
        logger.info("Initializing BM25Okapi index...")
        self.bm25 = BM25Okapi(corpus_tokens)
        self.chunks_map = chunks
        self.save()
        logger.info(f"BM25 index built successfully with {len(chunks)} documents.")

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks_map}, f)
        logger.info(f"Saved BM25 index artifact to {self.index_path}")

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index file missing at {self.index_path}")
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunks_map = data["chunks"]
        logger.info(f"Loaded BM25 index ({len(self.chunks_map)} chunks) from disk.")

    def search(self, query: str, top_k: int = config.TOP_K_BM25) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            self.load()

        tokenized_query = regulatory_tokenizer(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            # Only filter out strictly negative scores
            if score < 0:
                continue
            chunk = dict(self.chunks_map[idx])
            chunk["score"] = score
            chunk["retrieval_source"] = "bm25"
            results.append(chunk)

        return results
