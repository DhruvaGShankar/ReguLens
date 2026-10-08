import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from src import config
from src.retrieval.embeddings import generate_embeddings

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    def __init__(self, index_path: Path = config.FAISS_INDEX_PATH, map_path: Path = config.FAISS_ID_MAP_PATH):
        self.index_path = index_path
        self.map_path = map_path
        self.index = None
        self.chunks_map: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]], batch_size: int = 64):
        """
        Build FAISS IndexFlatIP index over chunk text embeddings and persist map.
        """
        logger.info(f"Generating dense embeddings for {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(texts, batch_size=batch_size)

        dimension = embeddings.shape[1]
        logger.info(f"Initializing FAISS IndexFlatIP (dimension={dimension})...")
        
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.chunks_map = chunks

        self.save()
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_map, f, indent=2, default=str)
        logger.info(f"Saved FAISS index to {self.index_path} and ID map to {self.map_path}")

    def load(self):
        if not self.index_path.exists() or not self.map_path.exists():
            raise FileNotFoundError(f"FAISS index or ID map missing at {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.map_path, "r", encoding="utf-8") as f:
            self.chunks_map = json.load(f)
        logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors) and ID map from disk.")

    def search(self, query: str, top_k: int = config.TOP_K_DENSE) -> List[Dict[str, Any]]:
        if self.index is None:
            self.load()

        query_embedding = generate_embeddings([query])
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_map):
                continue
            chunk = dict(self.chunks_map[idx])
            chunk["score"] = float(score)
            chunk["retrieval_source"] = "dense"
            results.append(chunk)

        return results
