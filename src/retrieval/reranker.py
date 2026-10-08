import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src import config

logger = logging.getLogger(__name__)

_RERANKER_MODEL_CACHE = None

def get_reranker_model(model_name: str = config.RERANKER_MODEL) -> CrossEncoder:
    global _RERANKER_MODEL_CACHE
    if _RERANKER_MODEL_CACHE is None:
        logger.info(f"Loading CrossEncoder model '{model_name}'...")
        _RERANKER_MODEL_CACHE = CrossEncoder(model_name)
    return _RERANKER_MODEL_CACHE


class CrossEncoderReranker:
    def __init__(self, model_name: str = config.RERANKER_MODEL):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = config.FINAL_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate chunks using CrossEncoder model.
        Returns top_k chunks sorted by reranker score descending.
        """
        if not candidates:
            return []

        model = get_reranker_model(self.model_name)
        pairs = [(query, c["text"]) for c in candidates]

        scores = model.predict(pairs)

        reranked_chunks = []
        for chunk, score in zip(candidates, scores):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = float(score)
            chunk_copy["score"] = float(score)
            chunk_copy["retrieval_source"] = "hybrid_reranked"
            reranked_chunks.append(chunk_copy)

        reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_chunks[:top_k]
