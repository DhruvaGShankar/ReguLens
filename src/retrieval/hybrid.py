import logging
from typing import List, Dict, Any
from src import config
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


def min_max_normalize(scores: List[float]) -> List[float]:
    """
    Min-max normalize a list of scores into [0, 1].
    """
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    if max_val == min_val:
        return [1.0 if max_val > 0 else 0.0 for _ in scores]
    return [(s - min_val) / (max_val - min_val) for s in scores]


class HybridRetriever:
    def __init__(
        self,
        vector_store: FAISSVectorStore = None,
        bm25_retriever: BM25Retriever = None
    ):
        self.vector_store = vector_store or FAISSVectorStore()
        self.bm25_retriever = bm25_retriever or BM25Retriever()

    def search_dense(self, query: str, top_k: int = config.TOP_K_DENSE) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, top_k=top_k)

    def search_bm25(self, query: str, top_k: int = config.TOP_K_BM25) -> List[Dict[str, Any]]:
        return self.bm25_retriever.search(query, top_k=top_k)

    def search_hybrid(
        self,
        query: str,
        top_k_dense: int = config.TOP_K_DENSE,
        top_k_bm25: int = config.TOP_K_BM25,
        alpha: float = config.HYBRID_ALPHA,
        candidate_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining dense semantic search and BM25 lexical search.
        Normalizes scores independently and performs weighted score fusion:
            hybrid_score = alpha * dense_norm + (1 - alpha) * bm25_norm
        """
        dense_results = self.search_dense(query, top_k=top_k_dense)
        bm25_results = self.search_bm25(query, top_k=top_k_bm25)

        # Build combined candidate dictionary keyed by chunk_id
        candidates: Dict[str, Dict[str, Any]] = {}
        
        # Dense scores map
        dense_map = {c["chunk_id"]: c for c in dense_results}
        bm25_map = {c["chunk_id"]: c for c in bm25_results}

        all_chunk_ids = list(set(dense_map.keys()).union(bm25_map.keys()))

        if not all_chunk_ids:
            return []

        # Extract raw scores for normalization
        dense_raw_scores = [dense_map[cid]["score"] if cid in dense_map else 0.0 for cid in all_chunk_ids]
        bm25_raw_scores = [bm25_map[cid]["score"] if cid in bm25_map else 0.0 for cid in all_chunk_ids]

        dense_norm_scores = min_max_normalize(dense_raw_scores)
        bm25_norm_scores = min_max_normalize(bm25_raw_scores)

        fused_results = []
        for i, cid in enumerate(all_chunk_ids):
            # Base metadata from whichever retriever found the chunk
            chunk_data = dict(dense_map.get(cid) or bm25_map.get(cid))

            dense_score = dense_raw_scores[i]
            bm25_score = bm25_raw_scores[i]

            d_norm = dense_norm_scores[i]
            b_norm = bm25_norm_scores[i]

            hybrid_score = (alpha * d_norm) + ((1.0 - alpha) * b_norm)

            chunk_data["dense_score"] = dense_score
            chunk_data["bm25_score"] = bm25_score
            chunk_data["hybrid_score"] = hybrid_score
            chunk_data["score"] = hybrid_score
            chunk_data["retrieval_source"] = "hybrid"

            fused_results.append(chunk_data)

        # Sort by hybrid score descending
        fused_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return fused_results[:candidate_k]
