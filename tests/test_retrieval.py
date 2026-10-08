import pytest
from unittest.mock import MagicMock
from src.retrieval.hybrid import min_max_normalize, HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


def test_min_max_normalize():
    scores = [10.0, 20.0, 30.0]
    normed = min_max_normalize(scores)
    assert normed == [0.0, 0.5, 1.0]

    # Equal values case
    assert min_max_normalize([5.0, 5.0]) == [1.0, 1.0]


def test_hybrid_retriever_fusion():
    mock_dense = MagicMock()
    mock_bm25 = MagicMock()

    chunk_a = {"chunk_id": "c1", "text": "Mutual fund borrowing rule", "circular_number": "C1"}
    chunk_b = {"chunk_id": "c2", "text": "Research analyst disclosure", "circular_number": "C2"}

    mock_dense.search.return_value = [
        {**chunk_a, "score": 0.9, "retrieval_source": "dense"},
        {**chunk_b, "score": 0.4, "retrieval_source": "dense"},
    ]
    mock_bm25.search.return_value = [
        {**chunk_b, "score": 15.0, "retrieval_source": "bm25"},
    ]

    hybrid = HybridRetriever(vector_store=mock_dense, bm25_retriever=mock_bm25)
    results = hybrid.search_hybrid("borrowing requirements", alpha=0.5, candidate_k=2)

    assert len(results) == 2
    for r in results:
        assert "hybrid_score" in r
        assert "dense_score" in r
        assert "bm25_score" in r


def test_cross_encoder_reranker(monkeypatch):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.12, 0.95]

    monkeypatch.setattr("src.retrieval.reranker.get_reranker_model", lambda model_name: mock_model)

    candidates = [
        {"chunk_id": "c1", "text": "General preamble text"},
        {"chunk_id": "c2", "text": "Specific borrowing restriction limit 20%"},
    ]

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("borrowing limits", candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[0]["rerank_score"] == 0.95
