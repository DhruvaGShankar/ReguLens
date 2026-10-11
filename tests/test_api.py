import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test GET /api/v1/health returns 200 and system readiness details."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert payload["version"] == "1.0.0"
    assert "timestamp" in payload
    assert "system_status" in payload
    assert "faiss_index_loaded" in payload["system_status"]
    assert "bm25_index_loaded" in payload["system_status"]


def test_chat_valid_query():
    """Test POST /api/v1/chat with valid parameters and mocked RAG response."""
    mock_rag_result = {
        "query": "What are mutual fund borrowing limits?",
        "answer": "ANSWER\n------\nMutual funds may borrow up to 20% of net assets.",
        "evidence_chunks": [
            {
                "chunk_id": "chunk_001",
                "document_id": "doc_001",
                "circular_number": "SEBI/HO/IMD/2021/01",
                "subject": "Mutual Fund Borrowing Guidelines",
                "issue_date": "2021-05-10",
                "section": "Borrowing Limits",
                "text": "Mutual funds may borrow up to 20% of net assets.",
                "source_url": "https://www.sebi.gov.in/test.html",
                "score": 0.92,
                "retrieval_source": "hybrid_reranked"
            }
        ],
        "retrieval_mode": "hybrid_reranked",
        "refused": False
    }

    with patch("src.api.routes.get_rag_service") as mock_get_rag:
        mock_rag = MagicMock()
        mock_rag.answer_query.return_value = mock_rag_result
        mock_get_rag.return_value = mock_rag

        response = client.post(
            "/api/v1/chat",
            json={
                "query": "What are mutual fund borrowing limits?",
                "mode": "hybrid_reranked",
                "top_k": 5,
                "alpha": 0.5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What are mutual fund borrowing limits?"
        assert "request_id" in data
        assert data["request_id"].startswith("req_")
        assert "Mutual funds may borrow" in data["answer"]
        assert len(data["evidence_chunks"]) == 1
        assert data["evidence_chunks"][0]["circular_number"] == "SEBI/HO/IMD/2021/01"
        assert data["refused"] is False
        assert data["metadata"]["retrieved_count"] == 1


def test_chat_empty_query():
    """Test POST /api/v1/chat rejects empty or whitespace-only queries with 422."""
    response = client.post("/api/v1/chat", json={"query": ""})
    assert response.status_code == 422
    err_payload = response.json()
    assert err_payload["error"]["code"] == "VALIDATION_ERROR"

    response_ws = client.post("/api/v1/chat", json={"query": "   \n\t  "})
    assert response_ws.status_code == 422


def test_chat_unsupported_mode():
    """Test POST /api/v1/chat rejects invalid retrieval mode enum strings with 422."""
    response = client.post(
        "/api/v1/chat",
        json={"query": "What is Regulation 30?", "mode": "unsupported_custom_mode"}
    )
    assert response.status_code == 422
    err_payload = response.json()
    assert err_payload["error"]["code"] == "VALIDATION_ERROR"


def test_chat_invalid_top_k_and_alpha():
    """Test POST /api/v1/chat enforces bounds on top_k (1-15) and alpha (0.0-1.0)."""
    # top_k > 15
    res_top_k = client.post("/api/v1/chat", json={"query": "Query", "top_k": 25})
    assert res_top_k.status_code == 422

    # top_k < 1
    res_top_k_zero = client.post("/api/v1/chat", json={"query": "Query", "top_k": 0})
    assert res_top_k_zero.status_code == 422

    # alpha > 1.0
    res_alpha = client.post("/api/v1/chat", json={"query": "Query", "alpha": 1.5})
    assert res_alpha.status_code == 422


def test_chat_refusal_response():
    """Test POST /api/v1/chat correctly handles refusal / insufficient evidence."""
    refusal_result = {
        "query": "What are crypto rules under EU MiCA?",
        "answer": "Insufficient evidence was found in the indexed SEBI corpus to answer this question reliably.",
        "evidence_chunks": [],
        "retrieval_mode": "hybrid_reranked",
        "refused": True
    }

    with patch("src.api.routes.get_rag_service") as mock_get_rag:
        mock_rag = MagicMock()
        mock_rag.answer_query.return_value = refusal_result
        mock_get_rag.return_value = mock_rag

        response = client.post("/api/v1/chat", json={"query": "What are crypto rules under EU MiCA?"})
        assert response.status_code == 200
        data = response.json()
        assert data["refused"] is True
        assert "Insufficient evidence" in data["answer"]
        assert len(data["evidence_chunks"]) == 0


def test_chat_rate_limit_handling():
    """Test POST /api/v1/chat converts upstream RateLimitError into HTTP 429."""
    with patch("src.api.routes.get_rag_service") as mock_get_rag:
        mock_rag = MagicMock()
        mock_rag.answer_query.side_effect = Exception("Rate limit exceeded: 429 Error")
        mock_get_rag.return_value = mock_rag

        response = client.post("/api/v1/chat", json={"query": "Query causing rate limit"})
        assert response.status_code == 429
        data = response.json()
        assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert "rate limit" in data["error"]["message"].lower()


def test_chat_upstream_failure_handling():
    """Test POST /api/v1/chat converts upstream 503 / network errors into HTTP 503."""
    with patch("src.api.routes.get_rag_service") as mock_get_rag:
        mock_rag = MagicMock()
        mock_rag.answer_query.side_effect = Exception("Service Unavailable 503")
        mock_get_rag.return_value = mock_rag

        response = client.post("/api/v1/chat", json={"query": "Query causing 503"})
        assert response.status_code == 503
        data = response.json()
        assert data["error"]["code"] == "SERVICE_UNAVAILABLE"
