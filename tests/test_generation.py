import pytest
from unittest.mock import MagicMock
from src.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.generation.rag import ReguLensRAG, call_llm


@pytest.fixture
def sample_evidence():
    return [
        {
            "chunk_id": "c1",
            "circular_number": "SEBI/HO/IMD/2022/456",
            "subject": "Mutual fund borrowing restrictions",
            "issue_date": "2022-04-10",
            "issuing_department": "IMD",
            "section": "Borrowing Limits",
            "source_url": "https://www.sebi.gov.in/test.html",
            "text": "Mutual funds shall not borrow except to meet temporary liquidity requirements.",
            "body_text": "Mutual funds shall not borrow except to meet temporary liquidity requirements."
        }
    ]


def test_build_rag_prompt(sample_evidence):
    prompt = build_rag_prompt("What are borrowing limits?", sample_evidence)
    assert "USER QUERY: What are borrowing limits?" in prompt
    assert "SEBI/HO/IMD/2022/456" in prompt
    assert "Mutual fund borrowing restrictions" in prompt


def test_rag_empty_evidence_refusal():
    mock_retriever = MagicMock()
    mock_retriever.search_dense.return_value = []

    rag = ReguLensRAG(hybrid_retriever=mock_retriever)
    res = rag.answer_query("Unknown query outside corpus", mode="dense")

    assert res["refused"] is True
    assert "Insufficient evidence was found in the indexed SEBI corpus" in res["answer"]


def test_rag_with_mock_llm(sample_evidence, monkeypatch):
    mock_retriever = MagicMock()
    mock_retriever.search_dense.return_value = sample_evidence

    monkeypatch.setattr(
        "src.generation.rag.call_llm",
        lambda prompt: "ANSWER\n------\nMutual funds may only borrow for temporary liquidity."
    )

    rag = ReguLensRAG(hybrid_retriever=mock_retriever)
    res = rag.answer_query("Borrowing by mutual funds", mode="dense")

    assert res["refused"] is False
    assert "Mutual funds may only borrow" in res["answer"]
    assert len(res["evidence_chunks"]) == 1
