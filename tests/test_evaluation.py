import pytest
from src.evaluation.metrics import (
    calculate_recall_at_k,
    calculate_mrr,
    evaluate_retrieval_metrics,
    evaluate_generation_metrics
)


def test_recall_at_k():
    retrieved = ["CIRC_001", "CIRC_002", "CIRC_003"]
    assert calculate_recall_at_k(retrieved, "CIRC_001", 1) == 1.0
    assert calculate_recall_at_k(retrieved, "CIRC_002", 1) == 0.0
    assert calculate_recall_at_k(retrieved, "CIRC_002", 5) == 1.0
    assert calculate_recall_at_k(retrieved, "CIRC_999", 5) == 0.0


def test_mrr():
    retrieved = ["CIRC_A", "CIRC_B", "CIRC_C"]
    assert calculate_mrr(retrieved, "CIRC_A") == 1.0
    assert calculate_mrr(retrieved, "CIRC_B") == 0.5
    assert calculate_mrr(retrieved, "CIRC_C") == 1.0 / 3.0
    assert calculate_mrr(retrieved, "CIRC_MISSING") == 0.0


def test_evaluate_retrieval_metrics():
    records = [
        {
            "expected_circular_number": "CIRC_001",
            "retrieved_circulars": ["CIRC_001", "CIRC_002"],
            "out_of_corpus": False
        },
        {
            "expected_circular_number": "CIRC_003",
            "retrieved_circulars": ["CIRC_999", "CIRC_003"],
            "out_of_corpus": False
        }
    ]
    metrics = evaluate_retrieval_metrics(records)
    assert metrics["Recall@1"] == 50.0
    assert metrics["Recall@5"] == 100.0
    assert metrics["MRR"] == 0.75  # (1.0 + 0.5) / 2


def test_evaluate_generation_metrics():
    records = [
        {
            "generated_answer": "ANSWER\n------\nMutual funds may borrow up to 20%.",
            "expected_answer_keywords": ["borrow", "20%"],
            "out_of_corpus": False
        },
        {
            "generated_answer": "Insufficient evidence was found in the indexed SEBI corpus to answer this question reliably.",
            "expected_answer_keywords": [],
            "out_of_corpus": True
        }
    ]
    metrics = evaluate_generation_metrics(records)
    assert metrics["Groundedness"] == 100.0
    assert metrics["Refusal_Accuracy"] == 100.0
    assert metrics["Answer_Accuracy"] == 100.0
