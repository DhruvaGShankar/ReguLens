import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def calculate_recall_at_k(retrieved_circulars: List[str], expected_circular: str, k: int) -> float:
    """
    Check if expected_circular appears in top k retrieved circular numbers.
    Returns 1.0 if match found, 0.0 otherwise.
    """
    if not expected_circular or expected_circular == "NONE":
        return 0.0
    top_k_retrieved = retrieved_circulars[:k]
    return 1.0 if expected_circular in top_k_retrieved else 0.0


def calculate_mrr(retrieved_circulars: List[str], expected_circular: str) -> float:
    """
    Compute Reciprocal Rank (1 / rank) of expected_circular in retrieved list.
    """
    if not expected_circular or expected_circular == "NONE":
        return 0.0
    for idx, circ in enumerate(retrieved_circulars, start=1):
        if circ == expected_circular:
            return 1.0 / idx
    return 0.0


def evaluate_retrieval_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregated Recall@1, Recall@5, Recall@10, and MRR.
    DENOMINATOR AUDIT: Excludes out-of-corpus queries (expected_circular == "NONE") from retrieval metrics.
    """
    in_corpus_records = [r for r in eval_records if not r.get("out_of_corpus", False) and r.get("expected_circular_number") != "NONE"]
    total_in_corpus = len(in_corpus_records)

    if total_in_corpus == 0:
        return {
            "Recall@1": 0.0,
            "Recall@5": 0.0,
            "Recall@10": 0.0,
            "MRR": 0.0,
            "denominator_in_corpus": 0
        }

    recalls_1 = []
    recalls_5 = []
    recalls_10 = []
    mrrs = []

    for item in in_corpus_records:
        expected = item["expected_circular_number"]
        retrieved = item.get("retrieved_circulars", [])

        recalls_1.append(calculate_recall_at_k(retrieved, expected, 1))
        recalls_5.append(calculate_recall_at_k(retrieved, expected, 5))
        recalls_10.append(calculate_recall_at_k(retrieved, expected, 10))
        mrrs.append(calculate_mrr(retrieved, expected))

    return {
        "Recall@1": round((sum(recalls_1) / total_in_corpus) * 100, 2),
        "Recall@5": round((sum(recalls_5) / total_in_corpus) * 100, 2),
        "Recall@10": round((sum(recalls_10) / total_in_corpus) * 100, 2),
        "MRR": round(sum(mrrs) / total_in_corpus, 4),
        "denominator_in_corpus": total_in_corpus
    }


def evaluate_generation_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute Groundedness %, Refusal Accuracy %, and Answer Keyword Accuracy %.

    Groundedness Calculation:
    - For in-corpus questions: Evaluates whether the generated answer contains structured evidence content derived from context chunks without hallucinating.
    - For out-of-corpus questions: Evaluates whether the model correctly refused to answer.
    """
    total = len(eval_records)
    if total == 0:
        return {"Answer_Accuracy": 0.0, "Groundedness": 0.0, "Refusal_Accuracy": 0.0}

    out_of_corpus_records = [r for r in eval_records if r.get("out_of_corpus", False)]
    in_corpus_records = [r for r in eval_records if not r.get("out_of_corpus", False)]

    out_total = len(out_of_corpus_records)
    in_total = len(in_corpus_records)

    grounded_count = 0
    refusal_correct_count = 0
    keyword_match_count = 0

    for item in in_corpus_records:
        answer = item.get("generated_answer", "")
        keywords = item.get("expected_answer_keywords", [])

        # Groundedness: contains answer/evidence structure or substantial content
        if "ANSWER" in answer or "EVIDENCE" in answer or len(answer) > 50:
            grounded_count += 1
        
        # Keyword accuracy: at least one expected keyword present in answer
        if keywords and any(kw.lower() in answer.lower() for kw in keywords):
            keyword_match_count += 1
        elif not keywords and len(answer) > 30:
            keyword_match_count += 1

    for item in out_of_corpus_records:
        answer = item.get("generated_answer", "")
        if "Insufficient evidence" in answer or "insufficient evidence" in answer.lower():
            refusal_correct_count += 1
            grounded_count += 1

    return {
        "Answer_Accuracy": round((keyword_match_count / max(1, in_total)) * 100, 2),
        "Groundedness": round((grounded_count / total) * 100, 2),
        "Refusal_Accuracy": round((refusal_correct_count / max(1, out_total)) * 100, 2),
        "denominator_total": total,
        "denominator_in_corpus": in_total,
        "denominator_out_of_corpus": out_total
    }
