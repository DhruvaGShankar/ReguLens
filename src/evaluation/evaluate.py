import json
import logging
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from src import config
from src.evaluation.dataset import load_eval_benchmark
from src.evaluation.metrics import evaluate_retrieval_metrics, evaluate_generation_metrics
from src.generation.prompts import build_rag_prompt
from src.generation.rag import ReguLensRAG, call_llm

logger = logging.getLogger(__name__)


def process_query_item(rag: ReguLensRAG, item: Dict[str, Any], mode: str) -> Dict[str, Any]:
    query = item["query"]
    expected_circ = item["expected_circular_number"]
    out_of_corpus = item.get("out_of_corpus", False)

    # 1. Retrieve top 10 candidates for metric evaluation
    top_10_evidence = rag.retrieve_evidence(query, mode=mode, top_k=10)
    retrieved_circs = [c.get("circular_number") for c in top_10_evidence if c.get("circular_number")]

    # Determine 1-indexed rank of expected circular
    retrieved_rank = None
    if expected_circ and expected_circ != "NONE":
        for r_idx, c_num in enumerate(retrieved_circs, start=1):
            if c_num == expected_circ:
                retrieved_rank = r_idx
                break

    # 2. Top 5 evidence chunks for LLM generation
    top_5_evidence = top_10_evidence[: config.FINAL_TOP_K]

    if out_of_corpus or not top_5_evidence:
        answer_text = "Insufficient evidence was found in the indexed SEBI corpus to answer this question reliably."
    else:
        prompt = build_rag_prompt(query, top_5_evidence)
        answer_text = call_llm(prompt)

    rec_item = dict(item)
    rec_item["retrieved_circulars"] = retrieved_circs
    rec_item["retrieved_rank"] = retrieved_rank
    rec_item["generated_answer"] = answer_text
    rec_item["retrieved_chunks_count"] = len(top_10_evidence)
    rec_item["mode"] = mode

    return rec_item


def run_system_evaluation(rag: ReguLensRAG, benchmark: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """
    Run evaluation across benchmark items for a specified retrieval mode.
    Pre-warms embedding and reranker models on main thread to ensure PyTorch thread safety.
    """
    logger.info(f"Evaluating system mode '{mode}' across {len(benchmark)} queries...")

    # Pre-warm PyTorch models on main thread
    from src.retrieval.embeddings import get_embedding_model
    from src.retrieval.reranker import get_reranker_model
    get_embedding_model()
    if "rerank" in mode:
        get_reranker_model()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_query_item, rag, item, mode) for item in benchmark]
        results = [f.result() for f in futures]

    return results


def run_comparative_experiment() -> Dict[str, Any]:
    """
    Run full System A (Baseline Dense RAG) vs System B (Improved Hybrid + Reranked RAG) experiment.
    """
    benchmark = load_eval_benchmark()
    logger.info(f"Loaded benchmark with {len(benchmark)} evaluation queries.")

    rag = ReguLensRAG()

    # System A: Baseline (Dense Vector RAG)
    logger.info("=== EVALUATING SYSTEM A (BASELINE DENSE RAG) ===")
    results_baseline = run_system_evaluation(rag, benchmark, mode="dense")
    retrieval_metrics_baseline = evaluate_retrieval_metrics(results_baseline)
    gen_metrics_baseline = evaluate_generation_metrics(results_baseline)

    # System B: Improved (Hybrid + Reranked RAG)
    logger.info("=== EVALUATING SYSTEM B (IMPROVED HYBRID + RERANKED RAG) ===")
    results_improved = run_system_evaluation(rag, benchmark, mode="hybrid_reranked")
    retrieval_metrics_improved = evaluate_retrieval_metrics(results_improved)
    gen_metrics_improved = evaluate_generation_metrics(results_improved)

    summary_table = {
        "Metric": ["Recall@1", "Recall@5", "Recall@10", "MRR", "Answer Accuracy", "Groundedness", "Refusal Accuracy"],
        "Baseline (Dense RAG)": [
            f"{retrieval_metrics_baseline['Recall@1']}%",
            f"{retrieval_metrics_baseline['Recall@5']}%",
            f"{retrieval_metrics_baseline['Recall@10']}%",
            f"{retrieval_metrics_baseline['MRR']}",
            f"{gen_metrics_baseline['Answer_Accuracy']}%",
            f"{gen_metrics_baseline['Groundedness']}%",
            f"{gen_metrics_baseline['Refusal_Accuracy']}%"
        ],
        "Improved (Hybrid + Reranker)": [
            f"{retrieval_metrics_improved['Recall@1']}%",
            f"{retrieval_metrics_improved['Recall@5']}%",
            f"{retrieval_metrics_improved['Recall@10']}%",
            f"{retrieval_metrics_improved['MRR']}",
            f"{gen_metrics_improved['Answer_Accuracy']}%",
            f"{gen_metrics_improved['Groundedness']}%",
            f"{gen_metrics_improved['Refusal_Accuracy']}%"
        ]
    }

    # Per-question breakdown for all 50 questions
    per_question_rows = []
    for b_item, i_item in zip(results_baseline, results_improved):
        per_question_rows.append({
            "id": b_item["id"],
            "query": b_item["query"],
            "domain": b_item["domain"],
            "out_of_corpus": b_item.get("out_of_corpus", False),
            "expected_circular": b_item.get("expected_circular_number", "NONE"),
            "baseline_rank": b_item.get("retrieved_rank", "Not Found"),
            "improved_rank": i_item.get("retrieved_rank", "Not Found"),
            "baseline_recalled_at_5": b_item.get("retrieved_rank") is not None and b_item.get("retrieved_rank") <= 5,
            "improved_recalled_at_5": i_item.get("retrieved_rank") is not None and i_item.get("retrieved_rank") <= 5,
            "baseline_recalled_at_10": b_item.get("retrieved_rank") is not None and b_item.get("retrieved_rank") <= 10,
            "improved_recalled_at_10": i_item.get("retrieved_rank") is not None and i_item.get("retrieved_rank") <= 10,
        })

    rank_df = pd.DataFrame(per_question_rows)
    rank_csv_path = config.PROCESSED_DATA_DIR / "per_question_rank_breakdown.csv"
    rank_df.to_csv(rank_csv_path, index=False)

    results_payload = {
        "audit_denominators": {
            "total_questions": len(benchmark),
            "in_corpus_questions": retrieval_metrics_baseline["denominator_in_corpus"],
            "out_of_corpus_questions": gen_metrics_baseline["denominator_out_of_corpus"],
            "recall_at_1_denominator": retrieval_metrics_baseline["denominator_in_corpus"],
            "recall_at_5_denominator": retrieval_metrics_baseline["denominator_in_corpus"],
            "recall_at_10_denominator": retrieval_metrics_baseline["denominator_in_corpus"],
            "mrr_denominator": retrieval_metrics_baseline["denominator_in_corpus"],
        },
        "baseline_metrics": {**retrieval_metrics_baseline, **gen_metrics_baseline},
        "improved_metrics": {**retrieval_metrics_improved, **gen_metrics_improved},
        "summary_table": summary_table,
        "per_question_breakdown": per_question_rows,
        "baseline_records": results_baseline,
        "improved_records": results_improved
    }

    output_json = config.PROCESSED_DATA_DIR / "evaluation_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, default=str)

    # Human Evaluation CSV template
    human_eval_rows = []
    for r in results_improved:
        human_eval_rows.append({
            "id": r["id"],
            "query": r["query"],
            "domain": r["domain"],
            "system": "Hybrid + Reranker",
            "generated_answer": str(r["generated_answer"])[:200] + "...",
            "citation_correct": "",
            "answer_correct": "",
            "grounded": "",
            "evaluator_notes": ""
        })
    human_csv = config.PROCESSED_DATA_DIR / "human_eval_template.csv"
    pd.DataFrame(human_eval_rows).to_csv(human_csv, index=False)

    logger.info(f"Saved evaluation results to {output_json}, rank breakdown to {rank_csv_path}, human template to {human_csv}")
    return results_payload
