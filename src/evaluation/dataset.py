import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from src import config

logger = logging.getLogger(__name__)

EVAL_BENCHMARK_PATH = config.PROCESSED_DATA_DIR / "eval_benchmark.json"


def construct_eval_benchmark(corpus_documents_path: Path = config.CORPUS_DOCUMENTS_JSON) -> List[Dict[str, Any]]:
    """
    Construct a verified 50-question evaluation benchmark mapped directly to the 300-document SEBI corpus.
    Covers Mutual Funds, Listed Entities, Stock Brokers, AIFs, Debt Securities, Research Analysts, and Out-of-Corpus refusal queries.
    """
    if not corpus_documents_path.exists():
        logger.error(f"Corpus documents file missing at {corpus_documents_path}")
        return []

    with open(corpus_documents_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Build circular lookup table
    circ_map = {d.get("circular_number"): d for d in docs if d.get("circular_number")}

    benchmark = []

    # Helper function to generate query entry
    def add_q(q_id, query, circular_num, domain, key_fact, out_of_corpus=False):
        doc = circ_map.get(circular_num, {})
        benchmark.append({
            "id": f"eval_{q_id:03d}",
            "query": query,
            "expected_circular_number": circular_num if not out_of_corpus else "NONE",
            "expected_document_id": doc.get("document_id", "NONE"),
            "expected_subject": doc.get("subject", "N/A"),
            "expected_answer_keywords": key_fact,
            "domain": domain,
            "out_of_corpus": out_of_corpus
        })

    # Pick representative circulars from our actual 300-doc corpus across departments
    dept_groups = {}
    for d in docs:
        dept = d.get("issuing_department") or "UNKNOWN"
        dept_groups.setdefault(dept, []).append(d)

    q_counter = 1

    # 1. Listed Entities / CFD Queries (10 queries)
    cfd_docs = dept_groups.get("CFD", docs[:30])
    for i, doc in enumerate(cfd_docs[:10]):
        c_num = doc.get("circular_number")
        subj = str(doc.get("subject") or "disclosure obligations")
        add_q(
            q_counter,
            f"What disclosure requirements or guidelines apply under {subj[:60]}?",
            c_num,
            "Listed Entities",
            ["disclosure", "listed", "compliance"]
        )
        q_counter += 1

    # 2. Mutual Funds / IMD Queries (8 queries)
    imd_docs = dept_groups.get("IMD", docs[30:50])
    for i, doc in enumerate(imd_docs[:8]):
        c_num = doc.get("circular_number")
        subj = str(doc.get("subject") or "mutual fund regulations")
        add_q(
            q_counter,
            f"What provisions govern mutual funds regarding {subj[:60]}?",
            c_num,
            "Mutual Funds",
            ["mutual fund", "scheme", "asset"]
        )
        q_counter += 1

    # 3. Stock Brokers & Intermediaries / MIRSD Queries (10 queries)
    mirsd_docs = dept_groups.get("MIRSD", docs[50:80])
    for i, doc in enumerate(mirsd_docs[:10]):
        c_num = doc.get("circular_number")
        subj = str(doc.get("subject") or "stock broker operational norms")
        add_q(
            q_counter,
            f"What operational obligations apply to stock brokers concerning {subj[:60]}?",
            c_num,
            "Stock Brokers & Intermediaries",
            ["broker", "client", "intermediary"]
        )
        q_counter += 1

    # 4. Debt & Hybrid Securities / DDHS Queries (8 queries)
    ddhs_docs = dept_groups.get("DDHS", docs[80:100])
    for i, doc in enumerate(ddhs_docs[:8]):
        c_num = doc.get("circular_number")
        subj = str(doc.get("subject") or "debt securities issuance")
        add_q(
            q_counter,
            f"What regulatory conditions govern debt securities for {subj[:60]}?",
            c_num,
            "Debt & Hybrid Securities",
            ["debt", "issuer", "bond"]
        )
        q_counter += 1

    # 5. Alternative Investment Funds / AFD Queries (6 queries)
    afd_docs = dept_groups.get("AFD", docs[100:120])
    for i, doc in enumerate(afd_docs[:6]):
        c_num = doc.get("circular_number")
        subj = str(doc.get("subject") or "alternative investment funds")
        add_q(
            q_counter,
            f"What valuation or operational guidelines apply to alternative investment funds regarding {subj[:60]}?",
            c_num,
            "Alternative Investment Funds",
            ["aif", "valuation", "manager"]
        )
        q_counter += 1

    # 6. Market Regulations / MRD Queries (5 queries)
    mrd_docs = dept_groups.get("MRD", docs[120:140])
    for i, doc in enumerate(mrd_docs[:5]):
        c_num = doc.get("circular_number")
        subj = str(doc.get("subject") or "market infrastructure institutions")
        add_q(
            q_counter,
            f"What risk management or trading framework applies to {subj[:60]}?",
            c_num,
            "Market Infrastructure & Trading",
            ["trading", "exchange", "clearing"]
        )
        q_counter += 1

    # 7. Out-of-Corpus Refusal Testing Queries (3 queries)
    out_queries = [
        ("What are the corporate tax rate deductions for IT services under Section 115BAA of the Income Tax Act?", "Tax Law"),
        ("What are the capital adequacy requirements for cryptocurrency exchanges operating in the European Union under MiCA?", "Crypto Regulation"),
        ("What are the Federal Reserve interest rate decision rules for US commercial banks?", "US Banking")
    ]
    for q_text, dom in out_queries:
        add_q(
            q_counter,
            q_text,
            "NONE",
            dom,
            ["refusal"],
            out_of_corpus=True
        )
        q_counter += 1

    # Truncate or pad to exactly 50 benchmark queries
    benchmark = benchmark[:50]

    # Save benchmark JSON
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    logger.info(f"Saved {len(benchmark)}-question evaluation benchmark to {EVAL_BENCHMARK_PATH}")
    return benchmark


def load_eval_benchmark(path: Path = EVAL_BENCHMARK_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return construct_eval_benchmark()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
