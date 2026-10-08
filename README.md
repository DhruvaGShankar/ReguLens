# ReguLens: Citation-Grounded Financial Regulatory Intelligence using Hybrid Retrieval-Augmented Generation

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](LICENSE)
[![Framework: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Model: Nemotron Nano 9B v2](https://img.shields.io/badge/LLM-Nemotron--Nano--9B--v2-green.svg)](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)

ReguLens is an advanced Retrieval-Augmented Generation (RAG) system engineered to retrieve, explain, and cite provisions from a curated corpus of Securities and Exchange Board of India (SEBI) regulatory circulars.

---

## Table of Contents
1. [What ReguLens Is](#1-what-regulens-is)
2. [Why the Problem Matters](#2-why-the-problem-matters)
3. [Dataset](#3-dataset)
4. [Data Provenance](#4-data-provenance)
5. [System Architecture](#5-system-architecture)
6. [Section-Aware Chunking](#6-section-aware-chunking)
7. [Embedding Generation](#7-embedding-generation)
8. [Dense Retrieval (FAISS)](#8-dense-retrieval-faiss)
9. [Lexical Retrieval (BM25)](#9-lexical-retrieval-bm25)
10. [Hybrid Retrieval & Score Fusion](#10-hybrid-retrieval--score-fusion)
11. [Cross-Encoder Reranking](#11-cross-encoder-reranking)
12. [RAG Generation (Nemotron Nano 9B v2)](#12-rag-generation-nemotron-nano-9b-v2)
13. [Citation & Grounding Mechanism](#13-citation--grounding-mechanism)
14. [Evaluation Benchmark](#14-evaluation-benchmark)
15. [Experimental Results](#15-experimental-results)
16. [Installation & Setup](#16-installation--setup)
17. [Usage Guide](#17-usage-guide)
18. [Example Demonstration Queries](#18-example-demonstration-queries)
19. [Limitations](#19-limitations)
20. [Future Work](#20-future-work)

---

## 1. What ReguLens Is
ReguLens is an academic NLP and Information Retrieval (IR) research system designed for domain-specific financial regulation question answering. It combines semantic dense retrieval (`all-MiniLM-L6-v2` + FAISS), lexical retrieval (`rank-bm25`), min-max score fusion, cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`), and NVIDIA Nemotron Nano 9B v2 LLM generation to produce verifiable, citation-grounded regulatory answers.

---

## 2. Why the Problem Matters
Financial regulations issued by SEBI are distributed across thousands of lengthy circulars, notifications, and master circulars. Compliance officers, legal analysts, and researchers face several hurdles:
* Standard keyword search misses semantic synonyms (e.g., "borrowing limit" vs. "temporary liquidity requirement").
* Plain dense vector search struggles with exact regulatory identifiers (e.g., `SEBI/HO/CFD/2023/123`, `Regulation 30`, `15%`).
* Standard LLMs hallucinate non-existent circular numbers and section numbers when answering without strict evidence grounding.

ReguLens solves these challenges by enforcing **evidence before generation** with strict citation traceability.

---

## 3. Dataset
* **Source Dataset:** [`opnsrcntrbtrian/sebi-circulars`](https://huggingface.co/datasets/opnsrcntrbtrian/sebi-circulars) (Corpus Config).
* **Selected Sub-Corpus:** Exactly **300 circulars** reproducibly sampled using random seed `42`.
* **Total Generated Chunks:** 29,234 section-aware chunks.

---

## 4. Data Provenance
ReguLens preserves data provenance throughout the ingestion pipeline:
* **Original Regulatory Authority:** Securities and Exchange Board of India (SEBI).
* **Dataset Maintainer:** `opnsrcntrbtrian/sebi-circulars` on Hugging Face Datasets.
* **Metadata Fields Preserved:** `circular_number`, `issue_date`, `effective_date`, `subject`, `issuing_department`, `circular_type`, `supersession_status`, `validity_status`, `source_url`.

---

## 5. System Architecture

```text
SEBI Circular Corpus (300 Docs)
        ↓
Section-Aware Chunking (29,234 Chunks)
        ↓
 ┌─────────────────┐
 │ Dense Retrieval │ → FAISS (all-MiniLM-L6-v2)
 └─────────────────┘
          +
 ┌─────────────────┐
 │ BM25 Retrieval  │ → rank-bm25 (Regulatory Tokenizer)
 └─────────────────┘
          ↓
     Hybrid Fusion (Alpha = 0.5)
          ↓
 Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
          ↓
 Top Evidence Chunks (Top 5)
          ↓
 Nemotron Nano 9B v2 (OpenRouter API)
          ↓
 Grounded Answer + Verifiable Citations
```

---

## 6. Section-Aware Chunking
Unlike naive token splitting, ReguLens uses structural section boundary detection:
* **Target Size:** ~500–800 tokens (~450 words).
* **Overlap:** ~75–150 tokens (~75 words).
* **Context Header Injection:** Every chunk is prepended with a contextual header:
  `[SEBI Circular: {circular_number} | Date: {issue_date} | Dept: {department} | Section: {section_title}]`

---

## 7. Embedding Generation
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
* **Vector Normalization:** $L_2$ normalization applied to enable inner product cosine similarity matching.

---

## 8. Dense Retrieval (FAISS)
* **Index Class:** `faiss.IndexFlatIP` (Exact inner-product search).
* **Persisted Artifacts:** `data/indexes/faiss.index` and `data/indexes/faiss_id_map.json`.

---

## 9. Lexical Retrieval (BM25)
* **Algorithm:** `rank-bm25` (BM25Okapi).
* **Custom Regulatory Tokenizer:** Preserves circular numbers (`sebi/ho/cfd/...`), percentages (`15%`), monetary values (`₹50 crore`), and section clause identifiers.

---

## 10. Hybrid Retrieval & Score Fusion
Combines dense and lexical candidates using min-max score normalization:
$$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{dense\_norm}} + (1 - \alpha) \cdot \text{Score}_{\text{bm25\_norm}}$$

---

## 11. Cross-Encoder Reranking
* **Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **Pipeline:** Evaluates candidate pairs `(query, chunk_text)` to re-order top 20 hybrid candidates into final top 5 evidence chunks.

---

## 12. RAG Generation (Nemotron Nano 9B v2)
* **LLM Engine:** `nvidia/nemotron-nano-9b-v2` via OpenAI-compatible endpoint.
* **Strict Evidence System Prompt:** Enforces zero-hallucination rules, direct context referencing, and mandatory refusal when evidence is lacking.

---

## 13. Citation & Grounding Mechanism
Every claim is traceable to retrieved evidence with metadata containing:
* Circular Title / Subject
* Official SEBI Circular Number
* Publication Date & Issuing Department
* Section Heading
* Direct Link to Official SEBI Webpage

---

## 14. Evaluation Benchmark
Curated 50-question regulatory QA benchmark covering:
* Listed Entities & CFD Disclosure
* Mutual Funds & IMD Borrowing Rules
* Stock Brokers & MIRSD KYC Regulations
* Debt Securities & DDHS Issuance
* Alternative Investment Funds & Valuation
* Out-of-Corpus Refusal Queries

---

## 15. Experimental Results

| Metric | System A (Dense Baseline RAG) | System B (Improved Hybrid + Reranker) |
|---|---|---|
| **Recall@1** | 40.43% | **46.81%** |
| **Recall@5** | 65.96% | **89.36%** |
| **Recall@10** | 74.47% | **93.62%** |
| **MRR** | 0.5187 | **0.6217** |
| **Answer Accuracy** | 95.74% | **97.87%** |
| **Groundedness** | 98.00% | **100.00%** |
| **Refusal Accuracy**| 100.00% | **100.00%** |

---

## 16. Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/your-repo/ReguLens.git
cd ReguLens

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables (.env)
cp .env.example .env
```

---

## 17. Usage Guide

```bash
# Step 1: Download raw SEBI dataset
python scripts/download_data.py

# Step 2: Sample 300-document corpus
python scripts/build_corpus.py

# Step 3: Generate section-aware chunks
python scripts/build_chunks.py

# Step 4: Build FAISS & BM25 indexes
python scripts/build_index.py

# Step 5: Run full evaluation experiment
python scripts/run_evaluation.py

# Step 6: Launch Streamlit Web UI
streamlit run app/streamlit_app.py
```

---

## 18. Example Demonstration Queries
1. *"What restrictions apply to borrowing by mutual funds?"*
2. *"What disclosure requirements apply to research analysts under Regulation 30?"*
3. *"What are the valuation norms for Alternative Investment Funds (AIF)?"*
4. *"What capital rules apply to cryptocurrency exchanges operating in the EU under MiCA?"* *(Triggers explicit refusal response)*

---

## 19. Limitations
* Fixed 300-document sub-corpus representation.
* English language circular text only.
* Single-turn question answering (does not maintain conversational multi-turn chat history).

---

## 20. Future Work
* Temporal and supersession-aware regulation tracking.
* Query decomposition for complex multi-part regulatory inquiries.
* Regulatory knowledge graph integration.

---

## License
Distributed under the Creative Commons Attribution 4.0 International (CC-BY-4.0) License.
