import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present (override shell variables)
load_dotenv(override=True)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

# Dataset Configuration
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "opnsrcntrbtrian/sebi-circulars")
TARGET_CORPUS_SIZE = int(os.getenv("TARGET_CORPUS_SIZE", "300"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Dataset File Artifact Paths
RAW_CORPUS_PARQUET = RAW_DATA_DIR / "sebi_circulars_corpus.parquet"
CORPUS_MANIFEST_CSV = PROCESSED_DATA_DIR / "corpus_manifest.csv"
CORPUS_DOCUMENTS_JSON = PROCESSED_DATA_DIR / "corpus_documents.json"
CHUNKS_JSON = PROCESSED_DATA_DIR / "chunks.json"

# Index Artifact Paths
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
FAISS_ID_MAP_PATH = INDEX_DIR / "faiss_id_map.json"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

# LLM Configuration (NVIDIA Nemotron 3 Nano Omni)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

# Retrieval Models & Config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "10"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

# Chunking Parameters
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 100
