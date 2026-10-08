import sys
import json
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.dataset_loader import load_raw_dataset, sample_stratified_corpus
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_doc_id(circular_number: str, idx: int) -> str:
    """
    Generate a deterministic document_id based on circular number index.
    """
    clean_num = str(circular_number).replace("/", "_").replace("-", "_").replace(" ", "_")
    return f"doc_{idx+1:04d}_{clean_num}"


def main():
    logger.info("Starting corpus selection script...")
    
    # 1. Load raw dataset
    df_raw = load_raw_dataset()
    logger.info(f"Loaded raw dataset with {len(df_raw)} records.")

    # 2. Sample 300 circulars reproducibly
    df_sampled = sample_stratified_corpus(
        df_raw,
        target_size=config.TARGET_CORPUS_SIZE,
        seed=config.RANDOM_SEED
    )

    # 3. Add deterministic document_id
    doc_ids = [generate_doc_id(row["circular_number"], i) for i, row in df_sampled.iterrows()]
    df_sampled.insert(0, "document_id", doc_ids)

    # 4. Save corpus manifest CSV
    manifest_cols = [
        "document_id",
        "circular_number",
        "subject",
        "issue_date",
        "effective_date",
        "issuing_department",
        "circular_type",
        "supersession_status",
        "validity_status",
        "source_url",
    ]
    manifest_df = df_sampled[manifest_cols].copy()
    manifest_df["dataset_identifier"] = config.HF_DATASET_ID
    manifest_df["selection_seed"] = config.RANDOM_SEED

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(config.CORPUS_MANIFEST_CSV, index=False)
    logger.info(f"Saved corpus manifest ({len(manifest_df)} records) to {config.CORPUS_MANIFEST_CSV}")

    # 5. Save corpus full documents JSON
    documents = df_sampled.to_dict(orient="records")
    with open(config.CORPUS_DOCUMENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, default=str)
    logger.info(f"Saved full corpus documents JSON to {config.CORPUS_DOCUMENTS_JSON}")

    # 6. Log detailed statistics
    logger.info("=== CORPUS SELECTION SUMMARY ===")
    logger.info(f"Total Selected Circulars: {len(df_sampled)}")
    
    dept_counts = df_sampled["issuing_department"].value_counts(dropna=False).to_dict()
    logger.info(f"Department Breakdown: {dept_counts}")
    
    df_sampled["issue_year"] = pd.to_datetime(df_sampled["issue_date"], errors="coerce").dt.year.fillna(2020).astype(int)
    year_counts = df_sampled["issue_year"].value_counts().sort_index().to_dict()
    logger.info(f"Publication Year Breakdown: {year_counts}")

    status_counts = df_sampled["supersession_status"].value_counts(dropna=False).to_dict()
    logger.info(f"Supersession Status Breakdown: {status_counts}")


if __name__ == "__main__":
    main()
