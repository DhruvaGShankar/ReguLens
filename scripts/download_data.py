import sys
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.dataset_loader import download_raw_dataset
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting dataset download script...")
    raw_path = download_raw_dataset(repo_id=config.HF_DATASET_ID, output_path=config.RAW_CORPUS_PARQUET)
    logger.info(f"Dataset successfully downloaded to: {raw_path}")

if __name__ == "__main__":
    main()
