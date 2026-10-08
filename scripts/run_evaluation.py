import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluation.evaluate import run_comparative_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting evaluation experiment script...")
    exp_results = run_comparative_experiment()

    summary_df = pd.DataFrame(exp_results["summary_table"])
    
    print("\n" + "=" * 65)
    print("      REGULENS EXPERIMENTAL EVALUATION RESULTS SUMMARY      ")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
