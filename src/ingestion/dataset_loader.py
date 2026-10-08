import logging
from pathlib import Path
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_raw_dataset(repo_id: str = config.HF_DATASET_ID, output_path: Path = config.RAW_CORPUS_PARQUET) -> Path:
    """
    Download raw SEBI circulars corpus parquet from HuggingFace dataset repository.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading raw dataset '{repo_id}' corpus config from HuggingFace...")
    downloaded_file = hf_hub_download(
        repo_id=repo_id,
        filename="corpus/corpus.parquet",
        repo_type="dataset"
    )
    df = pd.read_parquet(downloaded_file)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved raw dataset ({len(df)} records) to {output_path}")
    return output_path


def load_raw_dataset(path: Path = config.RAW_CORPUS_PARQUET) -> pd.DataFrame:
    """
    Load raw SEBI circular dataset from local disk, downloading if not present.
    """
    if not path.exists():
        download_raw_dataset(output_path=path)
    df = pd.read_parquet(path)
    logger.info(f"Loaded raw dataset with {len(df)} records from {path}")
    return df


def sample_stratified_corpus(
    df: pd.DataFrame,
    target_size: int = config.TARGET_CORPUS_SIZE,
    seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """
    Reproducibly select a stratified subset of SEBI circulars based on:
    - Publication Year
    - Issuing Department (explicitly tracking 'UNKNOWN')
    - Circular Type

    Returns a DataFrame containing exactly target_size records.
    """
    if len(df) <= target_size:
        logger.warning(f"Raw dataset size ({len(df)}) <= target ({target_size}). Returning full dataset.")
        return df.copy()

    df_clean = df.copy()

    # Extract issue_year from issue_date
    df_clean["issue_date_dt"] = pd.to_datetime(df_clean["issue_date"], errors="coerce")
    df_clean["issue_year"] = df_clean["issue_date_dt"].dt.year.fillna(2020).astype(int)

    # Standardize issuing_department (preserving UNKNOWN)
    df_clean["dept_clean"] = df_clean["issuing_department"].fillna("UNKNOWN").astype(str)
    df_clean["type_clean"] = df_clean["circular_type"].fillna("CIRCULAR").astype(str)

    # Create composite stratification key
    df_clean["strat_key"] = (
        df_clean["issue_year"].astype(str) + "_" +
        df_clean["dept_clean"] + "_" +
        df_clean["type_clean"]
    )

    # Calculate proportional target count per stratum
    frac = target_size / len(df_clean)
    
    sampled_indices = []
    np.random.seed(seed)

    # Group sampling
    grouped = df_clean.groupby("strat_key")
    for group_key, group in grouped:
        n_sample = int(np.round(len(group) * frac))
        if n_sample > 0:
            sample_ids = group.sample(n=min(n_sample, len(group)), random_state=seed).index.tolist()
            sampled_indices.extend(sample_ids)

    # Adjust if rounding led to slightly more or fewer than target_size
    sampled_indices = list(set(sampled_indices))
    
    if len(sampled_indices) < target_size:
        remaining_idx = [i for i in df_clean.index if i not in sampled_indices]
        needed = target_size - len(sampled_indices)
        rng = np.random.RandomState(seed)
        additional = rng.choice(remaining_idx, size=needed, replace=False).tolist()
        sampled_indices.extend(additional)
    elif len(sampled_indices) > target_size:
        rng = np.random.RandomState(seed)
        sampled_indices = rng.choice(sampled_indices, size=target_size, replace=False).tolist()

    sampled_df = df_clean.loc[sampled_indices].copy()
    
    # Sort deterministically by circular_number / issue_date
    sampled_df.sort_values(by=["issue_date", "circular_number"], inplace=True)
    sampled_df.reset_index(drop=True, inplace=True)

    # Clean up temporary helper columns
    sampled_df.drop(columns=["issue_date_dt", "issue_year", "dept_clean", "type_clean", "strat_key"], inplace=True)

    logger.info(f"Sampled {len(sampled_df)} circulars reproducibly with seed={seed}.")
    return sampled_df
