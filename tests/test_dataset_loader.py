import pytest
import pandas as pd
import numpy as np
from src import config
from src.ingestion.dataset_loader import sample_stratified_corpus


@pytest.fixture
def dummy_dataset():
    """Create a mock dataset of 500 rows to test stratification sampling."""
    rows = []
    departments = ["CFD", "MRD", "IMD", "UNKNOWN", "SEBI"]
    types = ["CIRCULAR", "MASTER_CIRCULAR"]
    years = [2018, 2020, 2022, 2024]

    for i in range(500):
        dept = departments[i % len(departments)]
        c_type = types[i % len(types)]
        year = years[i % len(years)]
        rows.append({
            "circular_number": f"SEBI/HO/TEST/{i:04d}",
            "issue_date": f"{year}-01-15",
            "effective_date": None,
            "subject": f"Test circular subject {i}",
            "issuing_department": dept,
            "supersession_status": "in_force",
            "source_url": f"https://www.sebi.gov.in/test_{i}.html",
            "text": f"This is test content for circular number {i}.",
            "circular_type": c_type,
            "validity_status": "current"
        })
    return pd.DataFrame(rows)


def test_sampling_target_size(dummy_dataset):
    sampled = sample_stratified_corpus(dummy_dataset, target_size=300, seed=42)
    assert len(sampled) == 300
    assert "circular_number" in sampled.columns
    assert "subject" in sampled.columns


def test_sampling_reproducibility(dummy_dataset):
    sample1 = sample_stratified_corpus(dummy_dataset, target_size=300, seed=42)
    sample2 = sample_stratified_corpus(dummy_dataset, target_size=300, seed=42)
    pd.testing.assert_frame_equal(sample1, sample2)


def test_unknown_department_retention(dummy_dataset):
    sampled = sample_stratified_corpus(dummy_dataset, target_size=300, seed=42)
    unknown_count = (sampled["issuing_department"] == "UNKNOWN").sum()
    assert unknown_count > 0, "UNKNOWN department records should be explicitly retained in sample"
