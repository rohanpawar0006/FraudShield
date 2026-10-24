"""Unit tests for time-based dataset splitting and zero-overlap guarantees."""

from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"


@pytest.fixture(scope="module")
def split_datasets():
    """Load train, val, and test split parquet files."""
    if not (TRAIN_PATH.exists() and VAL_PATH.exists() and TEST_PATH.exists()):
        pytest.skip("Processed parquet splits not generated yet.")

    train_df = pd.read_parquet(TRAIN_PATH, columns=["step", "isFraud"])
    val_df = pd.read_parquet(VAL_PATH, columns=["step", "isFraud"])
    test_df = pd.read_parquet(TEST_PATH, columns=["step", "isFraud"])
    return train_df, val_df, test_df


def test_splits_exist_and_non_empty(split_datasets) -> None:
    """Verify all split files exist and have records."""
    train_df, val_df, test_df = split_datasets
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_splits_strict_time_ordering(split_datasets) -> None:
    """CRITICAL: Assert zero time overlap between train, val, and test splits."""
    train_df, val_df, test_df = split_datasets

    train_max_step = train_df["step"].max()
    val_min_step = val_df["step"].min()
    val_max_step = val_df["step"].max()
    test_min_step = test_df["step"].min()

    # Train must be strictly earlier than validation
    assert train_max_step < val_min_step, (
        f"Temporal leakage: train_max ({train_max_step}) >= val_min ({val_min_step})"
    )

    # Validation must be strictly earlier than test
    assert val_max_step < test_min_step, (
        f"Temporal leakage: val_max ({val_max_step}) >= test_min ({test_min_step})"
    )


def test_fraud_rates_in_splits(split_datasets) -> None:
    """Verify fraud is present in all splits and respects < 1% imbalance expectation."""
    train_df, val_df, test_df = split_datasets

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        fraud_count = df["isFraud"].sum()
        fraud_rate = df["isFraud"].mean()

        assert fraud_count > 0, f"{name} split contains zero fraud examples!"
        assert fraud_rate < 0.01, f"{name} fraud rate ({fraud_rate:.4f}) exceeds 1%!"
