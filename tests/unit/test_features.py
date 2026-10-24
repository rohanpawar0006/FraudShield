"""Unit tests for point-in-time safe feature engineering.

Validates rolling windows, historical means, unseen account handling,
duplicate timestamps, and strict temporal causality without future leakage.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.features.feature_pipeline import (
    compute_features,
    get_feature_names,
)

PROCESSED_DIR = Path("data/processed")


@pytest.fixture
def synthetic_stream() -> pd.DataFrame:
    """Deterministic synthetic transaction stream with known history patterns."""
    return pd.DataFrame(
        {
            "step": [1, 1, 2, 2, 25, 26, 27],
            "nameOrig": [
                "C_ALICE",
                "C_BOB",
                "C_ALICE",
                "C_ALICE",
                "C_ALICE",
                "C_BOB",
                "C_CHARLIE",
            ],
            "nameDest": [
                "M_AMAZON",
                "M_WALMART",
                "M_AMAZON",
                "M_UBER",
                "M_AMAZON",
                "M_AMAZON",
                "M_NETFLIX",
            ],
            "amount": [100.0, 50.0, 200.0, 300.0, 150.0, 80.0, 10.0],
            "isFraud": [0, 0, 0, 0, 0, 0, 0],
        }
    )


def test_unseen_account_cold_start(synthetic_stream) -> None:
    """Verify first-time seen accounts produce deterministic cold-start values."""
    feats = compute_features(synthetic_stream)

    # First transaction for C_ALICE (row 0)
    alice_row0 = feats.iloc[0]
    assert alice_row0["orig_has_prev_txn"] == 0
    assert alice_row0["orig_txn_count_1h"] == 0
    assert alice_row0["orig_txn_count_24h"] == 0
    assert alice_row0["orig_time_since_last_txn"] == -1.0
    assert alice_row0["orig_amount_vs_mean"] == 1.0

    # First transaction for unseen C_CHARLIE (row 6)
    charlie_row = feats.iloc[6]
    assert charlie_row["orig_has_prev_txn"] == 0
    assert charlie_row["orig_time_since_last_txn"] == -1.0
    assert charlie_row["orig_amount_vs_mean"] == 1.0


def test_historical_mean_calculation(synthetic_stream) -> None:
    """Verify amount_vs_mean is calculated strictly using prior transactions."""
    feats = compute_features(synthetic_stream)

    # Row 0: C_ALICE amount=100.0 -> first txn -> 1.0
    assert feats.iloc[0]["orig_amount_vs_mean"] == pytest.approx(1.0)

    # Row 2: C_ALICE amount=200.0, prior mean = 100.0 -> ratio = 2.0
    assert feats.iloc[2]["orig_amount_vs_mean"] == pytest.approx(2.0, rel=1e-4)

    # Row 3: C_ALICE amount=300.0, prior amounts [100, 200] -> prior mean = 150.0 -> ratio = 300/150 = 2.0
    assert feats.iloc[3]["orig_amount_vs_mean"] == pytest.approx(2.0, rel=1e-4)

    # Row 4: C_ALICE amount=150.0, prior amounts [100, 200, 300] -> prior mean = 200.0 -> ratio = 150/200 = 0.75
    assert feats.iloc[4]["orig_amount_vs_mean"] == pytest.approx(0.75, rel=1e-4)


def test_rolling_window_boundaries(synthetic_stream) -> None:
    """Verify rolling 1h and 24h windows strictly exclude future or expired events."""
    feats = compute_features(synthetic_stream)

    # Row 2: C_ALICE at step 2 (prior at step 1: exactly 1h earlier)
    # In window [1, 2]: step 1 is included in 1h and 24h
    assert feats.iloc[2]["orig_txn_count_1h"] == 1
    assert feats.iloc[2]["orig_txn_count_24h"] == 1

    # Row 4: C_ALICE at step 25 (prior steps: 1, 2, 2)
    # 1h window [24, 25]: steps 1 and 2 are > 1h ago -> count_1h == 0
    # 24h window [1, 25]: step 1 (25-1=24h), step 2, step 2 are in window -> count_24h == 3
    assert feats.iloc[4]["orig_txn_count_1h"] == 0
    assert feats.iloc[4]["orig_txn_count_24h"] == 3


def test_destination_features(synthetic_stream) -> None:
    """Verify destination account history and 24h inbound frequency."""
    feats = compute_features(synthetic_stream)

    # Row 0: M_AMAZON first arrival -> has_prev == 0, count_24h == 0
    assert feats.iloc[0]["dest_has_prev_txn"] == 0
    assert feats.iloc[0]["dest_txn_count_24h"] == 0

    # Row 2: M_AMAZON second arrival at step 2 (prior at step 1) -> has_prev == 1, count_24h == 1
    assert feats.iloc[2]["dest_has_prev_txn"] == 1
    assert feats.iloc[2]["dest_txn_count_24h"] == 1

    # Row 4: M_AMAZON third arrival at step 25 (prior at step 1, 2) -> in [1, 25], both inside 24h
    assert feats.iloc[4]["dest_has_prev_txn"] == 1
    assert feats.iloc[4]["dest_txn_count_24h"] == 2

    # Row 5: M_AMAZON fourth arrival at step 26 (prior at step 1, 2, 25)
    # Window [2, 26]: step 1 is 25h ago (excluded), steps 2 and 25 are inside -> count_24h == 2
    assert feats.iloc[5]["dest_has_prev_txn"] == 1
    assert feats.iloc[5]["dest_txn_count_24h"] == 2


def test_duplicate_timestamps_deterministic(synthetic_stream) -> None:
    """Verify sequential ordering when multiple transactions share identical steps."""
    feats = compute_features(synthetic_stream)

    # Rows 2 and 3 both have step=2 for C_ALICE
    row2 = feats.iloc[2]
    row3 = feats.iloc[3]

    assert row2["orig_time_since_last_txn"] == 1.0  # step 2 - step 1
    assert row3["orig_time_since_last_txn"] == 0.0  # step 2 - step 2
    assert row3["orig_has_prev_txn"] == 1


def test_feature_schema_consistency() -> None:
    """Verify train, val, and test feature Parquet artifacts have identical schema."""
    train_path = PROCESSED_DIR / "train_features.parquet"
    val_path = PROCESSED_DIR / "val_features.parquet"
    test_path = PROCESSED_DIR / "test_features.parquet"

    assert train_path.exists(), "train_features.parquet missing"
    assert val_path.exists(), "val_features.parquet missing"
    assert test_path.exists(), "test_features.parquet missing"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    expected_cols = get_feature_names()
    for col in expected_cols:
        assert col in train_df.columns, f"Missing {col} in train_features"
        assert col in val_df.columns, f"Missing {col} in val_features"
        assert col in test_df.columns, f"Missing {col} in test_features"

    assert "event_timestamp" in train_df.columns
    assert "nameOrig" in train_df.columns
    assert "nameDest" in train_df.columns
    assert "isFraud" in train_df.columns
