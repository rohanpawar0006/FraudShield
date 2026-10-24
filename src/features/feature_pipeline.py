"""Point-in-time safe feature engineering pipeline for FraudShield.

Computes chronological transaction frequency, amount ratios, and history features
strictly using past information prior to each scored event.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "orig_txn_count_1h",
    "orig_txn_count_24h",
    "orig_amount_vs_mean",
    "orig_time_since_last_txn",
    "orig_has_prev_txn",
    "dest_txn_count_24h",
    "dest_has_prev_txn",
]

BASE_TIMESTAMP = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def step_to_timestamp(steps: pd.Series) -> pd.Series:
    """Convert integer step (hours from start) into deterministic UTC datetime."""
    return BASE_TIMESTAMP + pd.to_timedelta(steps, unit="h")


def compute_origin_features(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute point-in-time features for origin accounts (nameOrig).

    Features computed (strictly prior to current transaction):
    - orig_txn_count_1h: count of txns in preceding 1 hour.
    - orig_txn_count_24h: count of txns in preceding 24 hours.
    - orig_amount_vs_mean: amount / historical_mean (1.0 for first txn).
    - orig_time_since_last_txn: step - prev_step in hours (-1.0 for first txn).
    - orig_has_prev_txn: 1 if account seen previously, 0 otherwise.
    """
    n = len(df)
    orig_txn_count_1h = np.zeros(n, dtype=np.int64)
    orig_txn_count_24h = np.zeros(n, dtype=np.int64)
    orig_time_since_last = np.full(n, -1.0, dtype=np.float64)
    orig_has_prev = np.zeros(n, dtype=np.int64)
    orig_amount_vs_mean = np.ones(n, dtype=np.float64)

    # In PaySim, 99.8% of nameOrig accounts appear only once.
    # We identify repeated origin accounts to process only those with history.
    dup_orig_mask = df.duplicated(subset=["nameOrig"], keep=False)
    dup_indices = np.where(dup_orig_mask)[0]

    if len(dup_indices) > 0:
        sub_df = df.iloc[dup_indices][["nameOrig", "step", "amount"]]
        for _, group in sub_df.groupby("nameOrig"):
            idxs = group.index.values
            steps = group["step"].values
            amounts = group["amount"].values
            k = len(steps)

            cumulative_amount = 0.0
            for i in range(k):
                if i > 0:
                    orig_has_prev[idxs[i]] = 1
                    orig_time_since_last[idxs[i]] = float(steps[i] - steps[i - 1])

                    prior_mean = cumulative_amount / i
                    orig_amount_vs_mean[idxs[i]] = float(
                        amounts[i] / (prior_mean + 1e-5)
                    )

                    # Rolling 1h searchsorted (steps strictly prior in [step-1, step])
                    idx_1h = np.searchsorted(steps[:i], steps[i] - 1, side="left")
                    orig_txn_count_1h[idxs[i]] = int(i - idx_1h)

                    # Rolling 24h searchsorted (steps strictly prior in [step-24, step])
                    idx_24h = np.searchsorted(steps[:i], steps[i] - 24, side="left")
                    orig_txn_count_24h[idxs[i]] = int(i - idx_24h)

                cumulative_amount += amounts[i]

    return (
        orig_txn_count_1h,
        orig_txn_count_24h,
        orig_amount_vs_mean,
        orig_time_since_last,
        orig_has_prev,
    )


def compute_destination_features(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute point-in-time features for destination accounts (nameDest).

    Features computed (strictly prior to current transaction):
    - dest_txn_count_24h: count of inbound txns to nameDest in preceding 24h.
    - dest_has_prev_txn: 1 if destination seen previously, 0 otherwise.
    """
    n = len(df)
    dest_txn_count_24h = np.zeros(n, dtype=np.int64)
    dest_has_prev = np.zeros(n, dtype=np.int64)

    dup_dest_mask = df.duplicated(subset=["nameDest"], keep=False)
    dup_indices = np.where(dup_dest_mask)[0]

    if len(dup_indices) > 0:
        sub_df = df.iloc[dup_indices][["nameDest", "step"]]
        for _, group in sub_df.groupby("nameDest"):
            idxs = group.index.values
            steps = group["step"].values
            k = len(steps)

            for i in range(1, k):
                dest_has_prev[idxs[i]] = 1
                idx_24h = np.searchsorted(steps[:i], steps[i] - 24, side="left")
                dest_txn_count_24h[idxs[i]] = int(i - idx_24h)

    return dest_txn_count_24h, dest_has_prev


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all point-in-time features on a chronologically sorted DataFrame.

    Args:
        df: DataFrame containing step, amount, nameOrig, nameDest, etc.

    Returns:
        DataFrame augmented with engineered feature columns and event_timestamp.
    """
    logger.info(f"Computing point-in-time features for {len(df):,} records...")

    # Ensure chronological order
    if not df["step"].is_monotonic_increasing:
        logger.warning("Dataset not sorted by step. Sorting chronologically...")
        df = df.sort_values(by="step", kind="stable").reset_index(drop=True)
    else:
        df = df.copy()

    # 1. Origin account features
    (
        orig_count_1h,
        orig_count_24h,
        orig_amt_vs_mean,
        orig_time_since,
        orig_has_prev,
    ) = compute_origin_features(df)

    df["orig_txn_count_1h"] = orig_count_1h
    df["orig_txn_count_24h"] = orig_count_24h
    df["orig_amount_vs_mean"] = orig_amt_vs_mean
    df["orig_time_since_last_txn"] = orig_time_since
    df["orig_has_prev_txn"] = orig_has_prev

    # 2. Destination account features
    dest_count_24h, dest_has_prev = compute_destination_features(df)
    df["dest_txn_count_24h"] = dest_count_24h
    df["dest_has_prev_txn"] = dest_has_prev

    # 3. Deterministic UTC timestamp for Feast
    df["event_timestamp"] = step_to_timestamp(df["step"])

    logger.info("Feature engineering completed successfully.")
    return df


def get_feature_names() -> List[str]:
    """Return list of engineered feature column names."""
    return list(FEATURE_COLUMNS)


def process_and_save_splits(
    processed_dir: Path = Path("data/processed"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test parquet, compute continuous features, and save augmented parquet."""
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"

    logger.info("Loading existing time-based Parquet splits...")
    train_raw = pd.read_parquet(train_path)
    val_raw = pd.read_parquet(val_path)
    test_raw = pd.read_parquet(test_path)

    # Track split boundaries
    n_train = len(train_raw)
    n_val = len(val_raw)
    n_test = len(test_raw)

    logger.info(
        f"Concatenating {n_train:,} train, {n_val:,} val, {n_test:,} test rows "
        "to ensure seamless, continuous historical state without temporal boundary blindness..."
    )
    full_df = pd.concat([train_raw, val_raw, test_raw], ignore_index=True)

    # Compute point-in-time features on full chronological stream
    full_features = compute_features(full_df)

    # Partition back into exact original splits
    train_features = full_features.iloc[:n_train].copy()
    val_features = full_features.iloc[n_train : n_train + n_val].copy()
    test_features = full_features.iloc[n_train + n_val :].copy()

    # Save to parquet
    out_train = processed_dir / "train_features.parquet"
    out_val = processed_dir / "val_features.parquet"
    out_test = processed_dir / "test_features.parquet"

    logger.info(f"Writing {out_train}...")
    train_features.to_parquet(
        out_train, index=False, engine="pyarrow", compression="snappy"
    )

    logger.info(f"Writing {out_val}...")
    val_features.to_parquet(
        out_val, index=False, engine="pyarrow", compression="snappy"
    )

    logger.info(f"Writing {out_test}...")
    test_features.to_parquet(
        out_test, index=False, engine="pyarrow", compression="snappy"
    )

    logger.info("All feature splits saved successfully.")
    return train_features, val_features, test_features


if __name__ == "__main__":
    process_and_save_splits()
