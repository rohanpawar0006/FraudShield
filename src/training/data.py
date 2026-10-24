"""Data loading, leakage verification, and preprocessor builders for model training."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.logging import get_logger

logger = get_logger(__name__)

# Transaction types present in PaySim
PAYMENT_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

CATEGORICAL_FEATURES: list[str] = ["type"]

NUMERICAL_FEATURES: list[str] = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "orig_txn_count_1h",
    "orig_txn_count_24h",
    "orig_amount_vs_mean",
    "orig_time_since_last_txn",
    "orig_has_prev_txn",
    "dest_txn_count_24h",
    "dest_has_prev_txn",
]

ALL_FEATURE_COLUMNS: list[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

TARGET_COLUMN: str = "isFraud"

LEAKAGE_COLUMNS: list[str] = [
    "newbalanceOrig",
    "newbalanceDest",
    "isFlaggedFraud",
]

IDENTIFIER_COLUMNS: list[str] = [
    "nameOrig",
    "nameDest",
]

METADATA_COLUMNS: list[str] = [
    "step",
    "event_timestamp",
]

FORBIDDEN_COLUMNS: list[str] = (
    LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS + METADATA_COLUMNS
)


def validate_split_integrity(df: pd.DataFrame, split_name: str) -> None:
    """Validate data integrity, completeness, and absence of target/leakage in features.

    Args:
        df: DataFrame loaded from split parquet.
        split_name: Name of split (train, val, test) for logging.

    Raises:
        ValueError: If integrity checks, types, or non-finite values fail.
    """
    logger.info("Validating dataset integrity for split: %s", split_name)

    # 1. Target presence
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' missing in split '{split_name}'"
        )

    # 2. Required features presence
    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Required feature '{col}' missing in split '{split_name}'"
            )

    # 3. Missing values check
    null_counts = df[ALL_FEATURE_COLUMNS].isnull().sum()
    if null_counts.any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"Null values detected in split '{split_name}': {bad_cols}"
        )

    # 4. Finite values check for numerical features
    for col in NUMERICAL_FEATURES:
        if not np.isfinite(df[col]).all():
            raise ValueError(
                f"Non-finite values detected in numeric column '{col}' for split '{split_name}'"
            )

    logger.info(
        "Integrity check passed for '%s' (%d rows, %d fraud cases)",
        split_name,
        len(df),
        int(df[TARGET_COLUMN].sum()),
    )


def load_feature_split(
    split_name: str, data_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a feature split from Parquet with strict leakage and identity isolation.

    Args:
        split_name: Split name ('train', 'val', or 'test').
        data_dir: Path to directory containing processed feature parquets.

    Returns:
        Tuple of (X, y) where X is DataFrame with only legitimate pre-transaction features,
        and y is target Series.

    Raises:
        FileNotFoundError: If parquet file does not exist.
        ValueError: If any forbidden column is found in X.
    """
    path = Path(data_dir) / f"{split_name}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature split file not found: {path}")

    logger.info("Loading feature split from %s", path)
    # Memory optimization: Load only predictor columns and target, omitting unused identifier/leakage columns
    cols_to_load = list(dict.fromkeys(ALL_FEATURE_COLUMNS + [TARGET_COLUMN]))
    df = pd.read_parquet(path, columns=cols_to_load)

    # Validate dataset before feature extraction
    validate_split_integrity(df, split_name)

    # Extract target
    y = df[TARGET_COLUMN].astype(np.int64)

    # Extract feature matrix X
    x_df = df[ALL_FEATURE_COLUMNS].copy()

    # Zero-leakage verification: assert NO forbidden columns in X
    for col in FORBIDDEN_COLUMNS:
        if col in x_df.columns:
            raise ValueError(
                f"CRITICAL LEAKAGE ERROR: Forbidden column '{col}' present in model matrix X!"
            )

    return x_df, y


def create_preprocessor(scale_numerical: bool = True) -> ColumnTransformer:
    """Create a ColumnTransformer for feature encoding and optional scaling.

    Args:
        scale_numerical: If True, scale numerical features with StandardScaler.
                         If False, pass numerical features through unscaled.

    Returns:
        Unfitted ColumnTransformer.
    """
    num_transformer = StandardScaler() if scale_numerical else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    categories=[PAYMENT_TYPES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", num_transformer, NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Get output feature names from a fitted ColumnTransformer."""
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_names = [f"type_{cat}" for cat in cat_encoder.categories_[0]]
    return cat_names + NUMERICAL_FEATURES
