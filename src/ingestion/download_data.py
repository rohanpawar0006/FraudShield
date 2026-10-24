"""Dataset management and synthetic data generator for FraudShield.

Supports downloading the PaySim dataset or generating a representative
synthetic dataset matching PaySim schema for development, testing, and CI.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

# PaySim Schema columns:
# step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
# nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
PAYSIM_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]


def generate_synthetic_paysim(
    num_records: int = 10_000,
    fraud_rate: float = 0.005,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate synthetic transactions adhering to PaySim schema.

    Args:
        num_records: Total number of transactions to generate.
        fraud_rate: Ratio of fraudulent transactions (default ~0.5%).
        seed: Random seed for reproducibility.
        output_path: Optional file path to save parquet/csv.

    Returns:
        DataFrame containing synthetic transaction data.
    """
    logger.info(f"Generating {num_records} synthetic transactions (seed={seed})...")
    rng = np.random.default_rng(seed)

    # Time steps (representing hours, 1 to 744 ~ 30 days)
    steps = np.sort(rng.integers(1, 100, size=num_records))

    # Types
    types = rng.choice(
        TRANSACTION_TYPES,
        size=num_records,
        p=[0.35, 0.20, 0.35, 0.05, 0.05],
    )

    # Amounts: log-normal distribution to mirror financial skew
    amounts = np.round(rng.lognormal(mean=7.0, sigma=1.8, size=num_records), 2)
    amounts = np.clip(amounts, 1.0, 5_000_000.0)

    # Account IDs (pool of origin and destination accounts to simulate rings)
    num_orig_accounts = max(100, num_records // 5)
    num_dest_accounts = max(100, num_records // 6)
    orig_pool = [f"C{i:09d}" for i in range(1, num_orig_accounts + 1)]
    dest_pool = [
        f"M{i:09d}" if i % 2 == 0 else f"C{i:09d}"
        for i in range(1, num_dest_accounts + 1)
    ]

    name_orig = rng.choice(orig_pool, size=num_records)
    name_dest = rng.choice(dest_pool, size=num_records)

    # Balances
    old_bal_orig = np.round(rng.uniform(0, 100_000, size=num_records), 2)
    new_bal_orig = np.maximum(0.0, old_bal_orig - amounts)

    old_bal_dest = np.round(rng.uniform(0, 200_000, size=num_records), 2)
    new_bal_dest = old_bal_dest + amounts

    # Fraud label generation (disproportionately in TRANSFER and CASH_OUT)
    num_frauds = int(num_records * fraud_rate)
    is_fraud = np.zeros(num_records, dtype=int)
    fraud_eligible_indices = np.where(np.isin(types, ["TRANSFER", "CASH_OUT"]))[0]

    if len(fraud_eligible_indices) >= num_frauds:
        fraud_idx = rng.choice(fraud_eligible_indices, size=num_frauds, replace=False)
        is_fraud[fraud_idx] = 1
        # Fraud often drains the account
        new_bal_orig[fraud_idx] = 0.0

    is_flagged_fraud = np.zeros(num_records, dtype=int)
    # Rule in PaySim: flag transactions > 200,000
    flagged_idx = np.where((amounts > 200_000) & (types == "TRANSFER"))[0]
    if len(flagged_idx) > 0:
        is_flagged_fraud[flagged_idx] = 1

    df = pd.DataFrame(
        {
            "step": steps,
            "type": types,
            "amount": amounts,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_bal_orig,
            "newbalanceOrig": new_bal_orig,
            "nameDest": name_dest,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged_fraud,
        }
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if str(output_path).endswith(".parquet"):
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to {output_path} ({len(df)} records)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate PaySim dataset")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic dataset for testing",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50_000,
        help="Number of synthetic records to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/paysim_synthetic.parquet",
        help="Output file path",
    )
    args = parser.parse_args()

    out_file = Path(args.output)
    generate_synthetic_paysim(
        num_records=args.count,
        output_path=out_file,
    )
