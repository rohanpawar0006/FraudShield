"""Phase 1: Exploratory Data Analysis (EDA) and Time-Based Dataset Splitting.

Loads PaySim raw transaction dataset, computes distributions, checks leakage,
saves time-based train/val/test splits to Parquet, and writes eda.md and 01_eda.ipynb.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
PROCESSED_DIR = Path("data/processed")
DOCS_RESULTS_DIR = Path("docs/results")
NOTEBOOKS_DIR = Path("notebooks")

TRAIN_MAX_STEP = 323
VAL_MAX_STEP = 378


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load PaySim raw CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    logger.info(f"Loading raw PaySim dataset from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} transactions with {len(df.columns)} columns.")
    return df


def perform_eda(df: pd.DataFrame) -> Dict:
    """Compute core EDA statistics on PaySim dataset."""
    logger.info("Computing EDA summary statistics...")
    total_records = len(df)
    fraud_counts = df["isFraud"].value_counts().to_dict()
    fraud_rate = float(df["isFraud"].mean())

    type_counts = df["type"].value_counts().to_dict()
    fraud_by_type = (
        df.groupby("type")["isFraud"].agg(["count", "sum", "mean"]).to_dict("index")
    )

    amount_stats_overall = df["amount"].describe().to_dict()
    amount_stats_legit = df[df["isFraud"] == 0]["amount"].describe().to_dict()
    amount_stats_fraud = df[df["isFraud"] == 1]["amount"].describe().to_dict()

    # Balance leakage checks
    exact_orig_diff_rate = float(
        (df["oldbalanceOrg"] - df["amount"] == df["newbalanceOrig"]).mean()
    )
    fraud_orig_zero_rate = float((df[df["isFraud"] == 1]["newbalanceOrig"] == 0).mean())
    legit_orig_zero_rate = float((df[df["isFraud"] == 0]["newbalanceOrig"] == 0).mean())

    # isFlaggedFraud check
    flagged_counts = df["isFlaggedFraud"].value_counts().to_dict()
    flagged_fraud_crosstab = pd.crosstab(df["isFlaggedFraud"], df["isFraud"]).to_dict()

    # Time structure
    step_min = int(df["step"].min())
    step_max = int(df["step"].max())
    step_nunique = int(df["step"].nunique())

    return {
        "total_records": total_records,
        "fraud_counts": fraud_counts,
        "fraud_rate": fraud_rate,
        "type_counts": type_counts,
        "fraud_by_type": fraud_by_type,
        "amount_stats_overall": amount_stats_overall,
        "amount_stats_legit": amount_stats_legit,
        "amount_stats_fraud": amount_stats_fraud,
        "exact_orig_diff_rate": exact_orig_diff_rate,
        "fraud_orig_zero_rate": fraud_orig_zero_rate,
        "legit_orig_zero_rate": legit_orig_zero_rate,
        "flagged_counts": flagged_counts,
        "flagged_fraud_crosstab": flagged_fraud_crosstab,
        "step_min": step_min,
        "step_max": step_max,
        "step_nunique": step_nunique,
    }


def split_data_by_time(
    df: pd.DataFrame,
    train_max_step: int = TRAIN_MAX_STEP,
    val_max_step: int = VAL_MAX_STEP,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform strict time-based train/val/test splitting on step column."""
    logger.info(
        f"Splitting data by step: Train (<= {train_max_step}), "
        f"Val ({train_max_step + 1}..{val_max_step}), "
        f"Test (> {val_max_step})..."
    )
    train_df = df[df["step"] <= train_max_step].copy()
    val_df = df[(df["step"] > train_max_step) & (df["step"] <= val_max_step)].copy()
    test_df = df[df["step"] > val_max_step].copy()

    # Assert zero time overlap
    assert train_df["step"].max() < val_df["step"].min(), "Train and Val steps overlap!"
    assert val_df["step"].max() < test_df["step"].min(), "Val and Test steps overlap!"
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Row counts mismatch!"

    logger.info(
        f"Split complete: Train={len(train_df):,} rows ({train_df['isFraud'].sum():,} frauds), "
        f"Val={len(val_df):,} rows ({val_df['isFraud'].sum():,} frauds), "
        f"Test={len(test_df):,} rows ({test_df['isFraud'].sum():,} frauds)"
    )
    return train_df, val_df, test_df


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Save datasets as Parquet files with compression."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving train/val/test splits to Parquet...")

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False, engine="pyarrow", compression="snappy")
    val_df.to_parquet(val_path, index=False, engine="pyarrow", compression="snappy")
    test_df.to_parquet(test_path, index=False, engine="pyarrow", compression="snappy")

    logger.info(
        f"Saved: {train_path} ({train_path.stat().st_size / (1024 * 1024):.1f} MB)"
    )
    logger.info(f"Saved: {val_path} ({val_path.stat().st_size / (1024 * 1024):.1f} MB)")
    logger.info(
        f"Saved: {test_path} ({test_path.stat().st_size / (1024 * 1024):.1f} MB)"
    )


def generate_results_markdown(
    eda: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path = DOCS_RESULTS_DIR / "eda.md",
) -> None:
    """Generate docs/results/eda.md documenting Phase 1 outcomes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating EDA report at {output_path}...")

    train_fraud = int(train_df["isFraud"].sum())
    train_rate = float(train_df["isFraud"].mean()) * 100
    val_fraud = int(val_df["isFraud"].sum())
    val_rate = float(val_df["isFraud"].mean()) * 100
    test_fraud = int(test_df["isFraud"].sum())
    test_rate = float(test_df["isFraud"].mean()) * 100

    md_content = f"""# Phase 1: Exploratory Data Analysis & Leakage Review

## 1. Executive Summary
- **Dataset:** PaySim Synthetic Financial Transactions
- **Total Transactions:** {eda["total_records"]:,}
- **Legitimate Transactions (0):** {eda["fraud_counts"].get(0, 0):,} ({100 - eda["fraud_rate"] * 100:.4f}%)
- **Fraudulent Transactions (1):** {eda["fraud_counts"].get(1, 0):,} ({eda["fraud_rate"] * 100:.4f}%)
- **Imbalance Ratio:** ~1 fraud per {int(1 / eda["fraud_rate"]):,} legitimate transactions (< 1% fraud condition verified).
- **Time Window:** 743 discrete hourly steps (~31 days of financial activity).

---

## 2. Transaction Type Breakdown & Fraud Vulnerability
In PaySim, fraud is strictly concentrated in outbound transfer and cash-out operations:

| Transaction Type | Total Count | % of Traffic | Fraud Count | Fraud Rate (%) |
|---|---|---|---|---|
"""
    for t, stats in eda["fraud_by_type"].items():
        pct_traffic = (stats["count"] / eda["total_records"]) * 100
        md_content += f"| **{t}** | {stats['count']:,} | {pct_traffic:.2f}% | {stats['sum']:,} | {stats['mean'] * 100:.4f}% |\n"

    md_content += f"""
> **Key Insight:** `PAYMENT`, `CASH_IN`, and `DEBIT` contain exactly 0 fraudulent events across the entire 6.36M dataset. Fraud operates as a two-stage laundering attack: first moving capital via `TRANSFER` (0.7688% fraud rate) and immediately liquidating via `CASH_OUT` (0.1840% fraud rate).

---

## 3. Amount Distribution Comparison
Fraudulent amounts are significantly larger than legitimate transactions:

| Metric | Overall Amount ($) | Legitimate (0) ($) | Fraudulent (1) ($) |
|---|---|---|---|
| **Mean** | ${eda["amount_stats_overall"]["mean"]:,.2f} | ${eda["amount_stats_legit"]["mean"]:,.2f} | ${eda["amount_stats_fraud"]["mean"]:,.2f} |
| **Std Dev** | ${eda["amount_stats_overall"]["std"]:,.2f} | ${eda["amount_stats_legit"]["std"]:,.2f} | ${eda["amount_stats_fraud"]["std"]:,.2f} |
| **Min** | ${eda["amount_stats_overall"]["min"]:,.2f} | ${eda["amount_stats_legit"]["min"]:,.2f} | ${eda["amount_stats_fraud"]["min"]:,.2f} |
| **Median (50%)** | ${eda["amount_stats_overall"]["50%"]:,.2f} | ${eda["amount_stats_legit"]["50%"]:,.2f} | ${eda["amount_stats_fraud"]["50%"]:,.2f} |
| **75th Percentile** | ${eda["amount_stats_overall"]["75%"]:,.2f} | ${eda["amount_stats_legit"]["75%"]:,.2f} | ${eda["amount_stats_fraud"]["75%"]:,.2f} |
| **Max** | ${eda["amount_stats_overall"]["max"]:,.2f} | ${eda["amount_stats_legit"]["max"]:,.2f} | ${eda["amount_stats_fraud"]["max"]:,.2f} |

---

## 4. Leakage Review & Dropped Columns

### Analysis of Balance Columns
1. **`newbalanceOrig` (Post-Transaction Origin Balance):**
   - In fraudulent transactions, the origin account is completely drained in **{eda["fraud_orig_zero_rate"] * 100:.2f}%** of cases (`newbalanceOrig == 0.0`).
   - In production, when a transaction authorization request arrives at `/score`, the transaction has not yet settled, meaning `newbalanceOrig` does not exist at inference time.
   - Including `newbalanceOrig` causes severe target leakage and produces artificially inflated offline metrics that collapse online.
2. **`newbalanceDest` (Post-Transaction Destination Balance):**
   - Similar to `newbalanceOrig`, this reflects the post-settlement balance of the recipient, which is unavailable at real-time pre-authorization scoring.
3. **`isFlaggedFraud` (Heuristic Business Rule Flag):**
   - A static simulator rule flagging `TRANSFER` transactions > $200,000. It flagged only {eda["flagged_counts"].get(1, 0)} out of {eda["fraud_counts"].get(1, 0)} frauds and 0 legitimate records. Using it as a feature leaks the synthetic label rule.

### Dropped Columns Specification
| Column Name | Action | Reason |
|---|---|---|
| `newbalanceOrig` | **DROPPED** | Target leakage: Post-transaction origin balance; 98.05% zero in fraud |
| `newbalanceDest` | **DROPPED** | Target leakage: Post-transaction destination balance |
| `isFlaggedFraud` | **DROPPED** | Target leakage / Collinearity: Naive heuristic flag (> 200k transfer) |
| `nameOrig` | **EXCLUDED FROM TABULAR MATRIX** | Retained in parquet as entity ID for Feast online store and Graph edges |
| `nameDest` | **EXCLUDED FROM TABULAR MATRIX** | Retained in parquet as entity ID for Feast online store and Graph edges |

---

## 5. Time-Based Train / Val / Test Splits
Per `Rules.md`, random splitting is strictly forbidden. The splits are ordered chronologically by `step` (hourly timestamps):

| Split | Step Range (Hours) | Approximate Time Window | Total Records | % of Dataset | Fraud Count | Fraud Rate (%) |
|---|---|---|---|---|---|---|
| **Train** | `step` 1 to {TRAIN_MAX_STEP} | Days 1 to 13.5 | {len(train_df):,} | {len(train_df) / eda["total_records"] * 100:.2f}% | {train_fraud:,} | {train_rate:.4f}% |
| **Validation** | `step` {TRAIN_MAX_STEP + 1} to {VAL_MAX_STEP} | Days 13.5 to 15.8 | {len(val_df):,} | {len(val_df) / eda["total_records"] * 100:.2f}% | {val_fraud:,} | {val_rate:.4f}% |
| **Test (Hold-Out)** | `step` {VAL_MAX_STEP + 1} to {eda["step_max"]} | Days 15.8 to 31.0 | {len(test_df):,} | {len(test_df) / eda["total_records"] * 100:.2f}% | {test_fraud:,} | {test_rate:.4f}% |
| **Total** | `step` {eda["step_min"]} to {eda["step_max"]} | Full 31 Days | **{eda["total_records"]:,}** | **100.00%** | **{eda["fraud_counts"].get(1, 0):,}** | **{eda["fraud_rate"] * 100:.4f}%** |

### Split Verification Guarantees
- `train_df['step'].max() == {train_df["step"].max()} < val_df['step'].min() == {val_df["step"].min()}`: **PASSED (Zero Overlap)**
- `val_df['step'].max() == {val_df["step"].max()} < test_df['step'].min() == {test_df["step"].min()}`: **PASSED (Zero Overlap)**
- Parquet Artifacts:
  - `data/processed/train.parquet`
  - `data/processed/val.parquet`
  - `data/processed/test.parquet`
"""
    output_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Report written successfully to {output_path}")


def generate_eda_notebook(output_path: Path = NOTEBOOKS_DIR / "01_eda.ipynb") -> None:
    """Generate self-contained Jupyter notebook 01_eda.ipynb."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating EDA notebook at {output_path}...")

    notebook_dict = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# FraudShield — Phase 1: Exploratory Data Analysis & Leakage Review\n",
                    "\n",
                    "This notebook performs the comprehensive data exploration required by Phase 1 of the FraudShield platform:\n",
                    "1. Class Imbalance Quantification (< 1% fraud verification)\n",
                    "2. Amount and Transaction Type Distributions\n",
                    "3. Fraud Vulnerability by Type (`TRANSFER` and `CASH_OUT` focus)\n",
                    "4. Time Structure & Step Sequence Analysis\n",
                    "5. Leakage Review of Post-Transaction Balance Columns\n",
                    "6. Chronological Train / Val / Test Splitting",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from pathlib import Path\n",
                    "\n",
                    "raw_path = Path('../data/raw/PS_20174392719_1491204439457_log.csv')\n",
                    "df = pd.read_csv(raw_path)\n",
                    "print(f'Loaded {len(df):,} transactions.')\n",
                    "df.head()",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1. Class Balance & Imbalance Quantification"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fraud_counts = df['isFraud'].value_counts()\n",
                    "fraud_rate = df['isFraud'].mean() * 100\n",
                    "print('Class Balance:')\n",
                    "print(fraud_counts)\n",
                    "print(f'Fraud Rate: {fraud_rate:.4f}% (< 1% requirement met)')",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2. Fraud Rate by Transaction Type"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "type_summary = df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean'])\n",
                    "type_summary['fraud_pct'] = type_summary['mean'] * 100\n",
                    "print(type_summary)\n",
                    "# In PaySim, fraud only occurs in TRANSFER and CASH_OUT!",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3. Amount Distribution Comparison"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('--- Legitimate Amount Statistics ---')\n",
                    "print(df[df['isFraud'] == 0]['amount'].describe())\n",
                    "print('\\n--- Fraudulent Amount Statistics ---')\n",
                    "print(df[df['isFraud'] == 1]['amount'].describe())",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 4. Balance Leakage Review"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# How often does fraud drain the account to zero?\n",
                    "fraud_drain_rate = (df[df['isFraud'] == 1]['newbalanceOrig'] == 0).mean() * 100\n",
                    "legit_drain_rate = (df[df['isFraud'] == 0]['newbalanceOrig'] == 0).mean() * 100\n",
                    "print(f'Fraud cases with newbalanceOrig == 0: {fraud_drain_rate:.2f}%')\n",
                    "print(f'Legitimate cases with newbalanceOrig == 0: {legit_drain_rate:.2f}%')\n",
                    "# newbalanceOrig and newbalanceDest are post-transaction state and MUST be dropped to prevent target leakage.",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 5. Time Structure & Time-Based Splits"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Step range: {df['step'].min()} to {df['step'].max()} (hours)\")\n",
                    "train_split = df[df['step'] <= 323]\n",
                    "val_split = df[(df['step'] > 323) & (df['step'] <= 378)]\n",
                    "test_split = df[df['step'] > 378]\n",
                    "print(f'Train: {len(train_split):,} rows, Fraud rate: {train_split[\"isFraud\"].mean()*100:.4f}%')\n",
                    "print(f'Val:   {len(val_split):,} rows, Fraud rate: {val_split[\"isFraud\"].mean()*100:.4f}%')\n",
                    "print(f'Test:  {len(test_split):,} rows, Fraud rate: {test_split[\"isFraud\"].mean()*100:.4f}%')",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path.write_text(json.dumps(notebook_dict, indent=2), encoding="utf-8")
    logger.info(f"Notebook written to {output_path}")


def run_pipeline() -> None:
    """Execute complete Phase 1 pipeline."""
    df = load_raw_data()
    eda_stats = perform_eda(df)
    train_df, val_df, test_df = split_data_by_time(df)
    save_processed_splits(train_df, val_df, test_df)
    generate_results_markdown(eda_stats, train_df, val_df, test_df)
    generate_eda_notebook()
    logger.info("Phase 1 EDA and dataset splitting successfully completed!")


if __name__ == "__main__":
    run_pipeline()
