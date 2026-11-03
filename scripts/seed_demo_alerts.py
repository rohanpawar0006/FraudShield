"""Seed demo alerts by replaying a slice of hold-out test transactions through the scoring engine.

Ensures real fraud alerts are generated and persisted into PostgreSQL (or SQLite fallback)
for the analyst dashboard and feedback triage workflows.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.common.db import db_manager  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.serving.schemas import TransactionPayload  # noqa: E402
from src.serving.scoring import ScoringEngine  # noqa: E402

logger = get_logger(__name__)


def seed_alerts(
    start_index: int = 10000,
    size: int = 5000,
    test_data_path: str = "data/processed/test.parquet",
    clean: bool = False,
) -> None:
    """Replay hold-out test transactions through the scoring pipeline and persist alerts.

    Args:
        start_index: 0-based offset into test data.
        size: Number of transactions to replay.
        test_data_path: Path to hold-out test set parquet file.
        clean: If True, reset and clean SQLite database tables prior to seeding.
    """
    if clean:
        print("Cleaning demo database prior to seeding (--clean specified)...")
        conn = db_manager.get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM labels;")
        cur.execute("DELETE FROM cases;")
        cur.execute("DELETE FROM scores;")
        cur.execute("DELETE FROM transactions;")
        conn.commit()
        conn.close()
        print("Demo database tables cleared.")

    path = Path(test_data_path)
    if not path.exists():
        logger.error("Test data file not found at %s", path)
        sys.exit(1)

    print(f"Loading hold-out test set from {path}...")
    df = pd.read_parquet(path)
    total_test_rows = len(df)

    end_index = min(start_index + size, total_test_rows)
    slice_df = df.iloc[start_index:end_index]
    actual_size = len(slice_df)

    fraud_in_ground_truth = int(slice_df["isFraud"].sum()) if "isFraud" in slice_df.columns else 0
    print(
        f"Replaying slice [{start_index}:{end_index}] ({actual_size} transactions). "
        f"Ground-truth frauds in slice: {fraud_in_ground_truth}"
    )

    print("Initializing ScoringEngine...")
    engine = ScoringEngine()

    processed_count = 0
    alerts_count = 0
    start_time = time.time()

    print(f"Scoring {actual_size} transactions...")
    for idx, (row_idx, row) in enumerate(slice_df.iterrows()):
        txn_id = f"seed_txn_{row_idx}"
        payload = TransactionPayload(
            transaction_id=txn_id,
            step=int(row["step"]),
            type=str(row["type"]),
            amount=float(row["amount"]),
            nameOrig=str(row["nameOrig"]),
            oldbalanceOrg=float(row["oldbalanceOrg"]),
            nameDest=str(row["nameDest"]),
            oldbalanceDest=float(row["oldbalanceDest"]),
        )

        response = engine.score(payload)
        processed_count += 1
        if response.is_alert:
            alerts_count += 1

        # Enqueue for persistence off the critical path
        db_manager.save_transaction_and_score(
            txn_id=payload.txn_id,
            account_id=payload.nameOrig,
            dest_id=payload.nameDest,
            amount=payload.amount,
            txn_type=payload.type,
            ts=None,
            raw_payload=payload.model_dump(),
            xgb_score=response.xgb_score,
            gnn_score=response.gnn_score,
            blended_score=response.blend_score,
            model_version=response.model_version,
            latency_ms=response.latency_ms,
        )

        if (idx + 1) % 500 == 0 or (idx + 1) == actual_size:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processed {idx + 1}/{actual_size} ({rate:.1f} txn/s) - Alerts generated: {alerts_count}")

    print("Flushing database persistence worker queue...")
    db_manager.flush(timeout=10.0)
    # Brief pause to ensure async writes finish committing
    time.sleep(0.5)

    # Query final table counts
    conn = db_manager.get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM transactions;")
    txn_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM scores;")
    scores_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM cases;")
    cases_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM labels;")
    labels_count = cur.fetchone()[0]

    cur.execute(
        f"SELECT COUNT(*) FROM scores WHERE blended_score >= {settings.DECISION_THRESHOLD};"
    )
    total_db_alerts = cur.fetchone()[0]

    conn.close()

    print("\n" + "=" * 60)
    print("DEMO ALERT SEEDING SUMMARY")
    print("=" * 60)
    print(f"Transactions processed:     {processed_count}")
    print(f"Alerts generated in run:    {alerts_count}")
    print(f"Ground-truth fraud count:   {fraud_in_ground_truth}")
    print(f"Decision threshold:         {settings.DECISION_THRESHOLD}")
    print("-" * 60)
    print("Database Table Counts:")
    print(f"  - transactions:           {txn_count}")
    print(f"  - scores:                 {scores_count}")
    print(f"  - cases:                  {cases_count}")
    print(f"  - labels:                 {labels_count}")
    print(f"  - total alerts (in db):   {total_db_alerts}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Seed demo alerts from hold-out test set.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=10500,
        help="0-based row index in test.parquet to start replay (default: 10500)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2500,
        help="Number of transactions to replay (default: 2500, captures multiple test frauds)",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/processed/test.parquet",
        help="Path to hold-out test set parquet file",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean database tables before seeding to establish pristine 1:1 demo state",
    )
    args = parser.parse_args()

    seed_alerts(
        start_index=args.start_index,
        size=args.size,
        test_data_path=args.test_path,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
