"""Integration test executing end-to-end CRUD using the application's DB layer against PostgreSQL."""
# ruff: noqa: E402

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(".env", override=True)

import src.common.config as config
from src.common.db import db_manager
from src.feedback.service import (
    get_case,
    list_alerts,
    submit_feedback,
)


def run_crud_test():
    print("=== Running PostgreSQL Application CRUD Verification ===")
    print(f"PostgreSQL Host: {config.settings.POSTGRES_HOST}:{config.settings.POSTGRES_PORT}")
    print(f"PostgreSQL DB:   {config.settings.POSTGRES_DB}")
    print(f"Backend Active:  {db_manager.backend}")
    print(f"Reachable:       {db_manager.is_reachable()}")
    assert db_manager.backend == "postgresql", f"Expected postgresql, got {db_manager.backend}"
    assert db_manager.is_reachable(), "PostgreSQL is not reachable"

    test_txn_id = "test_crud_txn_001"
    account_orig = "C1234567890"
    account_dest = "M9876543210"

    # Step 1: Clean any pre-existing test record
    conn = db_manager.get_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM transactions WHERE txn_id = %s;", (test_txn_id,))
    conn.commit()

    # Step 2: Insert transaction and score using app persistence layer
    print("\n[Step 2] Inserting transaction and score via db_manager.save_transaction_and_score()...")
    saved = db_manager.save_transaction_and_score(
        txn_id=test_txn_id,
        account_id=account_orig,
        dest_id=account_dest,
        amount=5432.10,
        txn_type="TRANSFER",
        ts=datetime.now(timezone.utc),
        raw_payload={"step": 1, "type": "TRANSFER", "amount": 5432.10, "nameOrig": account_orig, "nameDest": account_dest},
        xgb_score=0.952,
        gnn_score=None,
        blended_score=0.952,
        model_version="1.0.0",
        latency_ms=12.4,
    )
    assert saved, "Failed to enqueue transaction"
    db_manager.flush()
    time.sleep(0.5)

    # Step 3: Read back transaction & score from PostgreSQL
    print("[Step 3] Reading back transaction and score...")
    with conn.cursor() as cur:
        cur.execute("SELECT txn_id, account_id, dest_id, amount, type FROM transactions WHERE txn_id = %s;", (test_txn_id,))
        txn_row = cur.fetchone()
        assert txn_row is not None, "Transaction not found in PostgreSQL"
        print(f" -> Found transaction: txn_id={txn_row[0]}, account_id={txn_row[1]}, amount={txn_row[3]}, type={txn_row[4]}")

        cur.execute("SELECT txn_id, xgb_score, blended_score, model_version FROM scores WHERE txn_id = %s;", (test_txn_id,))
        score_row = cur.fetchone()
        assert score_row is not None, "Score not found in PostgreSQL"
        print(f" -> Found score: txn_id={score_row[0]}, xgb_score={score_row[1]}, blended_score={score_row[2]}")

    # Step 4: Record analyst feedback (Under Review)
    print("\n[Step 4] Submitting analyst feedback via submit_feedback('Under Review')...")
    res1 = submit_feedback(
        transaction_id=test_txn_id,
        disposition="Under Review",
        analyst="analyst_alice",
        note="Suspected mule account pattern"
    )
    print(f" -> Result: {res1}")

    # Step 5: Read back via list_alerts and get_case
    print("\n[Step 5] Reading back via get_case() and list_alerts()...")
    case1 = get_case(test_txn_id)
    assert case1 is not None, "Case not found"
    print(f" -> Case status: {case1['status']}, assigned_to: {case1['assigned_to']}")
    print(f" -> Audit trail count: {len(case1['audit_trail'])}")
    assert case1["status"] == "Under Review"
    assert len(case1["audit_trail"]) == 1

    alerts = list_alerts(min_score=0.90, limit=10)
    matching_alerts = [a for a in alerts if a["txn_id"] == test_txn_id]
    assert len(matching_alerts) == 1, "Alert not found in list_alerts query"
    print(f" -> Found alert in list_alerts: txn_id={matching_alerts[0]['txn_id']}, status={matching_alerts[0]['status']}")

    # Step 6: Update feedback (Confirmed Fraud)
    print("\n[Step 6] Updating analyst feedback via submit_feedback('Confirmed Fraud')...")
    res2 = submit_feedback(
        transaction_id=test_txn_id,
        disposition="Confirmed Fraud",
        analyst="analyst_alice",
        note="Confirmed by bank operations phone inquiry"
    )
    print(f" -> Result: {res2}")
    assert res2["previous_disposition"] == "Under Review"

    case2 = get_case(test_txn_id)
    assert case2 is not None, "Case not found after update"
    print(f" -> Updated case status: {case2['status']}")
    print(f" -> Updated audit trail count: {len(case2['audit_trail'])}")
    assert case2["status"] == "Confirmed Fraud"
    assert len(case2["audit_trail"]) == 2
    assert case2["audit_trail"][0]["previous_disposition"] == "Under Review"

    # Step 7: Delete only the test row created
    print("\n[Step 7] Deleting test record (verifying ON DELETE CASCADE)...")
    with conn.cursor() as cur:
        cur.execute("DELETE FROM transactions WHERE txn_id = %s;", (test_txn_id,))
    conn.commit()

    # Step 8: Verify full cleanup across all 4 tables
    with conn.cursor() as cur:
        for tbl in ["transactions", "scores", "cases", "labels"]:
            cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE txn_id = %s;", (test_txn_id,))
            cnt = cur.fetchone()[0]
            print(f" -> Table {tbl} remaining rows for {test_txn_id}: {cnt}")
            assert cnt == 0, f"Table {tbl} still has records for {test_txn_id}"

    conn.close()
    print("\n=== Application CRUD Verification PASSED (100% Success) ===")

if __name__ == "__main__":
    run_crud_test()
