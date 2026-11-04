"""Helper script to test PostgreSQL data persistence across container restart and compose down/up."""
# ruff: noqa: E402

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(".env", override=True)

from src.common.db import db_manager

MARKER_TXN_ID = "marker_persist_txn_999"

def insert_marker():
    print(f"[Persistence Test] Inserting marker transaction '{MARKER_TXN_ID}'...")
    conn = db_manager.get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO transactions (txn_id, account_id, dest_id, amount, type, ts, raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (txn_id) DO NOTHING;
            """,
            (
                MARKER_TXN_ID,
                "C_MARKER_ORIG",
                "M_MARKER_DEST",
                9999.99,
                "TRANSFER",
                datetime.now(timezone.utc),
                '{"marker": true}',
            ),
        )
        cur.execute(
            """
            INSERT INTO scores (txn_id, xgb_score, blended_score, model_version, latency_ms)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (MARKER_TXN_ID, 0.999, 0.999, "persist-test", 5.0),
        )
    conn.commit()
    conn.close()
    verify_marker()

def verify_marker():
    print(f"[Persistence Test] Verifying marker transaction '{MARKER_TXN_ID}' exists in PostgreSQL...")
    conn = db_manager.get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT txn_id, amount, type FROM transactions WHERE txn_id = %s;", (MARKER_TXN_ID,))
        txn = cur.fetchone()
        cur.execute("SELECT txn_id, blended_score FROM scores WHERE txn_id = %s;", (MARKER_TXN_ID,))
        score = cur.fetchone()
    conn.close()

    if txn and score:
        print(f" -> VERIFIED: Transaction {txn[0]} (amount: {txn[1]}, type: {txn[2]}) and Score {score[0]} (blended: {score[1]}) are PRESENT.")
        return True
    else:
        print(f" -> ERROR: Marker {MARKER_TXN_ID} NOT FOUND in database!")
        sys.exit(1)

def cleanup_marker():
    print(f"[Persistence Test] Cleaning up marker transaction '{MARKER_TXN_ID}'...")
    conn = db_manager.get_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM transactions WHERE txn_id = %s;", (MARKER_TXN_ID,))
        cur.execute("SELECT COUNT(*) FROM transactions WHERE txn_id = %s;", (MARKER_TXN_ID,))
        cnt = cur.fetchone()[0]
    conn.commit()
    conn.close()
    print(f" -> CLEANUP COMPLETE: Remaining rows for {MARKER_TXN_ID}: {cnt}")
    assert cnt == 0

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if action == "insert":
        insert_marker()
    elif action == "verify":
        verify_marker()
    elif action == "cleanup":
        cleanup_marker()
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)

