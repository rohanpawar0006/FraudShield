"""Unit and acceptance tests for Phase 6 feedback data layer, case investigation, and audit trail."""

import sqlite3

import pytest

from src.common.config import settings
from src.common.db import db_manager
from src.feedback.service import (
    get_case,
    get_labeled_cases,
    list_alerts,
    submit_feedback,
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Use an isolated SQLite database for feedback tests."""
    test_db = tmp_path / "test_feedback.db"
    orig_path = settings.SQLITE_DB_PATH
    orig_fallback = settings.USE_SQLITE_FALLBACK

    object.__setattr__(settings, "SQLITE_DB_PATH", str(test_db))
    object.__setattr__(settings, "USE_SQLITE_FALLBACK", True)

    db_manager._backend = "sqlite"
    db_manager._init_sqlite_schema(test_db)

    # Insert sample transaction and score records
    with sqlite3.connect(str(test_db)) as conn:
        conn.execute(
            """
            INSERT INTO transactions (txn_id, account_id, dest_id, amount, type, ts, raw)
            VALUES
                ('txn_alert_001', 'C11111', 'C22222', 150000.0, 'TRANSFER', '2026-11-01T12:00:00Z', '{}'),
                ('txn_alert_002', 'C33333', 'C44444', 250000.0, 'CASH_OUT', '2026-11-01T12:05:00Z', '{}'),
                ('txn_low_003',   'C55555', 'C66666', 50.0,     'PAYMENT',  '2026-11-01T12:10:00Z', '{}');
            """
        )
        conn.execute(
            """
            INSERT INTO scores (txn_id, xgb_score, gnn_score, blended_score, model_version, scored_at, latency_ms)
            VALUES
                ('txn_alert_001', 0.995, 0.45, 0.995, 'FraudShield-XGBoost:1', '2026-11-01T12:00:01Z', 12.5),
                ('txn_alert_002', 0.992, 0.30, 0.992, 'FraudShield-XGBoost:1', '2026-11-01T12:05:01Z', 15.0),
                ('txn_low_003',   0.002, 0.10, 0.002, 'FraudShield-XGBoost:1', '2026-11-01T12:10:01Z', 10.0);
            """
        )
        conn.commit()

    yield test_db

    object.__setattr__(settings, "SQLITE_DB_PATH", orig_path)
    object.__setattr__(settings, "USE_SQLITE_FALLBACK", orig_fallback)


def test_submit_feedback_inserts_and_updates():
    """Verify submit_feedback creates case, logs label, and preserves audit trail on update."""
    txn_id = "txn_alert_001"

    # 1. First submission: Under Review
    res1 = submit_feedback(
        transaction_id=txn_id,
        disposition="Under Review",
        analyst="investigator_alice",
        note="Initial triage: high value transfer",
    )
    assert res1["disposition"] == "Under Review"
    assert res1["previous_disposition"] == "New"
    assert res1["label"] == -1

    case1 = get_case(txn_id)
    assert case1 is not None
    assert case1["status"] == "Under Review"
    assert case1["assigned_to"] == "investigator_alice"
    assert len(case1["audit_trail"]) == 1
    assert case1["audit_trail"][0]["comment"] == "Initial triage: high value transfer"

    # 2. Second submission: Confirmed Fraud
    res2 = submit_feedback(
        transaction_id=txn_id,
        disposition="Confirmed Fraud",
        analyst="supervisor_bob",
        note="Customer verified account takeover",
    )
    assert res2["disposition"] == "Confirmed Fraud"
    assert res2["previous_disposition"] == "Under Review"
    assert res2["label"] == 1

    case2 = get_case(txn_id)
    assert case2["status"] == "Confirmed Fraud"
    assert case2["assigned_to"] == "supervisor_bob"
    assert len(case2["audit_trail"]) == 2
    assert case2["audit_trail"][0]["disposition"] == "Confirmed Fraud"
    assert case2["audit_trail"][0]["previous_disposition"] == "Under Review"
    assert case2["audit_trail"][1]["disposition"] == "Under Review"


def test_invalid_disposition_raises_value_error():
    """Verify submitting unrecognized disposition raises ValueError."""
    with pytest.raises(ValueError, match="Invalid disposition"):
        submit_feedback(
            transaction_id="txn_alert_001",
            disposition="Invalid Disposition Text",
            analyst="alice",
        )


def test_get_labeled_cases_mapping_and_exclusion():
    """Verify get_labeled_cases maps Confirmed Fraud -> 1, False Positive -> 0, and excludes Under Review."""
    # Label txn_alert_001 as Confirmed Fraud
    submit_feedback("txn_alert_001", "Confirmed Fraud", "analyst_1")

    # Label txn_alert_002 as Under Review (must be excluded)
    submit_feedback("txn_alert_002", "Under Review", "analyst_2")

    # Label txn_low_003 as False Positive
    submit_feedback("txn_low_003", "False Positive", "analyst_3")

    df = get_labeled_cases()
    assert isinstance(df, object)
    assert len(df) == 2
    assert set(df["transaction_id"]) == {"txn_alert_001", "txn_low_003"}

    # Verify label integer values
    row_fraud = df[df["transaction_id"] == "txn_alert_001"].iloc[0]
    assert int(row_fraud["label"]) == 1

    row_legit = df[df["transaction_id"] == "txn_low_003"].iloc[0]
    assert int(row_legit["label"]) == 0


def test_list_alerts_filters():
    """Verify list_alerts filters by status, score ranges, and payment types."""
    # Low score transaction (0.002) is excluded by default decision threshold (0.99)
    alerts = list_alerts()
    assert len(alerts) == 2

    # Filter by transaction type
    transfer_alerts = list_alerts(txn_type="TRANSFER")
    assert len(transfer_alerts) == 1
    assert transfer_alerts[0]["txn_id"] == "txn_alert_001"

    # Submit feedback on txn_alert_001
    submit_feedback("txn_alert_001", "Confirmed Fraud", "alice")

    # Filter by status
    new_alerts = list_alerts(status="New")
    assert len(new_alerts) == 1
    assert new_alerts[0]["txn_id"] == "txn_alert_002"

    confirmed_alerts = list_alerts(status="Confirmed Fraud")
    assert len(confirmed_alerts) == 1
    assert confirmed_alerts[0]["txn_id"] == "txn_alert_001"


def test_acceptance_feedback_loop():
    """Acceptance test: Prove full 8-step feedback lifecycle from alert triage to retraining dataset.

    Steps:
    1. Score a transaction that generates an alert.
    2. Confirm it appears in list_alerts(status='New').
    3. Submit feedback 'Confirmed Fraud'.
    4. Confirm status changes to 'Confirmed Fraud' and it disappears from status='New'.
    5. get_labeled_cases() includes this transaction with label=1.
    6. Submit updated feedback 'False Positive' for the same transaction.
    7. get_labeled_cases() now returns label=0, not two rows.
    8. Audit trail shows both submissions with timestamps and analysts.
    """
    txn_id = "txn_acceptance_loop_001"

    # 1. Score a transaction that generates an alert and persist to DB
    db_manager.save_transaction_and_score(
        txn_id=txn_id,
        account_id="C999999",
        dest_id="C888888",
        amount=500000.0,
        txn_type="TRANSFER",
        ts=None,
        raw_payload={"type": "TRANSFER", "amount": 500000.0, "oldbalanceOrg": 500000.0, "oldbalanceDest": 0.0},
        xgb_score=0.998,
        gnn_score=0.50,
        blended_score=0.998,
        model_version="FraudShield-XGBoost:1",
        latency_ms=12.0,
    )
    db_manager.flush(timeout=2.0)

    # 2. Confirm it appears in list_alerts(status='New')
    new_alerts = list_alerts(status="New")
    assert any(a["txn_id"] == txn_id for a in new_alerts), "Alert must appear in list_alerts(status='New')"

    # 3. Submit feedback 'Confirmed Fraud'
    res1 = submit_feedback(
        transaction_id=txn_id,
        disposition="Confirmed Fraud",
        analyst="analyst_alice",
        note="Confirmed fraudulent transfer via stolen credentials",
    )
    assert res1["disposition"] == "Confirmed Fraud"

    # 4. Confirm status changes to 'Confirmed Fraud' and disappears from status='New'
    new_alerts_after = list_alerts(status="New")
    assert not any(a["txn_id"] == txn_id for a in new_alerts_after), "Must disappear from status='New'"
    confirmed_alerts = list_alerts(status="Confirmed Fraud")
    assert any(a["txn_id"] == txn_id for a in confirmed_alerts), "Must appear in status='Confirmed Fraud'"

    case = get_case(txn_id)
    assert case["status"] == "Confirmed Fraud"

    # 5. get_labeled_cases() includes this transaction with label=1
    df1 = get_labeled_cases()
    assert txn_id in df1["transaction_id"].values
    assert int(df1.loc[df1["transaction_id"] == txn_id, "label"].iloc[0]) == 1

    # 6. Submit updated feedback 'False Positive' for the same transaction
    res2 = submit_feedback(
        transaction_id=txn_id,
        disposition="False Positive",
        analyst="auditor_bob",
        note="Customer verified transaction on secondary authentication channel",
    )
    assert res2["disposition"] == "False Positive"
    assert res2["previous_disposition"] == "Confirmed Fraud"

    # 7. get_labeled_cases() now returns label=0, not two rows
    df2 = get_labeled_cases()
    txn_rows = df2[df2["transaction_id"] == txn_id]
    assert len(txn_rows) == 1, "Must deduplicate to the latest submission only, not two rows"
    assert int(txn_rows.iloc[0]["label"]) == 0, "Updated label must be 0"

    # 8. Audit trail shows both submissions with timestamps and analysts
    case_updated = get_case(txn_id)
    audit = case_updated["audit_trail"]
    assert len(audit) == 2, "Audit trail must preserve both historical submissions"
    assert audit[0]["disposition"] == "False Positive"
    assert audit[0]["analyst"] == "auditor_bob"
    assert audit[1]["disposition"] == "Confirmed Fraud"
    assert audit[1]["analyst"] == "analyst_alice"
    assert audit[0]["labeled_at"] is not None
    assert audit[1]["labeled_at"] is not None


def test_orphan_and_duplicate_scores_not_dropped():
    """Verify list_alerts does not drop orphan scores and deduplicates duplicate scores."""
    # Insert an orphan score (no transaction row)
    orphan_txn_id = "orphan_alert_999"
    with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO scores (txn_id, xgb_score, gnn_score, blended_score, model_version, scored_at, latency_ms)
            VALUES (?, 0.999, NULL, 0.999, 'v1', '2026-11-01T15:00:00Z', 10.0);
            """,
            (orphan_txn_id,),
        )
        # Insert a duplicate score for an existing transaction with higher score
        conn.execute(
            """
            INSERT INTO scores (txn_id, xgb_score, gnn_score, blended_score, model_version, scored_at, latency_ms)
            VALUES ('txn_alert_002', 0.998, 0.40, 0.998, 'v1', '2026-11-01T15:05:00Z', 11.0);
            """,
        )
        conn.commit()

    alerts = list_alerts(min_score=0.99)
    txn_ids = [a["txn_id"] for a in alerts]

    # Orphan score is retained and NOT dropped by joins
    assert orphan_txn_id in txn_ids, "Orphan score without transaction row must NOT be dropped by dashboard join"

    # Duplicate score does not create duplicate alert rows (deduplicated by latest_score)
    count_002 = sum(1 for tid in txn_ids if tid == "txn_alert_002")
    assert count_002 == 1, "Duplicate scores for same transaction must be deduplicated in list_alerts"
