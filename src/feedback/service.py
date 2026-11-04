"""Feedback and case investigation service for FraudShield analyst dashboard.

Provides parameterized data access for alert triage, case history, analyst feedback submission,
audit trail tracking, and export of verified ground-truth labels for retraining.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.config import settings
from src.common.db import db_manager
from src.common.logging import get_logger

logger = get_logger(__name__)

VALID_DISPOSITIONS = {"Confirmed Fraud", "False Positive", "Under Review"}
DISPOSITION_TO_LABEL = {
    "Confirmed Fraud": 1,
    "False Positive": 0,
    "Under Review": -1,
}


def _get_placeholder(backend: str) -> str:
    """Return parameterized query placeholder ('%s' for PostgreSQL, '?' for SQLite)."""
    return "%s" if backend == "postgresql" else "?"


def list_alerts(
    status: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    txn_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch paginated alerts meeting filter criteria.

    Args:
        status: Filter by case status ('New', 'Under Review', 'Confirmed Fraud', 'False Positive')
        min_score: Lower bound on blended fraud risk score in [0, 1]
        max_score: Upper bound on blended fraud risk score in [0, 1]
        txn_type: Transaction type ('TRANSFER', 'CASH_OUT', etc.)
        limit: Number of records to return
        offset: Pagination offset

    Returns:
        List of alert dictionaries sorted by blended_score DESC.
    """
    conn = db_manager.get_connection()
    ph = _get_placeholder(db_manager.backend)

    effective_min_score = min_score if min_score is not None else settings.DECISION_THRESHOLD
    conditions = [f"s.blended_score >= {ph}"]
    params: List[Any] = [float(effective_min_score)]

    if status:
        if status.lower() == "new":
            conditions.append(f"(c.status IS NULL OR c.status = {ph})")
            params.append("New")
        else:
            conditions.append(f"c.status = {ph}")
            params.append(status)

    if max_score is not None:
        conditions.append(f"s.blended_score <= {ph}")
        params.append(float(max_score))

    if txn_type:
        conditions.append(f"(t.type = {ph} OR (t.type IS NULL AND {ph} IS NOT NULL))")
        params.extend([txn_type.upper(), txn_type.upper()])

    where_clause = " AND ".join(conditions)
    query = f"""
        SELECT
            COALESCE(t.txn_id, s.txn_id) AS txn_id,
            COALESCE(t.account_id, 'UNKNOWN') AS account_id,
            COALESCE(t.dest_id, 'UNKNOWN') AS dest_id,
            COALESCE(t.amount, 0.0) AS amount,
            COALESCE(t.type, 'UNKNOWN') AS type,
            COALESCE(t.ts, s.scored_at) AS ts,
            s.xgb_score,
            s.gnn_score,
            s.blended_score,
            s.model_version,
            s.scored_at,
            COALESCE(c.status, 'New') AS status,
            c.assigned_to,
            c.updated_at
        FROM (
            SELECT txn_id, MAX(id) AS max_id
            FROM scores
            GROUP BY txn_id
        ) latest_score
        JOIN scores s ON latest_score.txn_id = s.txn_id AND latest_score.max_id = s.id
        LEFT JOIN transactions t ON s.txn_id = t.txn_id
        LEFT JOIN cases c ON s.txn_id = c.txn_id
        WHERE {where_clause}
        ORDER BY s.blended_score DESC
        LIMIT {ph} OFFSET {ph};
    """
    params.extend([limit, offset])

    try:
        cur = conn.cursor()
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for r in rows:
            if hasattr(r, "keys"):
                results.append(dict(r))
            else:
                col_names = [d[0] for d in cur.description]
                results.append(dict(zip(col_names, r)))
        return results
    finally:
        conn.close()


def get_case(transaction_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve complete case details, raw transaction payload, scores, and audit trail.

    Args:
        transaction_id: Unique transaction ID

    Returns:
        Dictionary containing transaction, score, case status, and historical label submissions.
    """
    conn = db_manager.get_connection()
    ph = _get_placeholder(db_manager.backend)

    try:
        cur = conn.cursor()
        # 1. Transaction and score details
        cur.execute(
            f"""
            SELECT
                t.txn_id, t.account_id, t.dest_id, t.amount, t.type, t.ts, t.raw,
                s.xgb_score, s.gnn_score, s.blended_score, s.model_version, s.scored_at, s.latency_ms,
                COALESCE(c.status, 'New') AS status, c.assigned_to, c.updated_at
            FROM transactions t
            LEFT JOIN scores s ON t.txn_id = s.txn_id
            LEFT JOIN cases c ON t.txn_id = c.txn_id
            WHERE t.txn_id = {ph};
            """,
            (transaction_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        if hasattr(row, "keys"):
            case_data = dict(row)
        else:
            col_names = [d[0] for d in cur.description]
            case_data = dict(zip(col_names, row))

        # 2. Audit trail from labels table
        cur.execute(
            f"""
            SELECT id, disposition, previous_disposition, analyst, comment, labeled_at
            FROM labels
            WHERE txn_id = {ph}
            ORDER BY id DESC;
            """,
            (transaction_id,),
        )
        audit_rows = cur.fetchall()
        audit_trail: List[Dict[str, Any]] = []
        for ar in audit_rows:
            if hasattr(ar, "keys"):
                audit_trail.append(dict(ar))
            else:
                col_names = [d[0] for d in cur.description]
                audit_trail.append(dict(zip(col_names, ar)))

        case_data["audit_trail"] = audit_trail
        return case_data
    finally:
        conn.close()


def submit_feedback(
    transaction_id: str,
    disposition: str,
    analyst: str,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit analyst review disposition, updating case status and appending to audit trail.

    Args:
        transaction_id: Unique transaction ID
        disposition: Must be one of 'Confirmed Fraud', 'False Positive', or 'Under Review'
        analyst: Name or ID of reviewer
        note: Optional investigation commentary

    Returns:
        Dictionary reporting update status, previous disposition, and recorded timestamp.
    """
    if disposition not in VALID_DISPOSITIONS:
        raise ValueError(
            f"Invalid disposition '{disposition}'. Allowed values: {VALID_DISPOSITIONS}"
        )

    if not analyst or not str(analyst).strip():
        raise ValueError("Analyst name/ID is required.")

    label_int = DISPOSITION_TO_LABEL[disposition]
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = db_manager.get_connection()
    backend = db_manager.backend
    ph = _get_placeholder(backend)

    try:
        cur = conn.cursor()
        # 1. Fetch previous disposition
        cur.execute(
            f"SELECT status FROM cases WHERE txn_id = {ph};",
            (transaction_id,),
        )
        row = cur.fetchone()
        prev_disposition = row[0] if row else "New"

        # 2. Append to labels table (Audit trail)
        cur.execute(
            f"""
            INSERT INTO labels (txn_id, label, disposition, previous_disposition, analyst, comment)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph});
            """,
            (
                transaction_id,
                label_int,
                disposition,
                prev_disposition,
                analyst.strip(),
                note.strip() if note else None,
            ),
        )

        # 3. Upsert into cases table
        if backend == "postgresql":
            cur.execute(
                f"""
                INSERT INTO cases (txn_id, status, assigned_to, updated_at)
                VALUES ({ph}, {ph}, {ph}, {ph})
                ON CONFLICT (txn_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    assigned_to = EXCLUDED.assigned_to,
                    updated_at = EXCLUDED.updated_at;
                """,
                (transaction_id, disposition, analyst.strip(), now_iso),
            )
        else:
            cur.execute(
                f"""
                INSERT INTO cases (txn_id, status, assigned_to, updated_at)
                VALUES ({ph}, {ph}, {ph}, {ph})
                ON CONFLICT(txn_id) DO UPDATE SET
                    status = excluded.status,
                    assigned_to = excluded.assigned_to,
                    updated_at = excluded.updated_at;
                """,
                (transaction_id, disposition, analyst.strip(), now_iso),
            )

        conn.commit()
        logger.info(
            "Feedback recorded for txn %s: '%s' by %s (prev: '%s')",
            transaction_id,
            disposition,
            analyst,
            prev_disposition,
        )

        return {
            "transaction_id": transaction_id,
            "disposition": disposition,
            "previous_disposition": prev_disposition,
            "label": label_int,
            "analyst": analyst,
            "updated_at": now_iso,
        }
    finally:
        conn.close()


def get_labeled_cases() -> pd.DataFrame:
    """Retrieve final ground-truth labels for model retraining in Phase 7.

    Returns DataFrame with columns ['transaction_id', 'label', 'labeled_at'].
    Excludes 'Under Review' cases. If an alert was reviewed multiple times,
    the most recent disposition is used.
    """
    conn = db_manager.get_connection()
    query = """
        SELECT
            l.txn_id AS transaction_id,
            l.label,
            l.labeled_at
        FROM labels l
        INNER JOIN (
            SELECT txn_id, MAX(id) AS max_id
            FROM labels
            GROUP BY txn_id
        ) latest ON l.txn_id = latest.txn_id AND l.id = latest.max_id
        WHERE l.disposition IN ('Confirmed Fraud', 'False Positive')
        ORDER BY l.labeled_at DESC;
    """
    try:
        df = pd.read_sql_query(query, conn)
        # Ensure correct column typing
        if not df.empty:
            df["label"] = df["label"].astype(int)
        return df[["transaction_id", "label", "labeled_at"]]
    finally:
        conn.close()


def get_alerts_summary_kpis() -> Dict[str, Any]:
    """Calculate header KPI metrics for the alerts dashboard."""
    conn = db_manager.get_connection()
    ph = _get_placeholder(db_manager.backend)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                COUNT(*) AS total_alerts,
                SUM(CASE WHEN c.status IN ('Confirmed Fraud', 'False Positive') THEN 1 ELSE 0 END) AS reviewed_count,
                SUM(CASE WHEN c.status = 'Confirmed Fraud' THEN 1 ELSE 0 END) AS confirmed_count,
                SUM(CASE WHEN c.status = 'False Positive' THEN 1 ELSE 0 END) AS fp_count
            FROM (
                SELECT txn_id, MAX(id) AS max_id
                FROM scores
                GROUP BY txn_id
            ) latest_score
            JOIN scores s ON latest_score.txn_id = s.txn_id AND latest_score.max_id = s.id
            LEFT JOIN transactions t ON s.txn_id = t.txn_id
            LEFT JOIN cases c ON s.txn_id = c.txn_id
            WHERE s.blended_score >= {ph};
            """,
            (settings.DECISION_THRESHOLD,),
        )
        row = cur.fetchone()
        total = row[0] or 0
        reviewed = row[1] or 0
        confirmed = row[2] or 0
        fp = row[3] or 0

        pct_reviewed = (reviewed / total * 100.0) if total > 0 else 0.0
        fp_rate = (fp / reviewed * 100.0) if reviewed > 0 else 0.0

        return {
            "total_alerts": total,
            "reviewed_count": reviewed,
            "pct_reviewed": round(pct_reviewed, 1),
            "confirmed_count": confirmed,
            "false_positive_count": fp,
            "false_positive_rate": round(fp_rate, 1),
        }
    finally:
        conn.close()


def get_account_history(account_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve other transactions in the database involving an account (as origin or destination).

    Args:
        account_id: Account ID (e.g. 'C123456789')
        limit: Maximum number of transactions to return

    Returns:
        List of transaction records.
    """
    conn = db_manager.get_connection()
    ph = _get_placeholder(db_manager.backend)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                t.txn_id, t.account_id, t.dest_id, t.amount, t.type, t.ts,
                s.blended_score, s.xgb_score, s.gnn_score
            FROM transactions t
            LEFT JOIN scores s ON t.txn_id = s.txn_id
            WHERE t.account_id = {ph} OR t.dest_id = {ph}
            ORDER BY t.ts DESC
            LIMIT {ph};
            """,
            (account_id, account_id, limit),
        )
        rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for r in rows:
            if hasattr(r, "keys"):
                results.append(dict(r))
            else:
                col_names = [d[0] for d in cur.description]
                results.append(dict(zip(col_names, r)))
        return results
    finally:
        conn.close()


def get_feature_contributions(
    raw_payload: Dict[str, Any],
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    """Compute top N feature contributions for a transaction using XGBoost tree SHAP (pred_contribs).

    Args:
        raw_payload: Transaction raw payload dictionary
        top_n: Number of top features to return

    Returns:
        List of dicts with feature name and contribution value.
    """
    import joblib
    import xgboost as xgb

    from src.serving.scoring import DEFAULT_FEATURE_VALUES

    # Prepare single row DataFrame matching training preprocessing input schema
    row = {
        "type": raw_payload.get("type", "TRANSFER"),
        "amount": float(raw_payload.get("amount", 0.0)),
        "oldbalanceOrg": float(raw_payload.get("oldbalanceOrg", 0.0)),
        "oldbalanceDest": float(raw_payload.get("oldbalanceDest", 0.0)),
        **DEFAULT_FEATURE_VALUES,
    }
    # If raw_payload contains feast features, use them
    for k in DEFAULT_FEATURE_VALUES:
        if k in raw_payload:
            row[k] = raw_payload[k]

    df = pd.DataFrame([row])

    preproc_path = settings.PREPROCESSOR_LOCAL_PATH
    preprocessor = joblib.load(preproc_path)
    X_trans = preprocessor.transform(df)
    feature_names = list(preprocessor.get_feature_names_out())

    booster = xgb.Booster()
    booster.load_model(settings.MODEL_LOCAL_PATH)

    dmat = xgb.DMatrix(X_trans, feature_names=feature_names)
    contribs = booster.predict(dmat, pred_contribs=True)
    # contribs has shape (1, num_features + 1), where last column is bias
    vals = contribs[0, :-1]

    items = [
        {
            "feature": feat,
            "contribution": float(val),
            "abs_contribution": abs(float(val)),
        }
        for feat, val in zip(feature_names, vals)
    ]
    items.sort(key=lambda x: x["abs_contribution"], reverse=True)
    return items[:top_n]
