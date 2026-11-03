"""Drift monitoring pipeline for FraudShield using Evidently.

Monitors statistical data drift across key features, transaction type mix,
and prediction score distribution between training reference window and serving/test window.
Outputs timestamped HTML reports and machine-readable JSON summaries.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from evidently import Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.common.db import db_manager  # noqa: E402
from src.common.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def load_reference_data(
    reference_path: Optional[str] = None,
    max_rows: int = 10000,
) -> pd.DataFrame:
    """Load reference dataset from training window features and compute reference predictions.

    Args:
        reference_path: Path to reference features parquet file.
        max_rows: Maximum rows to sample for fast drift baseline calculation.

    Returns:
        DataFrame with features and computed 'fraud_score'.
    """
    path = Path(reference_path or settings.TRAIN_FEATURES_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Reference features not found at {path}")

    logger.info("Loading training reference window from %s (max_rows=%d)", path, max_rows)
    df = pd.read_parquet(path)
    if "step" in df.columns:
        df = df[df["step"] <= settings.TRAIN_MAX_STEP]

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    # Compute reference fraud scores if not present
    if "fraud_score" not in df.columns:
        df["fraud_score"] = _compute_predictions(df)

    return df


def load_current_data(
    data_path: Optional[str] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    last_n: Optional[int] = 5000,
    from_db: bool = False,
) -> pd.DataFrame:
    """Load current evaluation slice from database or test features parquet.

    Args:
        data_path: Path to current evaluation parquet.
        start_step: Optional minimum simulation step.
        end_step: Optional maximum simulation step.
        last_n: Number of most recent records to sample.
        from_db: If True, query recent records from serving database.

    Returns:
        DataFrame with features and computed 'fraud_score'.
    """
    if from_db:
        logger.info("Loading current window from serving database (last_n=%s)", last_n)
        conn = db_manager.get_connection()
        limit = last_n or 5000
        query = f"""
            SELECT
                t.amount, t.type, t.raw, s.blended_score AS fraud_score
            FROM (
                SELECT txn_id, MAX(id) AS max_id FROM scores GROUP BY txn_id
            ) latest
            JOIN scores s ON latest.txn_id = s.txn_id AND latest.max_id = s.id
            LEFT JOIN transactions t ON s.txn_id = t.txn_id
            ORDER BY s.id DESC
            LIMIT {limit};
        """
        db_df = pd.read_sql_query(query, conn)
        conn.close()

        if db_df.empty:
            logger.warning("Serving database is empty; falling back to parquet evaluation slice.")
            return load_current_data(data_path=data_path, last_n=last_n, from_db=False)

        # Unpack raw payload if available
        rows = []
        for _, row in db_df.iterrows():
            rec = {"amount": row["amount"], "type": row["type"], "fraud_score": row["fraud_score"]}
            raw = row.get("raw")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            if isinstance(raw, dict):
                rec["oldbalanceOrg"] = raw.get("oldbalanceOrg", 0.0)
                rec["oldbalanceDest"] = raw.get("oldbalanceDest", 0.0)
                rec["orig_txn_count_1h"] = raw.get("orig_txn_count_1h", 1.0)
                rec["orig_txn_count_24h"] = raw.get("orig_txn_count_24h", 1.0)
                rec["orig_amount_vs_mean"] = raw.get("orig_amount_vs_mean", 1.0)
                rec["orig_time_since_last_txn"] = raw.get("orig_time_since_last_txn", 86400.0)
                rec["orig_has_prev_txn"] = raw.get("orig_has_prev_txn", 0.0)
                rec["dest_txn_count_24h"] = raw.get("dest_txn_count_24h", 0.0)
                rec["dest_has_prev_txn"] = raw.get("dest_has_prev_txn", 0.0)
            rows.append(rec)
        return pd.DataFrame(rows)

    path = Path(data_path or settings.TEST_FEATURES_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Current evaluation slice not found at {path}")

    logger.info("Loading current window from parquet %s", path)
    df = pd.read_parquet(path)

    if start_step is not None:
        df = df[df["step"] >= start_step]
    if end_step is not None:
        df = df[df["step"] <= end_step]

    if last_n is not None and len(df) > last_n:
        df = df.iloc[-last_n:].reset_index(drop=True)

    if "fraud_score" not in df.columns:
        df["fraud_score"] = _compute_predictions(df)

    return df


def _compute_predictions(df: pd.DataFrame) -> np.ndarray:
    """Batch compute predictions using local XGBoost preprocessor and model."""
    try:
        preprocessor = joblib.load(settings.PREPROCESSOR_LOCAL_PATH)
        booster = xgb.Booster()
        booster.load_model(settings.MODEL_LOCAL_PATH)

        feature_names = list(preprocessor.get_feature_names_out())
        X_trans = preprocessor.transform(df)
        dmat = xgb.DMatrix(X_trans, feature_names=feature_names)
        return booster.predict(dmat)
    except Exception as e:
        logger.warning("Could not compute live batch predictions: %s. Using default 0.0.", e)
        return np.zeros(len(df))


def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features_to_monitor: Optional[List[str]] = None,
    drift_share_threshold: Optional[float] = None,
    output_dir: Optional[str] = None,
    save_report: bool = True,
) -> Dict[str, Any]:
    """Execute Evidently data drift analysis and produce HTML report & JSON summary.

    Args:
        reference_df: Baseline training window DataFrame.
        current_df: Recent serving/evaluation window DataFrame.
        features_to_monitor: Explicit list of features to evaluate (defaults to settings.DRIFT_FEATURES).
        drift_share_threshold: Proportion of drifted features required to flag overall drift.
        output_dir: Directory where HTML and JSON artifacts are saved.
        save_report: Whether to persist HTML and JSON reports to disk.

    Returns:
        Structured drift results dictionary.
    """
    features = list(features_to_monitor or settings.DRIFT_FEATURES)
    threshold = float(drift_share_threshold or settings.DRIFT_SHARE_THRESHOLD)

    # Filter available columns
    available_cols = [c for c in features if c in reference_df.columns and c in current_df.columns]
    if not available_cols:
        raise ValueError(f"None of the target features {features} are present in datasets.")

    ref_slice = reference_df[available_cols].dropna().copy()
    cur_slice = current_df[available_cols].dropna().copy()

    logger.info(
        "Running Evidently DataDriftPreset on %d features (ref_rows=%d, cur_rows=%d)...",
        len(available_cols),
        len(ref_slice),
        len(cur_slice),
    )

    report = Report(metrics=[DataDriftPreset(drift_share=threshold)])
    snapshot = report.run(reference_data=ref_slice, current_data=cur_slice)

    # Parse results from snapshot dictionary
    snap_dict = snapshot.dict()
    metrics_list = snap_dict.get("metrics", [])

    drift_share = 0.0
    drifted_count = 0
    drift_by_feature: Dict[str, Dict[str, Any]] = {}
    prediction_drift = False

    if metrics_list:
        # First metric is DriftedColumnsCount
        count_metric = metrics_list[0].get("value", {})
        if isinstance(count_metric, dict):
            drift_share = float(count_metric.get("share", 0.0))
            drifted_count = int(count_metric.get("count", 0))

        # Subsequent metrics are ValueDrift per column
        for m in metrics_list[1:]:
            cfg = m.get("config", {})
            col = cfg.get("column")
            if not col:
                continue

            pval = float(m.get("value", 1.0))
            p_threshold = float(cfg.get("threshold", settings.DRIFT_P_VALUE_THRESHOLD))
            method = cfg.get("method", "K-S p_value")

            # Determine drift based on metric type:
            # - For p-value tests (K-S, chi-square): drift is detected when p_value < threshold
            # - For distance metrics (Wasserstein, Jensen-Shannon): drift is detected when distance >= threshold
            is_p_val_test = any(k in method.lower() for k in ("p_value", "p-value", "pvalue"))
            if is_p_val_test:
                is_col_drifted = bool(pval < p_threshold)
            else:
                is_col_drifted = bool(pval >= p_threshold)

            drift_by_feature[col] = {
                "metric_value": pval,
                "p_value": pval,
                "p_threshold": p_threshold,
                "threshold": p_threshold,
                "method": method,
                "drift_detected": is_col_drifted,
            }

            if col == "fraud_score" and is_col_drifted:
                prediction_drift = True

    # Sensitive score check for imbalanced fraud scores:
    # If scores are mostly near zero, statistical distance can be muted; check elevated score rate
    if "fraud_score" in ref_slice.columns and "fraud_score" in cur_slice.columns:
        ref_elevated = float((ref_slice["fraud_score"] >= 0.10).mean())
        cur_elevated = float((cur_slice["fraud_score"] >= 0.10).mean())
        if cur_elevated > 0.01 and (cur_elevated >= 2.0 * max(ref_elevated, 0.001) or abs(cur_elevated - ref_elevated) > 0.02):
            prediction_drift = True
            logger.info(
                "Sensitive prediction drift detected: elevated score rate (>=0.10) shifted from %.4f to %.4f",
                ref_elevated,
                cur_elevated,
            )

    overall_drift = bool(drift_share >= threshold or prediction_drift)

    now_iso = datetime.now(timezone.utc).isoformat()
    now_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    html_path = None
    json_path = None

    if save_report:
        out_path = Path(output_dir or settings.DRIFT_REPORT_DIR)
        out_path.mkdir(parents=True, exist_ok=True)

        html_file = out_path / f"drift_report_{now_slug}.html"
        json_file = out_path / f"drift_summary_{now_slug}.json"

        snapshot.save_html(str(html_file))
        html_path = str(html_file)

        summary_payload = {
            "timestamp": now_iso,
            "reference_count": len(ref_slice),
            "current_count": len(cur_slice),
            "drift_share_threshold": threshold,
            "drifted_features_count": drifted_count,
            "share_of_drifted_features": round(drift_share, 4),
            "overall_drift_detected": overall_drift,
            "prediction_drift": prediction_drift,
            "drift_by_feature": drift_by_feature,
            "report_html_path": html_path,
        }

        with open(json_file, "w") as f:
            json.dump(summary_payload, f, indent=2)
        json_path = str(json_file)

        logger.info(
            "Saved drift report to %s and summary to %s (overall_drift=%s)",
            html_file,
            json_file,
            overall_drift,
        )

    return {
        "timestamp": now_iso,
        "reference_count": len(ref_slice),
        "current_count": len(cur_slice),
        "drift_share_threshold": threshold,
        "drifted_features_count": drifted_count,
        "share_of_drifted_features": round(drift_share, 4),
        "drift_share": round(drift_share, 4),
        "drift_detected": overall_drift,
        "overall_drift": overall_drift,
        "prediction_drift": prediction_drift,
        "drift_by_feature": drift_by_feature,
        "report_html_path": html_path,
        "summary_json_path": json_path,
        "html_report_path": html_path,
        "json_summary_path": json_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Evidently data and prediction drift monitoring.")
    parser.add_argument("--last-n", type=int, default=5000, help="Number of recent records to evaluate")
    parser.add_argument("--start-step", type=int, default=None, help="Filter current slice by start simulation step")
    parser.add_argument("--end-step", type=int, default=None, help="Filter current slice by end simulation step")
    parser.add_argument("--data-path", type=str, default=None, help="Path to evaluation parquet file")
    parser.add_argument("--from-db", action="store_true", help="Load current window from database instead of parquet")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save reports")
    args = parser.parse_args()

    ref_df = load_reference_data(max_rows=10000)
    cur_df = load_current_data(
        data_path=args.data_path,
        start_step=args.start_step,
        end_step=args.end_step,
        last_n=args.last_n,
        from_db=args.from_db,
    )

    results = run_drift_analysis(
        reference_df=ref_df,
        current_df=cur_df,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 65)
    print(" FRAUDSHIELD DRIFT MONITORING SUMMARY")
    print("=" * 65)
    print(f" Reference Samples     : {results['reference_count']:,}")
    print(f" Current Samples       : {results['current_count']:,}")
    print(f" Drift Share Threshold : {results['drift_share_threshold']:.1%}")
    print(f" Drifted Features      : {results['drifted_features_count']} ({results['share_of_drifted_features']:.1%})")
    print(f" Prediction Drift Flag : {results['prediction_drift']}")
    print(f" Overall Drift Verdict : {'[DRIFT DETECTED]' if results['drift_detected'] else '[STABLE (NO DRIFT)]'}")
    print("-" * 65)
    print(" Feature-Level P-Values:")
    for feat, det in results["drift_by_feature"].items():
        status = "DRIFTED" if det["drift_detected"] else "OK"
        print(f"   - {feat:24s}: p={det['p_value']:.4e} ({status})")
    print("-" * 65)
    print(f" HTML Report Path      : {results['report_html_path']}")
    print(f" JSON Summary Path     : {results['summary_json_path']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
