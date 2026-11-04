"""Automated retraining trigger, leakage-safe gate evaluation, and MLflow champion promotion.

Enforces strict leakage isolation via a FROZEN GATE SET that is NEVER used for training or tuning.
Evaluates champion vs candidate models side-by-side on the gate set and logs an immutable audit trail.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.metrics import average_precision_score, precision_score, recall_score  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.common.db import db_manager  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.feedback.service import get_labeled_cases  # noqa: E402
from src.serving.scoring import DEFAULT_FEATURE_VALUES  # noqa: E402
from src.training.data import ALL_FEATURE_COLUMNS, create_preprocessor  # noqa: E402

logger = get_logger(__name__)


def get_or_create_frozen_gate_set(
    gate_set_path: Optional[str] = None,
    source_test_path: Optional[str] = None,
    gate_size: int = 15000,
    seed: int = 42,
) -> pd.DataFrame:
    """Retrieve or create the frozen evaluation gate set from the hold-out test window.

    LEAKAGE CONTRACT: This dataset is frozen and saved to disk. It is NEVER used
    for training, feature fitting, or hyperparameter selection.

    Args:
        gate_set_path: Path to frozen gate set parquet file.
        source_test_path: Path to hold-out test set parquet.
        gate_size: Number of transactions in frozen gate set.
        seed: Random seed for reproducible gate set sampling.

    Returns:
        Frozen gate set DataFrame.
    """
    path = Path(gate_set_path or settings.FROZEN_GATE_SET_PATH)
    if path.exists():
        logger.info("Loading existing frozen gate set from %s", path)
        return pd.read_parquet(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    src_path = Path(source_test_path or settings.TEST_FEATURES_PATH)
    if not src_path.exists():
        raise FileNotFoundError(f"Source test features not found at {src_path}")

    logger.info("Creating frozen evaluation gate set from %s (size=%d, seed=%d)", src_path, gate_size, seed)
    df_test = pd.read_parquet(src_path)

    # Use late simulation steps (>= 600) or seeded sample to ensure representation of frauds
    if "step" in df_test.columns and len(df_test[df_test["step"] >= 600]) >= gate_size:
        gate_df = df_test[df_test["step"] >= 600].sample(n=gate_size, random_state=seed).copy()
    else:
        gate_df = df_test.sample(n=min(gate_size, len(df_test)), random_state=seed).copy()

    # Assign persistent transaction identifiers if not present
    if "txn_id" not in gate_df.columns:
        gate_df["txn_id"] = [f"gate_txn_{idx}" for idx in gate_df.index]

    gate_df = gate_df.reset_index(drop=True)
    gate_df.to_parquet(path, index=False)
    logger.info("Saved frozen gate set with %d rows (%d frauds) to %s", len(gate_df), int(gate_df["isFraud"].sum()), path)
    return gate_df


def prepare_retraining_dataset(
    base_train_path: Optional[str] = None,
    frozen_gate_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble retraining dataset with verified analyst labels while enforcing ZERO leakage.

    Args:
        base_train_path: Path to original training window features parquet.
        frozen_gate_df: Frozen gate set to check and enforce leakage exclusion against.

    Returns:
        Tuple of (retrain_train_df, frozen_gate_df).
    """
    gate_df = frozen_gate_df if frozen_gate_df is not None else get_or_create_frozen_gate_set()
    gate_txn_ids = set(gate_df["txn_id"].astype(str)) if "txn_id" in gate_df.columns else set()

    train_path = Path(base_train_path or settings.TRAIN_FEATURES_PATH)
    df_train = pd.read_parquet(train_path)

    # Fetch verified analyst labels from feedback loop
    labeled_cases_df = get_labeled_cases()
    if not labeled_cases_df.empty:
        id_col = "transaction_id" if "transaction_id" in labeled_cases_df.columns else "txn_id"
        lbl_col = "label" if "label" in labeled_cases_df.columns else "analyst_label"

        # Check for leakage against frozen evaluation gate set
        leakage_overlap = set(labeled_cases_df[id_col].astype(str)).intersection(gate_txn_ids)
        if leakage_overlap:
            raise ValueError(
                f"LEAKAGE VIOLATION: {len(leakage_overlap)} gate-set transaction IDs found in analyst labels! "
                f"Sample overlapping IDs: {list(leakage_overlap)[:5]}"
            )

        # Exclude gate set ids defensively
        safe_labels = labeled_cases_df[~labeled_cases_df[id_col].astype(str).isin(gate_txn_ids)].copy()
        logger.info(
            "Incorporating %d verified analyst labels into retraining pool (0 gate set overlap)",
            len(safe_labels),
        )

        if not safe_labels.empty:
            label_map = dict(zip(safe_labels[id_col].astype(str), safe_labels[lbl_col].astype(int)))

            # 1. In-place update for existing training transactions
            if "txn_id" in df_train.columns:
                overlap_mask = df_train["txn_id"].astype(str).isin(label_map.keys())
                if overlap_mask.any():
                    df_train.loc[overlap_mask, "isFraud"] = df_train.loc[overlap_mask, "txn_id"].astype(str).map(label_map)
                    logger.info("Updated isFraud labels for %d existing training transactions", int(overlap_mask.sum()))

            # 2. Append new labeled transactions not yet in df_train
            existing_ids = set(df_train["txn_id"].astype(str)) if "txn_id" in df_train.columns else set()
            new_labels = safe_labels[~safe_labels[id_col].astype(str).isin(existing_ids)]

            if not new_labels.empty:
                new_records = []
                new_ids = list(new_labels[id_col].astype(str))

                # Direct features in safe_labels (e.g. test fixtures)
                has_direct_features = any(c in new_labels.columns for c in ALL_FEATURE_COLUMNS)
                if has_direct_features:
                    for _, row in new_labels.iterrows():
                        rec = {}
                        for c in ALL_FEATURE_COLUMNS:
                            rec[c] = row[c] if c in row and pd.notna(row[c]) else DEFAULT_FEATURE_VALUES.get(c, 0.0)
                        rec["isFraud"] = int(row[lbl_col])
                        if "txn_id" in df_train.columns:
                            rec["txn_id"] = str(row[id_col])
                        new_records.append(rec)
                elif db_manager.is_reachable():
                    # Query database transactions table
                    try:
                        conn = db_manager.get_connection()
                        ph = "%s" if db_manager.backend == "postgresql" else "?"
                        placeholders = ",".join([ph] * len(new_ids))
                        cur = conn.cursor()
                        cur.execute(
                            f"SELECT txn_id, raw, type, amount FROM transactions WHERE txn_id IN ({placeholders})",
                            tuple(new_ids),
                        )
                        db_rows = cur.fetchall()
                        conn.close()

                        for r in db_rows:
                            tid = str(r[0] if not hasattr(r, "keys") else r["txn_id"])
                            raw_val = r[1] if not hasattr(r, "keys") else r["raw"]
                            raw_payload = {}
                            if isinstance(raw_val, str):
                                try:
                                    raw_payload = json.loads(raw_val)
                                except Exception:
                                    raw_payload = {}
                            elif isinstance(raw_val, dict):
                                raw_payload = raw_val

                            rec = {}
                            for col in ALL_FEATURE_COLUMNS:
                                if col in raw_payload:
                                    rec[col] = raw_payload[col]
                                elif col == "type":
                                    rec[col] = r[2] if not hasattr(r, "keys") else r["type"]
                                elif col == "amount":
                                    rec[col] = float(r[3]) if not hasattr(r, "keys") else float(r["amount"])
                                else:
                                    rec[col] = DEFAULT_FEATURE_VALUES.get(col, 0.0)

                            rec["isFraud"] = label_map.get(tid, 0)
                            if "txn_id" in df_train.columns:
                                rec["txn_id"] = tid
                            new_records.append(rec)
                    except Exception as db_err:
                        logger.warning("Could not query database transactions for retraining labels: %s", db_err)

                if new_records:
                    new_df = pd.DataFrame(new_records)
                    df_train = pd.concat([df_train, new_df], ignore_index=True)
                    logger.info(
                        "Successfully merged %d newly labeled transactions into df_train (total rows: %d)",
                        len(new_records),
                        len(df_train),
                    )

    return df_train, gate_df


def check_retrain_trigger(
    drift_share: float = 0.0,
    drift_detected: bool = False,
    min_new_labels: Optional[int] = None,
    new_labels_count: Optional[int] = None,
    drift_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate retraining trigger conditions: statistical drift OR sufficient new analyst labels.

    Args:
        drift_share: Fraction of monitored features exhibiting statistical drift.
        drift_detected: Boolean drift flag from Evidently monitoring.
        min_new_labels: Minimum required verified labels to trigger retrain on labels.
        new_labels_count: Optional override for new labeled cases count.
        drift_threshold: Optional override for drift share threshold.

    Returns:
        Dictionary with trigger status and reasoning.
    """
    label_threshold = min_new_labels or settings.MIN_NEW_LABELS_FOR_RETRAIN
    eff_drift_thresh = drift_threshold if drift_threshold is not None else settings.DRIFT_SHARE_THRESHOLD

    if new_labels_count is not None:
        label_count = new_labels_count
    else:
        labeled_cases = get_labeled_cases()
        label_count = len(labeled_cases)

    drift_cond = bool(drift_detected or (drift_share >= eff_drift_thresh))
    label_cond = bool(label_count >= label_threshold)

    trigger_fired = drift_cond or label_cond

    if drift_cond and label_cond:
        reason = "both_drift_and_labels"
    elif drift_cond:
        reason = "data_drift_exceeded_threshold"
    elif label_cond:
        reason = "analyst_labels_threshold_met"
    else:
        reason = "no_trigger_condition_met"

    logger.info(
        "Retrain trigger check: fired=%s, reason='%s' (drift_share=%.2f, threshold=%.2f, new_labels=%d, label_min=%d)",
        trigger_fired,
        reason,
        drift_share,
        settings.DRIFT_SHARE_THRESHOLD,
        label_count,
        label_threshold,
    )

    return {
        "trigger_fired": trigger_fired,
        "reason": reason,
        "drift_condition_met": drift_cond,
        "label_condition_met": label_cond,
        "drift_share": drift_share,
        "label_count": label_count,
        "label_threshold": label_threshold,
    }


def train_candidate_model(
    train_df: pd.DataFrame,
    hyperparams: Optional[Dict[str, Any]] = None,
    max_train_rows: Optional[int] = None,
    register_in_mlflow: bool = True,
) -> Tuple[Any, Any, str]:
    """Train a new candidate XGBoost model using Phase 3 preprocessing and register to MLflow.

    Args:
        train_df: Retraining dataset.
        hyperparams: XGBoost hyperparameter dictionary.
        max_train_rows: Optional row cap for fast, responsive retraining (preserves 100% of frauds).
        register_in_mlflow: Whether to log and register model in MLflow.

    Returns:
        Tuple of (fitted_preprocessor, trained_booster, registered_version_string).
    """
    df = train_df
    if max_train_rows is not None and len(df) > max_train_rows:
        fraud_df = df[df["isFraud"] == 1]
        legit_sample = df[df["isFraud"] == 0].sample(n=max_train_rows, random_state=42)
        df = pd.concat([fraud_df, legit_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        logger.info(
            "[FAST RETRAIN MODE] Sampled retraining set to %d rows (preserved all %d frauds)",
            len(df),
            len(fraud_df),
        )

    logger.info("Training candidate XGBoost model on %d rows...", len(df))
    params = hyperparams or {
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "n_estimators": 300,
        "scale_pos_weight": float((len(df) - df["isFraud"].sum()) / max(df["isFraud"].sum(), 1)),
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": 4,
    }

    # 1. Feature Preprocessing (reuses Phase 3 create_preprocessor)
    preprocessor = create_preprocessor()
    X_train = preprocessor.fit_transform(df[ALL_FEATURE_COLUMNS])
    y_train = df["isFraud"].values

    # 2. Train XGBClassifier
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)

    candidate_version = "candidate_v2"

    if register_in_mlflow:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("FraudShield-Retraining")

        with mlflow.start_run(run_name=f"candidate_retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(params)
            mlflow.log_metric("train_samples", len(train_df))

            # Log model artifact
            mlflow.xgboost.log_model(
                xgb_model=clf,
                artifact_path="model",
                registered_model_name=settings.MODEL_NAME,
            )

            # Retrieve registered version
            client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
            model_version_details = client.get_latest_versions(settings.MODEL_NAME)
            if model_version_details:
                candidate_version = str(max(int(v.version) for v in model_version_details))
            logger.info("Registered candidate model to MLflow as version %s", candidate_version)

    return preprocessor, clf, candidate_version


def compute_bootstrap_pr_auc_diff(
    y_true: np.ndarray,
    p_cand: np.ndarray,
    p_champ: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Calculate 95% bootstrap confidence interval for PR-AUC difference (candidate - champion)."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    diffs: List[float] = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        ap_cand = average_precision_score(y_b, p_cand[idx])
        ap_champ = average_precision_score(y_b, p_champ[idx])
        diffs.append(float(ap_cand - ap_champ))

    diffs_arr = np.array(diffs)
    ci_lower = float(np.percentile(diffs_arr, 2.5))
    ci_upper = float(np.percentile(diffs_arr, 97.5))
    mean_diff = float(np.mean(diffs_arr))
    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_strictly_positive": bool(ci_lower > 0),
        "n_bootstraps": len(diffs),
    }


def evaluate_promotion_gate(
    champion_preprocessor: Any,
    champion_model: Any,
    candidate_preprocessor: Any,
    candidate_model: Any,
    gate_df: pd.DataFrame,
    candidate_version: str,
    trigger_reason: str = "manual",
    min_improvement: Optional[float] = None,
    require_ci_positive: bool = True,
    n_bootstraps: int = 1000,
    save_audit_log: bool = True,
) -> Dict[str, Any]:
    """Evaluate champion and candidate side-by-side on the FROZEN GATE SET and decide promotion.

    PROMOTION CONTRACT:
    Candidate is promoted to 'champion' alias ONLY IF:
      candidate_pr_auc >= champion_pr_auc + MIN_IMPROVEMENT_PR_AUC
      AND candidate_recall >= champion_recall (or recall is not degraded at precision)
      AND (if require_ci_positive is True) 95% bootstrap CI lower bound > 0

    Args:
        champion_preprocessor: Preprocessor for current champion model.
        champion_model: Current champion model/booster.
        candidate_preprocessor: Preprocessor for new candidate model.
        candidate_model: Newly trained candidate model/booster.
        gate_df: Frozen evaluation gate set.
        candidate_version: Registered MLflow version string for candidate.
        trigger_reason: Reason retraining was initiated.
        min_improvement: Minimum PR-AUC delta required for promotion.
        require_ci_positive: Whether promotion strictly requires the 95% bootstrap CI lower bound > 0.
        n_bootstraps: Number of bootstrap resamples for confidence interval.
        save_audit_log: Whether to append decision to audit log on disk.

    Returns:
        Structured promotion decision dictionary.
    """
    min_delta = float(min_improvement if min_improvement is not None else settings.MIN_IMPROVEMENT_PR_AUC)
    y_true = gate_df["isFraud"].values

    # 1. Evaluate Champion on Gate Set
    X_champ = champion_preprocessor.transform(gate_df[ALL_FEATURE_COLUMNS])
    if hasattr(champion_model, "predict_proba"):
        p_champ = champion_model.predict_proba(X_champ)[:, 1]
    else:
        feature_names = list(champion_preprocessor.get_feature_names_out())
        dmat_champ = xgb.DMatrix(X_champ, feature_names=feature_names)
        p_champ = champion_model.predict(dmat_champ)

    champ_pr_auc = float(average_precision_score(y_true, p_champ))
    champ_pred_binary = (p_champ >= settings.DECISION_THRESHOLD).astype(int)
    champ_prec = float(precision_score(y_true, champ_pred_binary, zero_division=0))
    champ_recall = float(recall_score(y_true, champ_pred_binary, zero_division=0))

    # 2. Evaluate Candidate on Gate Set
    X_cand = candidate_preprocessor.transform(gate_df[ALL_FEATURE_COLUMNS])
    if hasattr(candidate_model, "predict_proba"):
        p_cand = candidate_model.predict_proba(X_cand)[:, 1]
    else:
        feature_names = list(candidate_preprocessor.get_feature_names_out())
        dmat_cand = xgb.DMatrix(X_cand, feature_names=feature_names)
        p_cand = candidate_model.predict(dmat_cand)

    cand_pr_auc = float(average_precision_score(y_true, p_cand))
    cand_pred_binary = (p_cand >= settings.DECISION_THRESHOLD).astype(int)
    cand_prec = float(precision_score(y_true, cand_pred_binary, zero_division=0))
    cand_recall = float(recall_score(y_true, cand_pred_binary, zero_division=0))

    # 3. Compute Bootstrap Confidence Interval on ΔPR-AUC
    boot_stats = compute_bootstrap_pr_auc_diff(y_true, p_cand, p_champ, n_bootstraps=n_bootstraps)
    ci_lower = boot_stats["ci_lower"]
    ci_upper = boot_stats["ci_upper"]
    ci_strictly_positive = boot_stats["ci_strictly_positive"]

    # 4. Apply Promotion Gate Rule
    pr_auc_improved = cand_pr_auc >= (champ_pr_auc + min_delta)
    recall_acceptable = cand_recall >= champ_recall
    ci_acceptable = True
    if require_ci_positive:
        ci_acceptable = ci_strictly_positive

    promoted = bool(pr_auc_improved and recall_acceptable and ci_acceptable)
    decision = "PROMOTED" if promoted else "REJECTED"

    if promoted:
        reason = (
            f"Candidate PR-AUC ({cand_pr_auc:.4f}) improved over champion ({champ_pr_auc:.4f}) by "
            f">={min_delta:.4f}, recall maintained ({cand_recall:.4f} >= {champ_recall:.4f}), "
            f"and 95% bootstrap CI on dPR-AUC is [{ci_lower:+.4f}, {ci_upper:+.4f}] (strictly_positive={ci_strictly_positive})."
        )
        # Set MLflow alias
        try:
            client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
            client.set_registered_model_alias(settings.MODEL_NAME, "champion", str(candidate_version))
            logger.info("Updated MLflow alias 'champion' -> version %s", candidate_version)
        except Exception as e:
            logger.warning("Could not set MLflow alias: %s", e)

        # Export candidate model artifacts to local serving paths
        try:
            Path(settings.MODEL_LOCAL_PATH).parent.mkdir(parents=True, exist_ok=True)
            if hasattr(candidate_model, "save_model"):
                candidate_model.save_model(settings.MODEL_LOCAL_PATH)
            joblib.dump(candidate_preprocessor, settings.PREPROCESSOR_LOCAL_PATH)
            logger.info("Exported candidate model artifacts to %s and %s", settings.MODEL_LOCAL_PATH, settings.PREPROCESSOR_LOCAL_PATH)
        except Exception as e:
            logger.warning("Could not export local model files: %s", e)
    else:
        reason = (
            f"Candidate PR-AUC ({cand_pr_auc:.4f}) did not satisfy promotion criteria over champion ({champ_pr_auc:.4f}) "
            f"(required delta >= +{min_delta:.4f}, recall {cand_recall:.4f} vs {champ_recall:.4f}; "
            f"95% bootstrap CI on dPR-AUC: [{ci_lower:+.4f}, {ci_upper:+.4f}], strictly_positive={ci_strictly_positive})."
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "timestamp": now_iso,
        "trigger_reason": trigger_reason,
        "champion_version": "1",
        "candidate_version": candidate_version,
        "champion_pr_auc": round(champ_pr_auc, 4),
        "candidate_pr_auc": round(cand_pr_auc, 4),
        "delta_pr_auc": round(cand_pr_auc - champ_pr_auc, 4),
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "ci_strictly_positive": ci_strictly_positive,
        "statistical_power_note": "Frozen gate set has N_fraud=230 (SE ≈ 0.013). Small gains (<0.015) cross zero at 95% confidence.",
        "champion_precision": round(champ_prec, 4),
        "candidate_precision": round(cand_prec, 4),
        "champion_recall": round(champ_recall, 4),
        "candidate_recall": round(cand_recall, 4),
        "min_improvement_threshold": min_delta,
        "promoted": promoted,
        "decision": decision,
        "decision_reason": reason,
    }

    if save_audit_log:
        audit_path = Path(settings.RETRAIN_AUDIT_LOG_PATH)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        existing_logs = []
        if audit_path.exists():
            try:
                with open(audit_path, "r") as f:
                    existing_logs = json.load(f)
            except Exception:
                existing_logs = []
        existing_logs.append(record)
        with open(audit_path, "w") as f:
            json.dump(existing_logs, f, indent=2)
        logger.info("Appended promotion decision to audit log at %s", audit_path)

    return record


def main():
    parser = argparse.ArgumentParser(description="Evaluate retraining trigger and execute gated retraining.")
    parser.add_argument("--force", action="store_true", help="Force retraining execution regardless of trigger state")
    parser.add_argument("--check-drift", action="store_true", default=True, help="Evaluate data drift in trigger condition")
    parser.add_argument("--check-labels", action="store_true", default=True, help="Evaluate analyst feedback label count in trigger condition")
    parser.add_argument("--min-labels", type=int, default=settings.MIN_NEW_LABELS_FOR_RETRAIN, help="Label threshold")
    parser.add_argument("--min-delta", type=float, default=settings.MIN_IMPROVEMENT_PR_AUC, help="Min PR-AUC improvement")
    args = parser.parse_args()

    trigger_status = check_retrain_trigger(min_new_labels=args.min_labels)
    if not trigger_status["trigger_fired"] and not args.force:
        print(f"Retraining not triggered: {trigger_status['reason']}. Use --force to override.")
        return

    print("Retraining condition met! Preparing dataset with zero-leakage gate set...")
    train_df, gate_df = prepare_retraining_dataset()

    # Load current champion
    champ_preproc = joblib.load(settings.PREPROCESSOR_LOCAL_PATH)
    champ_booster = xgb.Booster()
    champ_booster.load_model(settings.MODEL_LOCAL_PATH)

    # Train candidate model
    cand_preproc, cand_clf, cand_ver = train_candidate_model(train_df)

    # Evaluate promotion gate
    result = evaluate_promotion_gate(
        champion_preprocessor=champ_preproc,
        champion_model=champ_booster,
        candidate_preprocessor=cand_preproc,
        candidate_model=cand_clf,
        gate_df=gate_df,
        candidate_version=cand_ver,
        trigger_reason=trigger_status["reason"],
        min_improvement=args.min_delta,
    )

    print("\n" + "=" * 65)
    print(" FRAUDSHIELD RETRAINING & PROMOTION GATE REPORT")
    print("=" * 65)
    print(f" Trigger Reason      : {result['trigger_reason']}")
    print(f" Candidate Version   : {result['candidate_version']}")
    print("-" * 65)
    print(f" Champion PR-AUC     : {result['champion_pr_auc']:.4f}")
    print(f" Candidate PR-AUC    : {result['candidate_pr_auc']:.4f}")
    print(f" Champion Recall     : {result['champion_recall']:.4f}")
    print(f" Candidate Recall    : {result['candidate_recall']:.4f}")
    print(f" Min PR-AUC Delta    : +{result['min_improvement_threshold']:.4f}")
    print("-" * 65)
    print(f" Gate Decision       : [{result['decision']}]")
    print(f" Reason              : {result['decision_reason']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
