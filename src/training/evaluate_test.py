"""Hold-out test set evaluation for selected XGBoost model.

Evaluates the chosen model and decision threshold ONCE on the strictly held-out
test dataset (data/processed/test_features.parquet) after model selection is complete.
"""

import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import numpy as np
import psutil
import xgboost as xgb

from src.common.config import settings
from src.common.logging import get_logger
from src.training.data import (
    load_feature_split,
)
from src.training.evaluate import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_pr_curve,
)

logger = get_logger(__name__)


def run_test_evaluation(
    model_path: str = "models/xgboost_model.json",
    preprocessor_path: str = "models/xgboost_preprocessor.joblib",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate selected model on hold-out test set once.

    Args:
        model_path: Path to serialized XGBoost model json.
        preprocessor_path: Path to serialized ColumnTransformer.
        threshold: Decision threshold chosen during validation tuning.
                   If None, reads from models/xgboost_val_sweep.csv (max F1).

    Returns:
        Dictionary with hold-out test evaluation metrics and analysis.
    """
    logger.info("=" * 60)
    logger.info("STARTING STRICT HOLD-OUT TEST SET EVALUATION")
    logger.info("=" * 60)

    # 1. Load trained model & preprocessor
    model_p = Path(model_path)
    prep_p = Path(preprocessor_path)
    if not model_p.exists() or not prep_p.exists():
        raise FileNotFoundError(
            f"Trained model artifacts missing: {model_p} or {prep_p}. "
            "Please run train_xgb.py first!"
        )

    model = xgb.XGBClassifier()
    model.load_model(str(model_p))
    preprocessor = joblib.load(prep_p)

    # 2. Determine selected threshold from validation sweep if not specified
    if threshold is None:
        val_sweep_path = Path("models/xgboost_val_sweep.csv")
        if val_sweep_path.exists():
            import pandas as pd
            sweep_df = pd.read_csv(val_sweep_path)
            best_idx = sweep_df["f1"].idxmax()
            threshold = float(sweep_df.iloc[best_idx]["threshold"])
            logger.info(
                "Read selected validation threshold: %.2f (Validation F1: %.4f)",
                threshold,
                float(sweep_df.iloc[best_idx]["f1"]),
            )
        else:
            threshold = 0.50
            logger.warning("Validation sweep not found. Defaulting threshold to 0.50")

    # 3. Load hold-out test split
    proc = psutil.Process()
    mem_before_mb = proc.memory_info().rss / (1024 * 1024)
    start_time = time.perf_counter()

    logger.info("Loading test split (data/processed/test_features.parquet)...")
    X_test_raw, y_test = load_feature_split("test")
    y_test_arr = y_test.to_numpy()

    test_samples = len(y_test_arr)
    test_fraud_cnt = int(y_test_arr.sum())
    test_legit_cnt = test_samples - test_fraud_cnt
    test_prevalence = float(test_fraud_cnt / test_samples)

    logger.info(
        "Hold-out Test set size: %d transactions (Fraud: %d, Legit: %d, Prevalence: %.4f%%)",
        test_samples,
        test_fraud_cnt,
        test_legit_cnt,
        test_prevalence * 100,
    )

    # 4. Preprocess test features using training-fitted preprocessor ONLY
    X_test = preprocessor.transform(X_test_raw)
    del X_test_raw
    gc.collect()

    # 5. Predict probabilities
    y_test_prob = model.predict_proba(X_test)[:, 1]
    eval_duration = time.perf_counter() - start_time
    mem_peak_mb = proc.memory_info().rss / (1024 * 1024)

    # Clean up test feature matrix
    del X_test
    gc.collect()

    # 6. Compute evaluation metrics
    default_metrics = evaluate_predictions(y_test_arr, y_test_prob, threshold=0.50)
    opt_metrics = evaluate_predictions(y_test_arr, y_test_prob, threshold=threshold)

    logger.info("HOLD-OUT TEST RESULTS:")
    logger.info("  Average Precision (PR-AUC): %.5f", opt_metrics.pr_auc)
    logger.info("  ROC-AUC:                     %.5f", opt_metrics.roc_auc)
    logger.info("  Selected Threshold:          %.2f", threshold)
    logger.info("  Precision at Threshold:      %.4f", opt_metrics.precision)
    logger.info("  Recall at Threshold:         %.4f", opt_metrics.recall)
    logger.info("  F1 Score at Threshold:       %.4f", opt_metrics.f1)
    logger.info(
        "  Confusion Matrix: TP=%d, FP=%d, FN=%d, TN=%d",
        opt_metrics.tp,
        opt_metrics.fp,
        opt_metrics.fn,
        opt_metrics.tn,
    )
    logger.info(
        "  Error Rates: False Positive Rate=%.4f%%, False Negative Rate=%.4f%%",
        opt_metrics.false_positive_rate * 100,
        opt_metrics.false_negative_rate * 100,
    )

    # 7. Generate test evaluation plots
    models_dir = Path("models")
    test_pr_curve_path = models_dir / "test_pr_curve.png"
    test_cm_path = models_dir / "test_confusion_matrix.png"

    plot_pr_curve(
        y_test_arr,
        y_test_prob,
        output_path=str(test_pr_curve_path),
        title=f"Hold-Out Test Precision-Recall Curve (AP = {opt_metrics.pr_auc:.4f})",
        chosen_threshold=threshold,
        chosen_precision=opt_metrics.precision,
        chosen_recall=opt_metrics.recall,
    )
    cm_test = np.array(
        [[opt_metrics.tn, opt_metrics.fp], [opt_metrics.fn, opt_metrics.tp]]
    )
    plot_confusion_matrix(
        cm_test,
        output_path=str(test_cm_path),
        title="Hold-Out Test Confusion Matrix",
        threshold=threshold,
    )

    # 8. Log test evaluation to MLflow
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-holdout-test-evaluation") as run:
        run_id = run.info.run_id
        logger.info("Logging hold-out test run to MLflow (run_id: %s)", run_id)

        mlflow.log_params(
            {
                "model_type": "XGBClassifier",
                "evaluation_split": "test",
                "test_dataset": "data/processed/test_features.parquet",
                "test_samples": test_samples,
                "test_fraud_samples": test_fraud_cnt,
                "test_prevalence_pct": round(test_prevalence * 100, 4),
                "selected_threshold": threshold,
                "threshold_origin": "validation_max_f1",
            }
        )

        mlflow.log_metrics(
            {
                "test_pr_auc": opt_metrics.pr_auc,
                "test_roc_auc": opt_metrics.roc_auc,
                "test_precision": opt_metrics.precision,
                "test_recall": opt_metrics.recall,
                "test_f1": opt_metrics.f1,
                "test_tp": opt_metrics.tp,
                "test_fp": opt_metrics.fp,
                "test_fn": opt_metrics.fn,
                "test_tn": opt_metrics.tn,
                "test_fpr": opt_metrics.false_positive_rate,
                "test_fnr": opt_metrics.false_negative_rate,
                "test_precision_at_0_5": default_metrics.precision,
                "test_recall_at_0_5": default_metrics.recall,
                "test_f1_at_0_5": default_metrics.f1,
                "eval_duration_sec": eval_duration,
                "mem_before_mb": mem_before_mb,
                "mem_peak_mb": mem_peak_mb,
            }
        )

        mlflow.log_artifact(str(test_pr_curve_path), artifact_path="test_evaluation")
        mlflow.log_artifact(str(test_cm_path), artifact_path="test_evaluation")

    return {
        "run_id": run_id,
        "selected_threshold": threshold,
        "default_metrics": default_metrics.to_dict(),
        "test_metrics": opt_metrics.to_dict(),
        "test_samples": test_samples,
        "test_fraud_cnt": test_fraud_cnt,
        "test_prevalence": test_prevalence,
        "eval_duration_sec": eval_duration,
        "mem_before_mb": mem_before_mb,
        "mem_peak_mb": mem_peak_mb,
    }


if __name__ == "__main__":
    results = run_test_evaluation()
    print("\n" + "=" * 60)
    print("HOLD-OUT TEST EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Test AP (PR-AUC): {results['test_metrics']['pr_auc']:.5f}")
    print(f"Test ROC-AUC:    {results['test_metrics']['roc_auc']:.5f}")
    print(f"Selected Threshold (from Val): {results['selected_threshold']:.2f}")
    print(
        f"Test Precision:  {results['test_metrics']['precision']:.4f} | "
        f"Recall: {results['test_metrics']['recall']:.4f} | "
        f"F1: {results['test_metrics']['f1']:.4f}"
    )
    print(
        f"Confusion Matrix: TP={results['test_metrics']['tp']}, "
        f"FP={results['test_metrics']['fp']}, "
        f"FN={results['test_metrics']['fn']}, "
        f"TN={results['test_metrics']['tn']}"
    )
    print(
        f"Prevalence: Train=0.0816%, Val=0.0575%, Test={results['test_prevalence']:.4%}"
    )
    print("=" * 60)
