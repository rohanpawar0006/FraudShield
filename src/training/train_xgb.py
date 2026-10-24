import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import psutil
import xgboost as xgb

from src.common.config import settings
from src.common.logging import get_logger
from src.training.data import (
    create_preprocessor,
    get_feature_names_out,
    load_feature_split,
)
from src.training.evaluate import (
    evaluate_predictions,
    find_optimal_thresholds,
    get_pr_curve_data,
    plot_confusion_matrix,
    plot_pr_curve,
    sweep_thresholds,
)

logger = get_logger(__name__)


# Hyperparameter search grid evaluated against validation PR-AUC
HYPERPARAMETER_CANDIDATES: List[Dict[str, Any]] = [
    {
        "name": "xgb_shallow_lr08",
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "n_estimators": 300,
    },
    {
        "name": "xgb_medium_lr08",
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "n_estimators": 300,
    },
    {
        "name": "xgb_medium_lr05",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
        "n_estimators": 300,
    },
]


def train_single_xgb(
    params: Dict[str, Any],
    scale_pos_weight: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[xgb.XGBClassifier, float, float]:
    """Train an XGBoost classifier with early stopping and evaluate validation PR-AUC.

    Args:
        params: Model hyperparameters dict.
        scale_pos_weight: Positive class weight (negative count / positive count).
        X_train: Preprocessed training features.
        y_train: Training labels.
        X_val: Preprocessed validation features.
        y_val: Validation labels.

    Returns:
        Tuple of (fitted model, validation PR-AUC, candidate duration in seconds).
    """
    clf_params = {
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "n_estimators": params["n_estimators"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "min_child_weight": params.get("min_child_weight", 1),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 20,
        "random_state": 42,
        "n_jobs": 4,
        "tree_method": "hist",
    }

    t0 = time.perf_counter()
    model = xgb.XGBClassifier(**clf_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    cand_duration = time.perf_counter() - t0

    val_probs = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_predictions(y_val, val_probs)
    return model, metrics.pr_auc, cand_duration


def run_xgboost_training() -> Dict[str, Any]:
    """Run full XGBoost hyperparameter tuning, evaluation, threshold selection, and MLflow registration.

    Returns:
        Dictionary containing tuning and evaluation summaries.
    """
    logger.info("Starting XGBoost training and tuning pipeline")
    proc = psutil.Process()
    mem_before_mb = proc.memory_info().rss / (1024 * 1024)
    start_total_time = time.perf_counter()

    # 1. Load splits with zero leakage assertion
    X_train_raw, y_train = load_feature_split("train")
    X_val_raw, y_val = load_feature_split("val")

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = float(n_neg / n_pos)

    logger.info(
        "Training samples: %d (Pos: %d, Neg: %d). Calculated scale_pos_weight: %.2f",
        len(X_train_raw),
        n_pos,
        n_neg,
        scale_pos_weight,
    )

    # 2. Fit preprocessing ONLY on training data
    logger.info(
        "Fitting preprocessor (OneHotEncoder + Passthrough) on training data..."
    )
    preprocessor = create_preprocessor(scale_numerical=False)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    feature_names = get_feature_names_out(preprocessor)

    # Clean up raw training and val frames to save memory
    del X_train_raw, X_val_raw
    gc.collect()

    # 3. Hyperparameter search over documented candidate configurations
    logger.info(
        "Evaluating %d hyperparameter candidates using validation PR-AUC...",
        len(HYPERPARAMETER_CANDIDATES),
    )

    best_model: Optional[xgb.XGBClassifier] = None
    best_params: Dict[str, Any] = {}
    best_pr_auc = -1.0
    tuning_records = []

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    for candidate in HYPERPARAMETER_CANDIDATES:
        logger.info(
            "Training candidate '%s' (max_depth=%d, lr=%.2f)...",
            candidate["name"],
            candidate["max_depth"],
            candidate["learning_rate"],
        )
        model, pr_auc, cand_duration = train_single_xgb(
            candidate,
            scale_pos_weight,
            X_train,
            y_train_arr,
            X_val,
            y_val_arr,
        )
        best_iteration = getattr(model, "best_iteration", candidate["n_estimators"])
        logger.info(
            "Candidate '%s' achieved Val PR-AUC: %.5f (Best iteration: %d, Duration: %.2fs)",
            candidate["name"],
            pr_auc,
            best_iteration,
            cand_duration,
        )
        tuning_records.append(
            {
                "candidate": candidate["name"],
                "max_depth": candidate["max_depth"],
                "learning_rate": candidate["learning_rate"],
                "subsample": candidate["subsample"],
                "colsample_bytree": candidate["colsample_bytree"],
                "best_iteration": best_iteration,
                "val_pr_auc": pr_auc,
                "duration_sec": round(cand_duration, 2),
            }
        )
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            if best_model is not None:
                del best_model
            best_model = model
            best_params = {**candidate, "best_iteration": best_iteration}
        else:
            del model
            gc.collect()

    assert best_model is not None, "Failed to train any XGBoost model"
    total_training_duration = time.perf_counter() - start_total_time
    mem_peak_mb = proc.memory_info().rss / (1024 * 1024)

    logger.info(
        "Selected best candidate: '%s' with Val PR-AUC: %.5f (Total tuning time: %.2fs, Peak RSS: %.1f MB)",
        best_params["name"],
        best_pr_auc,
        total_training_duration,
        mem_peak_mb,
    )

    # Clean up training data array - no longer needed for evaluation
    del X_train, y_train_arr
    gc.collect()

    # 4. Comprehensive validation evaluation and threshold selection
    val_probs = best_model.predict_proba(X_val)[:, 1]
    default_metrics = evaluate_predictions(y_val_arr, val_probs, threshold=0.5)

    # Sweep thresholds across [0.01 .. 0.99]
    sweep_df = sweep_thresholds(
        y_val_arr, val_probs, cost_fp=15.0, cost_fn=500.0
    )
    optimal_dict = find_optimal_thresholds(
        sweep_df, target_recalls=[0.80, 0.85, 0.90]
    )

    best_f1_record = optimal_dict["max_f1"]
    min_cost_record = optimal_dict["min_cost"]
    rec85_record = optimal_dict["recall_at_85"]

    opt_threshold_f1 = float(best_f1_record["threshold"])
    opt_threshold_cost = float(min_cost_record["threshold"])

    f1_metrics = evaluate_predictions(
        y_val_arr, val_probs, threshold=opt_threshold_f1
    )
    cost_metrics = evaluate_predictions(
        y_val_arr, val_probs, threshold=opt_threshold_cost
    )

    # Precision-Recall curve coordinates
    precisions_curve, recalls_curve, pr_thresholds = get_pr_curve_data(
        y_val_arr, val_probs
    )

    # 5. Local persistence & plot generation
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_json_path = models_dir / "xgboost_model.json"
    preprocessor_path = models_dir / "xgboost_preprocessor.joblib"
    sweep_csv_path = models_dir / "xgboost_val_sweep.csv"
    pr_curve_csv_path = models_dir / "xgboost_pr_curve.csv"
    pr_curve_img_path = models_dir / "xgboost_pr_curve.png"
    cm_img_path = models_dir / "xgboost_confusion_matrix.png"

    best_model.save_model(str(model_json_path))
    joblib.dump(preprocessor, preprocessor_path)
    sweep_df.to_csv(sweep_csv_path, index=False)

    # Save sampled PR curve data
    curve_df = pd.DataFrame(
        {
            "recall": recalls_curve[:-1],
            "precision": precisions_curve[:-1],
            "threshold": pr_thresholds,
        }
    )
    curve_df.to_csv(pr_curve_csv_path, index=False)

    # Generate PR curve and confusion matrix plots
    plot_pr_curve(
        y_val_arr,
        val_probs,
        output_path=str(pr_curve_img_path),
        title="XGBoost - Validation Precision-Recall Curve",
        chosen_threshold=opt_threshold_f1,
        chosen_precision=f1_metrics.precision,
        chosen_recall=f1_metrics.recall,
    )
    cm_f1 = np.array(
        [[f1_metrics.tn, f1_metrics.fp], [f1_metrics.fn, f1_metrics.tp]]
    )
    plot_confusion_matrix(
        cm_f1,
        output_path=str(cm_img_path),
        title="XGBoost Confusion Matrix",
        threshold=opt_threshold_f1,
    )

    # 6. MLflow Tracking
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-tuned") as run:
        run_id = run.info.run_id
        logger.info("Logging XGBoost run to MLflow (run_id: %s)", run_id)

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "XGBClassifier",
                "scale_pos_weight": scale_pos_weight,
                "max_depth": best_params["max_depth"],
                "learning_rate": best_params["learning_rate"],
                "best_iteration": best_params["best_iteration"],
                "n_estimators_max": best_params["n_estimators"],
                "subsample": best_params["subsample"],
                "colsample_bytree": best_params["colsample_bytree"],
                "min_child_weight": best_params.get("min_child_weight", 1),
                "eval_metric": "aucpr",
                "early_stopping_rounds": 20,
                "tree_method": "hist",
                "random_state": 42,
                "train_dataset": "data/processed/train_features.parquet",
                "val_dataset": "data/processed/val_features.parquet",
                "train_samples": len(y_train),
                "train_fraud_samples": n_pos,
                "train_legit_samples": n_neg,
                "val_samples": len(y_val),
                "val_fraud_samples": int(y_val.sum()),
                "feature_count": len(feature_names),
                "optimal_threshold_f1": opt_threshold_f1,
                "optimal_threshold_cost": opt_threshold_cost,
                "training_duration_sec": round(total_training_duration, 2),
            }
        )

        # Log headline and thresholded metrics
        mlflow.log_metrics(
            {
                "training_duration_sec": total_training_duration,
                "memory_before_mb": mem_before_mb,
                "memory_peak_mb": mem_peak_mb,
                "val_pr_auc": default_metrics.pr_auc,
                "val_roc_auc": default_metrics.roc_auc,
                "val_precision_at_0_5": default_metrics.precision,
                "val_recall_at_0_5": default_metrics.recall,
                "val_f1_at_0_5": default_metrics.f1,
                "val_tp_at_0_5": default_metrics.tp,
                "val_fp_at_0_5": default_metrics.fp,
                "val_fn_at_0_5": default_metrics.fn,
                "val_tn_at_0_5": default_metrics.tn,
                "val_precision_f1_opt": f1_metrics.precision,
                "val_recall_f1_opt": f1_metrics.recall,
                "val_f1_opt": f1_metrics.f1,
                "val_tp_f1_opt": f1_metrics.tp,
                "val_fp_f1_opt": f1_metrics.fp,
                "val_fn_f1_opt": f1_metrics.fn,
                "val_precision_cost_opt": cost_metrics.precision,
                "val_recall_cost_opt": cost_metrics.recall,
                "val_f1_cost_opt": cost_metrics.f1,
                "val_tp_cost_opt": cost_metrics.tp,
                "val_fp_cost_opt": cost_metrics.fp,
                "val_fn_cost_opt": cost_metrics.fn,
            }
        )

        # Log artifacts
        mlflow.log_artifact(str(sweep_csv_path), artifact_path="evaluation")
        mlflow.log_artifact(str(pr_curve_csv_path), artifact_path="evaluation")
        mlflow.log_artifact(str(pr_curve_img_path), artifact_path="evaluation")
        mlflow.log_artifact(str(cm_img_path), artifact_path="evaluation")
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

        # Log and register model
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model",
            registered_model_name="FraudShield-XGBoost",
        )

    # 7. Model loading verification
    logger.info("Verifying MLflow model loading and inference...")
    logged_model_uri = f"runs:/{run_id}/model"
    loaded_xgb = mlflow.xgboost.load_model(logged_model_uri)
    sample_preds = loaded_xgb.predict_proba(X_val[:100])[:, 1]
    original_preds = best_model.predict_proba(X_val[:100])[:, 1]
    np.testing.assert_allclose(
        sample_preds,
        original_preds,
        rtol=1e-5,
        err_msg="MLflow loaded model predictions mismatch!",
    )
    logger.info(
        "MLflow model loaded successfully and verified with exact predictions!"
    )

    return {
        "run_id": run_id,
        "best_params": best_params,
        "tuning_records": tuning_records,
        "default_metrics": default_metrics.to_dict(),
        "f1_metrics": f1_metrics.to_dict(),
        "cost_metrics": cost_metrics.to_dict(),
        "optimal_threshold_f1": opt_threshold_f1,
        "optimal_threshold_cost": opt_threshold_cost,
        "recall_85_record": rec85_record,
        "feature_names": feature_names,
        "train_duration": total_training_duration,
        "mem_peak_mb": mem_peak_mb,
    }


if __name__ == "__main__":
    results = run_xgboost_training()
    print("\n" + "=" * 60)
    print("XGBOOST TRAINING & TUNING COMPLETED")
    print("=" * 60)
    print(f"Best Candidate: {results['best_params']['name']}")
    print(f"Validation PR-AUC: {results['default_metrics']['pr_auc']:.5f}")
    print(f"Validation ROC-AUC: {results['default_metrics']['roc_auc']:.5f}")
    print(
        f"Default (0.50) -> Prec: {results['default_metrics']['precision']:.4f}, "
        f"Rec: {results['default_metrics']['recall']:.4f}, F1: {results['default_metrics']['f1']:.4f}"
    )
    print(
        f"Optimal F1 ({results['optimal_threshold_f1']:.2f}) -> Prec: {results['f1_metrics']['precision']:.4f}, "
        f"Rec: {results['f1_metrics']['recall']:.4f}, F1: {results['f1_metrics']['f1']:.4f}"
    )
    print(
        f"Optimal Cost ({results['optimal_threshold_cost']:.2f}) -> Prec: {results['cost_metrics']['precision']:.4f}, "
        f"Rec: {results['cost_metrics']['recall']:.4f}, F1: {results['cost_metrics']['f1']:.4f}"
    )
    if results["recall_85_record"]:
        print(
            f"Recall >= 85% Target (Thresh: {results['recall_85_record']['threshold']:.2f}) -> "
            f"Prec: {results['recall_85_record']['precision']:.4f}, Rec: {results['recall_85_record']['recall']:.4f}"
        )
    print("=" * 60)
