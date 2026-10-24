"""Train and evaluate baseline Logistic Regression model with MLflow tracking."""

import gc
import os
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression

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
    plot_confusion_matrix,
    plot_pr_curve,
    sweep_thresholds,
)

logger = get_logger(__name__)


def run_baseline_training() -> Dict[str, Any]:
    """Train baseline Logistic Regression, evaluate on validation set, and log to MLflow.

    Returns:
        Dictionary with evaluation results and model paths.
    """
    logger.info("Starting baseline Logistic Regression training pipeline")
    proc = psutil.Process()
    mem_before_mb = proc.memory_info().rss / (1024 * 1024)

    # 1. Load splits
    X_train, y_train = load_feature_split("train")
    X_val, y_val = load_feature_split("val")

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    imbalance_ratio = float(n_neg / n_pos)

    logger.info(
        "Train size: %d (Pos: %d, Neg: %d, Imbalance: %.2f:1), Val size: %d (Pos: %d)",
        len(X_train),
        n_pos,
        n_neg,
        imbalance_ratio,
        len(X_val),
        int(y_val.sum()),
    )

    # 2. Fit preprocessing ONLY on training data
    logger.info("Fitting ColumnTransformer on training data only...")
    preprocessor = create_preprocessor(scale_numerical=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    feature_names = get_feature_names_out(preprocessor)

    # Clean up raw training dataframe to save memory
    del X_train
    gc.collect()

    # 3. Model training with balanced class weights
    logger.info("Training LogisticRegression with class_weight='balanced'...")
    lr_params = {
        "class_weight": "balanced",
        "max_iter": 500,
        "random_state": 42,
        "solver": "lbfgs",
    }
    start_time = time.perf_counter()
    model = LogisticRegression(**lr_params)
    model.fit(X_train_proc, y_train)
    train_duration = time.perf_counter() - start_time
    mem_peak_mb = proc.memory_info().rss / (1024 * 1024)

    # Clean up transformed train array
    del X_train_proc
    gc.collect()

    logger.info(
        "Logistic Regression fit completed in %.2fs. Peak RSS: %.1f MB",
        train_duration,
        mem_peak_mb,
    )

    # 4. Predict probabilities on validation data
    logger.info("Evaluating on validation split...")
    y_val_prob = model.predict_proba(X_val_proc)[:, 1]

    # Evaluate at default threshold (0.5)
    default_metrics = evaluate_predictions(
        y_val.to_numpy(), y_val_prob, threshold=0.5
    )
    sweep_df = sweep_thresholds(y_val.to_numpy(), y_val_prob)
    optimal_thresholds = find_optimal_thresholds(sweep_df)

    best_f1_info = optimal_thresholds["max_f1"]
    best_f1_threshold = float(best_f1_info["threshold"])
    optimal_metrics = evaluate_predictions(
        y_val.to_numpy(), y_val_prob, threshold=best_f1_threshold
    )

    # Calculate naive accuracy (predicting all 0)
    naive_accuracy = float((y_val == 0).mean())
    baseline_accuracy = float(
        (((y_val_prob >= 0.5).astype(int) == y_val).mean())
    )

    logger.info(
        "Validation PR-AUC: %.5f | ROC-AUC: %.5f",
        default_metrics.pr_auc,
        default_metrics.roc_auc,
    )
    logger.info(
        "Default Threshold (0.50): Prec=%.4f, Rec=%.4f, F1=%.4f, TP=%d, FP=%d, FN=%d",
        default_metrics.precision,
        default_metrics.recall,
        default_metrics.f1,
        default_metrics.tp,
        default_metrics.fp,
        default_metrics.fn,
    )
    logger.info(
        "Optimal F1 Threshold (%.2f): Prec=%.4f, Rec=%.4f, F1=%.4f, TP=%d, FP=%d, FN=%d",
        best_f1_threshold,
        optimal_metrics.precision,
        optimal_metrics.recall,
        optimal_metrics.f1,
        optimal_metrics.tp,
        optimal_metrics.fp,
        optimal_metrics.fn,
    )

    # 5. Persist artifacts locally and generate evaluation plots
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "baseline_logistic_regression.joblib"
    preprocessor_path = models_dir / "baseline_preprocessor.joblib"
    sweep_path = models_dir / "baseline_val_sweep.csv"
    pr_curve_path = models_dir / "baseline_pr_curve.png"
    cm_path = models_dir / "baseline_confusion_matrix.png"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    sweep_df.to_csv(sweep_path, index=False)

    y_val_arr = y_val.to_numpy()
    plot_pr_curve(
        y_val_arr,
        y_val_prob,
        output_path=str(pr_curve_path),
        title="Baseline Logistic Regression - Validation PR Curve",
        chosen_threshold=best_f1_threshold,
        chosen_precision=optimal_metrics.precision,
        chosen_recall=optimal_metrics.recall,
    )
    cm_opt = np.array(
        [[optimal_metrics.tn, optimal_metrics.fp], [optimal_metrics.fn, optimal_metrics.tp]]
    )
    plot_confusion_matrix(
        cm_opt,
        output_path=str(cm_path),
        title="Baseline LR Confusion Matrix",
        threshold=best_f1_threshold,
    )

    # 6. MLflow Tracking
    tracking_uri = settings.MLFLOW_TRACKING_URI
    experiment_name = settings.MLFLOW_EXPERIMENT_NAME
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    with mlflow.start_run(run_name="baseline-logistic-regression") as run:
        run_id = run.info.run_id
        logger.info("Logging baseline run to MLflow (run_id: %s)", run_id)

        # Log params
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 500,
                "solver": "lbfgs",
                "random_state": 42,
                "scale_numerical": True,
                "train_dataset": "data/processed/train_features.parquet",
                "val_dataset": "data/processed/val_features.parquet",
                "train_samples": len(y_train),
                "train_fraud_samples": n_pos,
                "train_legit_samples": n_neg,
                "imbalance_ratio": imbalance_ratio,
                "val_samples": len(X_val),
                "val_fraud_samples": int(y_val.sum()),
                "feature_count": len(feature_names),
                "optimal_threshold": best_f1_threshold,
                "train_duration_sec": round(train_duration, 2),
            }
        )

        # Log default and optimal metrics
        mlflow.log_metrics(
            {
                "train_duration_sec": train_duration,
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
                "val_precision_optimal": optimal_metrics.precision,
                "val_recall_optimal": optimal_metrics.recall,
                "val_f1_optimal": optimal_metrics.f1,
                "val_tp_optimal": optimal_metrics.tp,
                "val_fp_optimal": optimal_metrics.fp,
                "val_fn_optimal": optimal_metrics.fn,
                "val_naive_accuracy": naive_accuracy,
                "val_accuracy_at_0_5": baseline_accuracy,
            }
        )

        # Log artifacts
        mlflow.log_artifact(str(sweep_path), artifact_path="evaluation")
        mlflow.log_artifact(str(pr_curve_path), artifact_path="evaluation")
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="FraudShield-Baseline-LR",
        )

    return {
        "run_id": run_id,
        "default_metrics": default_metrics.to_dict(),
        "optimal_metrics": optimal_metrics.to_dict(),
        "optimal_threshold": best_f1_threshold,
        "naive_accuracy": naive_accuracy,
        "feature_names": feature_names,
        "train_duration": train_duration,
        "mem_peak_mb": mem_peak_mb,
    }


if __name__ == "__main__":
    results = run_baseline_training()
    print("\n" + "=" * 60)
    print("BASELINE LOGISTIC REGRESSION COMPLETED")
    print("=" * 60)
    print(f"Validation PR-AUC: {results['default_metrics']['pr_auc']:.5f}")
    print(f"Validation ROC-AUC: {results['default_metrics']['roc_auc']:.5f}")
    print(f"Optimal Threshold: {results['optimal_threshold']:.2f}")
    print(
        f"Optimal F1: {results['optimal_metrics']['f1']:.4f} "
        f"(Precision: {results['optimal_metrics']['precision']:.4f}, "
        f"Recall: {results['optimal_metrics']['recall']:.4f})"
    )
    print(
        f"Confusion Matrix at optimal: TP={results['optimal_metrics']['tp']}, "
        f"FP={results['optimal_metrics']['fp']}, FN={results['optimal_metrics']['fn']}, "
        f"TN={results['optimal_metrics']['tn']}"
    )
    print(
        f"Naive accuracy (always predicting 0): {results['naive_accuracy']:.6%}"
    )
    print("=" * 60)
