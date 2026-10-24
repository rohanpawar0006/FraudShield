"""Comprehensive evaluation metrics, PR curve, and business threshold optimization."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Summary of model evaluation metrics at a specific decision threshold."""

    pr_auc: float
    roc_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    total_samples: int
    positive_samples: int

    @property
    def false_positive_rate(self) -> float:
        """Rate of legitimate transactions incorrectly flagged."""
        return float(self.fp / (self.fp + self.tn)) if (self.fp + self.tn) > 0 else 0.0

    @property
    def false_negative_rate(self) -> float:
        """Rate of fraud transactions missed."""
        return float(self.fn / (self.fn + self.tp)) if (self.fn + self.tp) > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging and serialization."""
        return {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "fpr": self.false_positive_rate,
            "fnr": self.false_negative_rate,
            "total_samples": self.total_samples,
            "positive_samples": self.positive_samples,
        }


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> EvaluationMetrics:
    """Compute headline fraud metrics (PR-AUC, ROC-AUC, thresholded metrics).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities for positive class [0, 1].
        threshold: Decision threshold for classification.

    Returns:
        EvaluationMetrics dataclass instance.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    # Headline metric: Average Precision / PR-AUC
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    # Binary decisions at threshold
    y_pred = (y_prob_arr >= threshold).astype(np.int64)

    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    return EvaluationMetrics(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        threshold=threshold,
        precision=prec,
        recall=rec,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        total_samples=len(y_true_arr),
        positive_samples=int(y_true_arr.sum()),
    )


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
    cost_fp: float = 15.0,
    cost_fn: float = 500.0,
) -> pd.DataFrame:
    """Evaluate performance across a spectrum of decision thresholds.

    Args:
        y_true: Binary ground truth array.
        y_prob: Predicted probability array.
        thresholds: List of thresholds to evaluate. Defaults to 0.01..0.99.
        cost_fp: Estimated operational cost per False Positive (analyst triage).
        cost_fn: Estimated financial loss per False Negative (unmitigated fraud).

    Returns:
        DataFrame with threshold performance and business costs.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.01, 1.00, 0.01)]

    records = []
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    for t in thresholds:
        y_pred = (y_prob_arr >= t).astype(np.int64)
        prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

        cm = confusion_matrix(y_true_arr, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        business_cost = (fp * cost_fp) + (fn * cost_fn)

        records.append(
            {
                "threshold": t,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "business_cost": business_cost,
            }
        )

    return pd.DataFrame(records)


def find_optimal_thresholds(
    sweep_df: pd.DataFrame,
    target_recalls: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Identify optimal operating thresholds under distinct business criteria.

    Criteria:
    1. Max F1: Optimal harmonic balance between precision and recall.
    2. Min Business Cost: Minimizes weighted financial losses (FP cost vs FN loss).
    3. Target Recall: Highest precision while maintaining recall >= target.

    Args:
        sweep_df: Output from sweep_thresholds.
        target_recalls: Specific recall targets to evaluate (e.g., [0.80, 0.85, 0.90]).

    Returns:
        Dictionary mapping criterion names to optimal threshold records.
    """
    if target_recalls is None:
        target_recalls = [0.80, 0.85, 0.90]

    optimal_results: Dict[str, Any] = {}

    # 1. Best F1
    best_f1_idx = sweep_df["f1"].idxmax()
    optimal_results["max_f1"] = sweep_df.iloc[best_f1_idx].to_dict()

    # 2. Min Cost
    min_cost_idx = sweep_df["business_cost"].idxmin()
    optimal_results["min_cost"] = sweep_df.iloc[min_cost_idx].to_dict()

    # 3. Recall targets
    for target in target_recalls:
        qualifying = sweep_df[sweep_df["recall"] >= target]
        if not qualifying.empty:
            # Pick highest precision among qualifying
            best_idx = qualifying["precision"].idxmax()
            optimal_results[f"recall_at_{int(target*100)}"] = qualifying.loc[
                best_idx
            ].to_dict()
        else:
            optimal_results[f"recall_at_{int(target*100)}"] = None

    return optimal_results


def get_pr_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve coordinates."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    title: str = "Precision-Recall Curve",
    chosen_threshold: Optional[float] = None,
    chosen_precision: Optional[float] = None,
    chosen_recall: Optional[float] = None,
) -> None:
    """Plot and save Precision-Recall curve to file.

    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted fraud probabilities.
        output_path: Path to save generated plot.
        title: Plot title.
        chosen_threshold: Selected threshold value to annotate.
        chosen_precision: Precision at selected threshold.
        chosen_recall: Recall at selected threshold.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#2563EB", lw=2, label=f"PR Curve (AP = {ap:.4f})")

    # Baseline random guessing line (prevalence)
    prevalence = float(np.mean(y_true))
    ax.axhline(y=prevalence, color="#DC2626", linestyle="--", lw=1.5, label=f"Random Baseline ({prevalence:.4%})")

    if (
        chosen_threshold is not None
        and chosen_precision is not None
        and chosen_recall is not None
    ):
        ax.scatter(
            [chosen_recall],
            [chosen_precision],
            color="#D97706",
            s=80,
            zorder=5,
            label=f"Selected (t={chosen_threshold:.2f}, P={chosen_precision:.2f}, R={chosen_recall:.2f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    title: str = "Confusion Matrix",
    threshold: Optional[float] = None,
) -> None:
    """Plot and save confusion matrix heatmap to file.

    Args:
        cm: 2x2 confusion matrix array [[TN, FP], [FN, TP]].
        output_path: Path to save plot.
        title: Plot title.
        threshold: Decision threshold.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            text = f"{val:,}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"], fontsize=11)
    ax.set_yticklabels(["Legit (0)", "Fraud (1)"], fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    sub = f" (Threshold = {threshold:.2f})" if threshold is not None else ""
    ax.set_title(f"{title}{sub}", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

