# Phase 3: Model Training, Threshold Optimization & MLflow Tracking

## 1. Executive Summary

Phase 3 establishes the core predictive engine for **FraudShield**, transitioning from feature engineering to production-grade fraud risk scoring. In accordance with `Rules.md`, all model exploration, hyperparameter tuning, and threshold calibration were strictly conducted using chronological `train` (steps 1–323) and `val` (steps 324–378) splits. The strictly held-out `test` split (steps 379–743) was evaluated exactly once after all decisions were finalized.

Key headline results:
- **Baseline Logistic Regression (Validation):** Average Precision (PR-AUC) **0.1455**, ROC-AUC 0.9819.
- **XGBoost Classifier (Validation):** Average Precision (PR-AUC) **0.7806** (+436% relative lift over baseline), ROC-AUC 0.9994.
- **Hold-Out Test Set (Final Evaluation):** Average Precision (PR-AUC) **0.9238**, ROC-AUC 0.9994. At selected validation threshold ($t = 0.99$), achieved **95.95% Precision**, **73.27% Recall**, and **$F_1 = 0.8309$** on 918,617 strictly future transactions.
- **MLflow Tracking & Registry:** 100% of runs, parameters, metrics, PR curve plots, confusion matrices, and model artifacts logged locally to `sqlite:///mlruns.db` with registered model `FraudShield-XGBoost` version 1. Reload inference parity verified at $\Delta = 0.0$.

---

## 2. Pre-Training Invariants & Leakage Elimination

Before training any model, the feature matrix was validated to eliminate target leakage and future data snooping:

| Column Name | Status | Rationale |
|---|---|---|
| `isFraud` | **Target Only** | Binary ground truth label (0 = Legitimate, 1 = Fraud). Detached as $y$. |
| `newbalanceOrig` | **Excluded** | Target leakage: Post-transaction origin balance. Drained to $0.00$ in 98.05% of fraud cases. |
| `newbalanceDest` | **Excluded** | Target leakage: Post-transaction destination balance. Unavailable at pre-authorization inference time. |
| `isFlaggedFraud` | **Excluded** | Target leakage / Collinearity: Naive heuristic flag (> $200k transfer rule). |
| `nameOrig`, `nameDest` | **Excluded from X** | Account identifiers; used as join entities in Feast and graph nodes in Phase 5, excluded from tabular matrix. |
| `step`, `event_timestamp` | **Excluded from X** | Chronological ordering keys used only for temporal boundary separation. |

### Chronological Split Integrity
- **Train Split:** Steps 1 to 323 ($N = 4,463,587$, Fraud: 3,643, Prevalence: 0.0816%).
- **Validation Split:** Steps 324 to 378 ($N = 980,416$, Fraud: 564, Prevalence: 0.0575%).
- **Test Split (Hold-Out):** Steps 379 to 743 ($N = 918,617$, Fraud: 4,006, Prevalence: 0.4361%).
- Strict ordering verified: $\max(\text{train.step}) = 323 < \min(\text{val.step}) = 324 \le \max(\text{val.step}) = 378 < \min(\text{test.step}) = 379$.

### Preprocessing Protocol
- Feature transformations (`OneHotEncoder` on `type`, `StandardScaler` for LR, passthrough for trees) were **fit exclusively on training data** and applied out-of-sample to validation and test data.
- Zero synthetic oversampling (SMOTE) was applied to validation or test data.

---

## 3. Baseline Logistic Regression

To benchmark performance, a deterministic, class-weighted Logistic Regression baseline was implemented (`src/training/train_baseline.py`).

- **Solver:** `lbfgs`, `max_iter=500`, `random_state=42`.
- **Class Weighting:** `class_weight='balanced'` to offset the 1224.25:1 legitimate-to-fraud imbalance.
- **Preprocessing:** One-hot encoding on `type` + `StandardScaler` on numerical features, fit strictly on `train`.

### Why Accuracy is Ineffective in Fraud Detection
In PaySim, legitimate transactions constitute 99.87% of traffic. A trivial dummy model that predicts legitimate for all transactions achieves **99.9425% accuracy** on the validation split while detecting **0 out of 564 fraud cases** (100% loss of fraud capital).

Furthermore, while Logistic Regression achieved a seemingly stellar ROC-AUC of **0.9819**, its true detection capability is revealed by the Precision-Recall curve:
- **Validation PR-AUC (Average Precision):** **0.1455**
- **Default Threshold ($t = 0.50$):**
  - Precision: **0.71%** (0.0071)
  - Recall: **91.67%** (517 / 564)
  - Confusion Matrix: $TP = 517$, $FP = 72,151$, $FN = 47$, $TN = 907,701$
  - Operational Burden: **72,151 false alarms** for 517 confirmed frauds (140 false alerts per genuine fraud case).
- **Optimal $F_1$ Threshold ($t = 0.99$):**
  - Precision: **5.86%**, Recall: **34.93%**, $F_1$: **0.1003**
  - Confusion Matrix: $TP = 197$, $FP = 3,166$, $FN = 367$, $TN = 976,686$

The linear decision boundary of Logistic Regression is fundamentally incapable of capturing non-linear feature interactions (such as the ratio between velocity and historical account volume).

---

## 4. XGBoost Classifier Tuning & Validation Results

An XGBoost classifier was trained with histogram binning (`tree_method='hist'`) using the exact negative-to-positive training ratio as `scale_pos_weight = 1224.25`.

### Hyperparameter Search Candidates

All candidates used `tree_method='hist'`, `eval_metric='aucpr'`, `early_stopping_rounds=20`, and `random_state=42`:

| Candidate Name | `max_depth` | `learning_rate` | `subsample` | `colsample_bytree` | Best Iteration | Training Time (s) | Val PR-AUC |
|---|---|---|---|---|---|---|---|
| `xgb_shallow_lr08` | 4 | 0.08 | 0.8 | 0.8 | 292 | 183.7s | 0.73757 |
| `xgb_medium_lr08` | 6 | 0.08 | 0.8 | 0.8 | 289 | 238.5s | 0.77901 |
| **`xgb_medium_lr05`** | **6** | **0.05** | **0.9** | **0.9** | **299** | **232.9s** | **0.78057** |

Selected Winner: **`xgb_medium_lr05`** with a Validation PR-AUC of **0.78057**.

---

## 5. Threshold Selection & Business Trade-Offs

Because `scale_pos_weight = 1224.25` shifts raw model probabilities toward 1.0, the naive decision threshold of $0.50$ is not economically or operationally optimal. A comprehensive sweep across 99 thresholds ($t \in [0.01, 0.99]$) was conducted on validation data.

### Decision Criteria

```
                        Predicted Legit (0)    Predicted Fraud (1)
True Legit (0)                 TN                      FP (Cost: $15 analyst review)
True Fraud (1)                 FN (Loss: $500 avg)     TP
```

| Operating Objective | Threshold ($t$) | Precision | Recall | $F_1$ Score | $TP$ | $FP$ | $FN$ | Business Cost |
|---|---|---|---|---|---|---|---|---|
| Default ($t = 0.50$) | 0.50 | 0.0467 | **0.9947** | 0.0892 | 561 | 11,460 | 3 | $173,400 |
| **Max $F_1$ Harmonic** | **0.99** | **0.8193** | **0.7074** | **0.7593** | **399** | **88** | **165** | **$83,820** |
| Min Business Cost | 0.92 | 0.1830 | 0.9344 | 0.3061 | 527 | 2,352 | 37 | **$53,780** |
| Target Recall $\ge 85\%$ | 0.94 | 0.2559 | 0.8582 | 0.3943 | 484 | 1,407 | 80 | $61,105 |

### Selected Operating Threshold: $t = 0.99$ (Max $F_1$)
- **Rationale:** In high-volume consumer payment platforms, analyst capacity is finite. Operating at $t = 0.50$ produces **11,460 alerts**, inducing catastrophic alert fatigue. Operating at $t = 0.99$ yields **81.93% Precision** (only 88 false alerts for 399 detected frauds) while recovering **70.74% of all fraud incidents**, achieving an optimal balance between loss prevention and operational throughput.

---

## 6. Hold-Out Test Set Evaluation

The test split (`data/processed/test_features.parquet`, $N = 918,617$, steps 379–743) was loaded and evaluated **strictly once** using the frozen preprocessor and selected threshold ($t = 0.99$).

### Final Hold-Out Test Metrics
- **Average Precision (PR-AUC):** **0.92381**
- **ROC-AUC:** **0.99941**
- **Precision at $t = 0.99$:** **0.9595** (95.95%)
- **Recall at $t = 0.99$:** **0.7327** (73.27%)
- **$F_1$ Score at $t = 0.99$:** **0.8309**
- **Confusion Matrix:**
  - True Positives ($TP$): **2,935**
  - False Positives ($FP$): **124**
  - False Negatives ($FN$): **1,071**
  - True Negatives ($TN$): **914,487**
- **False Positive Rate:** **0.0136%** (only 124 false alarms out of 914,611 legitimate payments)
- **False Negative Rate:** **26.73%**

### Model Comparison Summary

| Model | Evaluation Split | PR-AUC | ROC-AUC | Threshold | Precision | Recall | $F_1$ | False Positives |
|---|---|---|---|---|---|---|---|---|
| Baseline Logistic Regression | Validation | 0.14547 | 0.98191 | 0.99 | 0.0586 | 0.3493 | 0.1003 | 3,166 |
| XGBoost (`xgb_medium_lr05`) | Validation | 0.78057 | 0.99941 | 0.99 | 0.8193 | 0.7074 | 0.7593 | 88 |
| **XGBoost (`xgb_medium_lr05`)** | **Hold-Out Test** | **0.92381** | **0.99941** | **0.99** | **0.9595** | **0.7327** | **0.8309** | **124** |

---

## 7. Chronological Drift & Fraud Prevalence Investigation

### The Prevalence Jump
Across the three chronological splits, the fraud prevalence exhibits significant non-stationarity:
- **Train split (steps 1–323, days 1–13.5):** 3,643 frauds / 4,463,587 txns = **0.0816%**
- **Validation split (steps 324–378, days 13.5–15.8):** 564 frauds / 980,416 txns = **0.0575%**
- **Test split (steps 379–743, days 15.8–31.0):** 4,006 frauds / 918,617 txns = **0.4361%**

Fraud prevalence in the test split is **5.34x higher than train** and **7.58x higher than validation**.

### Impact on Precision and Recall Interpretation
Why did test precision increase to 95.95% compared to 81.93% on validation?

From Bayes' Theorem, Precision (Positive Predictive Value) is mathematically coupled to base-rate prevalence $\pi$:
$$\text{Precision} = \frac{\text{Sensitivity} \cdot \pi}{\text{Sensitivity} \cdot \pi + (1 - \text{Specificity}) \cdot (1 - \pi)}$$

Where:
- $\text{Sensitivity} = \text{Recall} \approx 71\% - 73\%$
- $1 - \text{Specificity} = \text{FPR} \approx 0.013\% - 0.014\%$

Because the false positive rate $(1 - \text{Specificity})$ remained virtually identical across validation ($0.009\%$) and test ($0.0136\%$), the ~7.6x surge in fraud occurrence density in the test window mechanically increased the ratio of True Positives to False Positives ($2935 / 124 = 23.7:1$ vs $399 / 88 = 4.5:1$).

This highlights why:
1. **Reporting prevalence is mandatory:** Stating a 95.95% precision without noting test prevalence would be statistically deceptive.
2. **PR-AUC is the true invariant:** PR-AUC reflects ranking quality across the entire threshold spectrum, and the test PR-AUC of 0.9238 confirms that the engineered features maintain discriminative power across chronological distribution shifts.

---

## 8. MLflow Experiment Tracking & Model Registry

A local SQLite-backed MLflow tracking store was configured at `sqlite:///mlruns.db` (zero Docker dependency).

### Verified Runs

| Run Name | Run ID | Key Parameters | Key Metrics | Logged Artifacts |
|---|---|---|---|---|
| `baseline-logistic-regression` | `915c6ec...` | `class_weight='balanced'`, `solver='lbfgs'` | `val_pr_auc=0.1455`, `optimal_threshold=0.99` | `baseline_val_sweep.csv`, `baseline_pr_curve.png`, `baseline_confusion_matrix.png`, model |
| `xgboost-tuned` | `17bd29a...` | `max_depth=6`, `lr=0.05`, `scale_pos_weight=1224.25` | `val_pr_auc=0.7806`, `f1_opt=0.7593` | `xgboost_val_sweep.csv`, `xgboost_pr_curve.png`, `xgboost_confusion_matrix.png`, model |
| `xgboost-holdout-test-evaluation` | `f8e32d4...` | `evaluation_split='test'`, `selected_threshold=0.99` | `test_pr_auc=0.9238`, `test_f1=0.8309`, `test_precision=0.9595` | `test_pr_curve.png`, `test_confusion_matrix.png` |

### Registered Model
- Model Name: `FraudShield-XGBoost`
- Version: `1`
- Stage: Ready for Promotion to Production serving (Phase 4).

### How to Launch MLflow UI
To explore runs, parameters, metrics, and PR curve artifacts interactively:
```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlruns.db --port 5000
```
Then navigate to `http://localhost:5000` in any web browser.

---

## 9. Hardware Constraints, Memory & Runtime Observations

All operations were executed on a Windows laptop with ~16 GB installed RAM and ~4.0 GB free memory.

| Pipeline Component | Runtime (s) | Peak RSS Memory (MB) | Memory Safeguards Applied |
|---|---|---|---|
| Parquet Feature Loading | ~1.8s per split | ~210 MB | Read only 12 essential predictor columns, avoiding string accounts |
| Baseline Logistic Regression | 14.0s fit (61.2s total) | 1,199.5 MB | Fit on scaled arrays; freed raw DataFrames with `gc.collect()` |
| XGBoost Tuning (3 Candidates) | 674.3s (11.2 min) | 1,595.8 MB | Sequential candidate execution; immediate deletion of unselected candidate models |
| Hold-Out Test Evaluation | 10.3s | ~320 MB | Preprocessing with training-fitted transformer; immediate GC |
| Full Pytest Suite (23 tests) | 48.9s | ~450 MB | In-memory fixtures and synthetic samples for fast unit tests |

At no point did peak memory usage exceed 1.6 GB, ensuring complete system stability and leaving ample headroom (> 2.4 GB) on the host machine.

---

## 10. Reproducibility & Verification

To replicate all Phase 3 results from scratch:

```powershell
# 1. Run baseline logistic regression
.\.venv\Scripts\python.exe -m src.training.train_baseline

# 2. Run XGBoost training, tuning, and MLflow registration
.\.venv\Scripts\python.exe -m src.training.train_xgb

# 3. Evaluate selected model on hold-out test set
.\.venv\Scripts\python.exe -m src.training.evaluate_test

# 4. Verify MLflow integration and reload parity
.\.venv\Scripts\python.exe -m scratch.verify_mlflow

# 5. Run linter and full unit test suite
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe -v
```

---

## 11. Acceptance Criteria Checklist & Phase 4 Handoff

- [x] Baseline Logistic Regression evaluated on validation data with PR-AUC, confusion matrix, and threshold optimization.
- [x] XGBoost trained with `scale_pos_weight = neg/pos` (1224.25:1) and early stopping on `aucpr`.
- [x] Hyperparameter tuning conducted sequentially without GPU dependency or parallel memory blowout.
- [x] Operating threshold chosen via explicit business trade-offs on validation split ($t = 0.99$).
- [x] Strictly preserved hold-out test set evaluated once, reporting 0.9238 PR-AUC and 95.95% Precision.
- [x] Fraud prevalence drift analyzed and mathematically explained via Bayes' theorem.
- [x] All runs, metrics, parameters, and plots tracked to local SQLite MLflow (`FraudShield-XGBoost` registered).
- [x] Model reload parity verified with exact numerical match ($\Delta = 0.0$).
- [x] All 23 unit tests passing, zero ruff linter errors.

**Phase 3 is complete and ready for Phase 4 (Streaming Scoring Service).**
