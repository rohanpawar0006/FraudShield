# FraudShield — Machine Learning & Model Evaluation Audit

**Auditor:** Senior AI/ML & MLOps Architect  
**Date:** 2026-11-03  
**Status:** Empirically Verified & Reconciled  

---

## 1. Class Imbalance & Metric Selection

In the 6,362,620-transaction PaySim universe, fraud occurs in only **8,213 transactions (0.1291% prevalence, ~0.13%)**, representing an extreme class imbalance of **1,224 to 1**. 

### The Metric Trap
1. **Accuracy Fallacy:** A trivial majority-class classifier that blindly predicts "Legitimate" for all transactions achieves **99.87% accuracy** on the hold-out test set while allowing 100% of fraud volume to pass undetected.
2. **ROC-AUC Fallacy:** The baseline Logistic Regression model achieves a deceptive **ROC-AUC of 0.9819**, but its **PR-AUC is only 0.1455**. At standard decision threshold $t=0.50$, it generates 3,363 alerts of which only 197 are genuine frauds (5.86% precision, creating 16 false positives for every true detection).
3. **Primary Evaluation Target:** FraudShield optimizes strictly for **Precision-Recall AUC (PR-AUC / Average Precision)** and **Recall @ Precision $\ge 0.90$**, bounding false alarms and capping human analyst review fatigue.

---

## 2. Model Performance Across Paradigms (Hold-Out Test Split)

The hold-out test set contains **918,617 transactions** (steps 481 to 743) with 4,006 ground-truth fraud occurrences (**0.4361% local prevalence**). Metrics are harvested directly from `docs/results/phase5_comparison.csv` and verified against model predictions:

| Model Architecture | PR-AUC | ROC-AUC | Precision @ t | Recall @ t | Recall @ P>=0.90 | Recall @ P>=0.95 | Generated Alerts | Decision Threshold ($t$) | Alpha ($\alpha$) |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline Logistic Regression** | 0.1455 | 0.9819 | 5.86% | 34.93% | 0.00% | 0.00% | 3,363 | $t = 0.50$ | N/A |
| **GNN Edge Risk Classifier alone** | 0.0031 | 0.4029 | 0.21% | 0.22% | 0.00% | 0.00% | 4,190 | $t = 0.50$ | $\alpha = 0.0$ |
| **XGBoost Champion (`@champion`)** | **0.9239** | **0.9994** | **95.95%** | **73.27%** | **78.91%** | **74.84%** | **3,059** | $t = 0.99$ | $\alpha = 1.0$ |
| **Active Production System (XGBoost)** | **0.9239** | **0.9994** | **95.95%** | **73.27%** | **78.91%** | **74.84%** | **3,059** | $t = 0.99$ | $\alpha = 1.0$ |

---

## 3. Empirical Evaluation of GNN Blending & Why Blending is Disabled

While the PyG GraphSAGE model detected money-mule forwarding rings within the training window (validation PR-AUC 0.58), evaluating the GNN Edge Risk Classifier alone on future hold-out test transactions resulted in **PR-AUC 0.0031**.

### Empirical Experiment on Warm Accounts
Rather than assuming blending weight $\alpha = 0.85$ is beneficial, empirical measurements were conducted directly on the warm-account test subset ($N=389,097$ transactions, 975 frauds where origin or destination account was present in the GNN training graph):

```text
Warm subset XGBoost alone PR-AUC (alpha=1.00) : 0.7757
Warm subset Blend (alpha=0.95)                : 0.7740
Warm subset Blend (alpha=0.90)                : 0.7667
Warm subset Blend (alpha=0.85)                : 0.7500
Warm subset Blend (alpha=0.80)                : 0.7106
Warm subset Blend (alpha=0.70)                : 0.6056
Warm subset Blend (alpha=0.50)                : 0.4221
Warm subset GNN alone (alpha=0.00)            : 0.0022
```

### Root Cause Analysis
1. **Single-Use Burner Accounts:** In PaySim, 99.89% of origin accounts in the test split are brand-new accounts with zero history in the training graph (only 995 of 918,617 test transactions had a known `nameOrig`).
2. **Merchant Node Characteristics:** The 42.29% account overlap occurs almost exclusively on destination accounts (`nameDest`), which are legitimate merchant nodes without fraud history.
3. **Temporal Decay of Static Graph:** Because fraud originates at the origin account, static destination node embeddings carry no predictive fraud signal into future simulation days. Blending static GNN scores into XGBoost monotonically degrades performance from $\alpha=1.0$ down to $\alpha=0.0$.

### Final Active Configuration
- **Configuration:** `BLEND_ALPHA = 1.0` in `src/common/config.py`.
- **Pipeline Guarantee:** In `compute_blend_score()`, when `alpha >= 1.0`, the function strictly returns `(float(xgb_score), False)`. GNN risk scores cannot alter fraud scores or alert thresholds.
- **Infrastructure Preserved:** All GNN training code, PyG GraphSAGE architectures, graph datasets, and lookup parquets remain intact for future online streaming graph expansion.

---

## 4. Promotion Gate Statistical Power & Confidence Intervals

The retraining pipeline evaluates candidate models on a frozen gate set of **15,000 transactions (230 frauds)** sampled from late simulation steps ($\ge 600$). 

### Statistical Power Limitation
With only $N_{\text{fraud}} = 230$ in the gate set, the standard error of PR-AUC is:
$$SE \approx \sqrt{\frac{p(1-p)}{N_{\text{fraud}}}} \approx \sqrt{\frac{0.96 \times 0.04}{230}} \approx 0.0129$$
Consequently, minor point-estimate differences ($+0.0010$ to $+0.0040$) yield 95% bootstrap confidence intervals that cross zero (e.g. $[-0.0038, +0.0127]$).

### Dual Gate Safeguards
1. **Rejection Safeguard (Scenario 1):** Candidate v8 achieves PR-AUC 0.9647 vs Champion v1 0.9607 ($\Delta = +0.0040$). Because the 95% bootstrap CI $[-0.0038, +0.0127]$ crosses zero (`ci_strictly_positive = False`), the gate marks it **`[REJECTED]`**, preventing noisy deployments.
2. **Promotion Safeguard (Scenario 2):** When evaluated against a degraded champion (PR-AUC 0.3079), Candidate v8 achieves $\Delta = +0.6568$ with 95% bootstrap CI $[+0.6022, +0.7095]$ strictly positive (`ci_strictly_positive = True`), triggering promotion to `@champion`.
