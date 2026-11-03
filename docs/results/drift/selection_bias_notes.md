# Selection Bias & Label Verification Limitations in FraudShield

## 1. Overview
In FraudShield's analyst feedback loop, production transactions undergo continuous scoring. Transactions that score above the decision threshold ($t = 0.99$) generate high-risk alerts. Fraud analysts inspect these alerts and submit definitive ground-truth dispositions:
- **Confirmed Fraud** (mapped to label $1$)
- **False Positive** (mapped to label $0$)
- **Under Review** (non-final; excluded from retraining datasets)

## 2. Selection Bias Dynamics (Positive-Only Triage)
Because alert triage queues are populated strictly by model predictions:
1. **Unlabeled False Negatives:** Fraudulent transactions that slip past the scoring threshold ($S_{\text{blend}} < 0.99$) are routed directly to downstream banking cores without alert generation. Because human investigators rarely review low-risk transactions, these fraudulent instances are **never triaged or labeled**.
2. **Selective Sampling of True Negatives:** Legitimate transactions triaged by analysts are only those that exhibited suspicious characteristics (e.g., high transaction amounts or unusual velocity) resulting in false positives. True negatives with nominal transaction parameters are almost never reviewed.
3. **Training Distribution Drift:** Over time, retraining models purely on analyst-verified cases creates a localized distribution skewed toward hard boundary cases rather than the natural population prior.

## 3. Mitigation Strategies in FraudShield
To safeguard against catastrophic drift and localized overconfidence:
- **Zero-Leakage Frozen Gate Set:** Every retrained candidate model is strictly evaluated on an immutable hold-out gate set drawn from natural population distribution across all transaction tiers.
- **Strict Promotion Contract:** Candidates are rejected if PR-AUC or Recall degraded on the representative gate set, preventing biased boundary models from displacing the baseline champion.
- **Random Sampling Queue (Future Roadmap):** Phase 8+ will introduce a 0.05% random sample audit queue where unflagged transactions are spot-checked to capture stealthy fraud patterns.

## 4. Frozen Gate Set Prevalence vs Headline Test Window Performance
- The frozen evaluation gate set (`data/processed/frozen_gate_set.parquet`) contains 15,000 stratified rows sampled from the hold-out test window ($step \ge 379$), with 230 ground-truth fraud transactions (~1.53% fraud prevalence).
- In contrast, the real, natural hold-out test window has a fraud prevalence of ~0.44% (4,115 frauds out of 918,851 transactions).
- **Critical Methodological Distinction:** Because PR-AUC is sensitive to positive class prevalence, gate-set PR-AUC (~0.95–0.96) is **strictly an internal benchmark for champion-vs-candidate delta comparisons ($\Delta \ge +0.0010$) and must NOT be quoted as external model performance**.
- The headline test performance of FraudShield remains the Phase 3 benchmark: **PR-AUC 0.9239** on the full, unstratified hold-out test split.

