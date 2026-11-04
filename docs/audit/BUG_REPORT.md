# FraudShield — Critical Bug Investigation & Engineering Report

**Auditor:** Senior AI/ML & MLOps Architect  
**Date:** 2026-11-03  
**Status:** Confirmed & Prioritized for Remediation  

---

## 1. Summary of Investigated Issues

| Bug ID | Component | Severity | Description | Status |
|---|---|---|---|---|
| **BUG-01** | `src/monitoring/retrain_trigger.py` | **CRITICAL** | `safe_labels` is filtered for gate-set leakage but never merged into training DataFrame `df_train`. | **CONFIRMED** |
| **BUG-02** | `src/feedback/service.py` | **HIGH** | Hardcoded `s.blended_score >= settings.DECISION_THRESHOLD` in `list_alerts` prevents retrieving alerts below 0.99. | **CONFIRMED** |
| **BUG-03** | `src/serving/app.py` | **HIGH (SECURITY)** | `POST /admin/reload-model` accepts `X-Admin-Key` header parameter but never validates it. | **CONFIRMED** |
| **BUG-04** | `src/serving/app.py` | **MEDIUM** | Documented `POST /admin/retrain` endpoint in `Architecture.md` is not implemented in FastAPI service. | **CONFIRMED** |
| **BUG-05** | `src/training/blend.py`, `config.py` | **MEDIUM** | Global `BLEND_ALPHA = 1.0` renders GNN graph risk score inert even for warm, connected accounts. | **CONFIRMED** |

---

## 2. In-Depth Root Cause & Code Analysis

### BUG-01: Analyst Feedback Not Incorporated into Retraining
- **Location:** `src/monitoring/retrain_trigger.py:105-122`
- **Code Inspection:**
  ```python
  # Fetch verified analyst labels from feedback loop
  labeled_cases_df = get_labeled_cases()
  if not labeled_cases_df.empty:
      leakage_overlap = set(labeled_cases_df["transaction_id"].astype(str)).intersection(gate_txn_ids)
      if leakage_overlap:
          raise ValueError(...)

      safe_labels = labeled_cases_df[~labeled_cases_df["transaction_id"].astype(str).isin(gate_txn_ids)]
      logger.info("Incorporating %d verified analyst labels into retraining pool...", len(safe_labels))

  return df_train, gate_df
  ```
- **Root Cause:** Notice that while `safe_labels` is computed and logged, `df_train` is returned completely untouched! The retrained candidate booster is trained solely on the static `df_train` Parquet file without any analyst ground-truth feedback.
- **Proposed Fix:**
  1. Retrieve raw transaction feature records for the transactions in `safe_labels` from the database `transactions` table.
  2. Parse the payload features (`type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`, etc.) and set target `isFraud = safe_labels['label']`.
  3. If existing records in `df_train` share transaction IDs with `safe_labels`, update their `isFraud` column in-place.
  4. Concatenate new verified rows to `df_train` before model fitting, ensuring zero gate-set overlap.

---

### BUG-02: Dashboard Alert Filtering Obstruction
- **Location:** `src/feedback/service.py:55-78`
- **Code Inspection:**
  ```python
  conditions = ["s.blended_score >= " + ph]
  params: List[Any] = [settings.DECISION_THRESHOLD]

  if min_score is not None:
      conditions.append(f"s.blended_score >= {ph}")
      params.append(float(min_score))
  ```
- **Root Cause:** `conditions` is pre-populated with `settings.DECISION_THRESHOLD` (0.99). When `min_score` is supplied from the Streamlit UI slider (which spans 0.50 to 1.00), the SQL query executes:
  `WHERE s.blended_score >= 0.99 AND s.blended_score >= min_score`.
  If the analyst moves the slider to 0.70 to inspect borderline cases, the query still filters out all records with score $< 0.99$.
- **Proposed Fix:**
  Initialize the lower score threshold condition dynamically:
  ```python
  effective_min_score = min_score if min_score is not None else settings.DECISION_THRESHOLD
  conditions = [f"s.blended_score >= {ph}"]
  params: List[Any] = [float(effective_min_score)]
  ```

---

### BUG-03: Unauthenticated Administrative Operations
- **Location:** `src/serving/app.py:162-171`
- **Code Inspection:**
  ```python
  @app.post("/admin/reload-model", status_code=status.HTTP_200_OK)
  async def reload_model(
      x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
  ) -> Dict[str, Any]:
      try:
          scoring_engine.reload()
          ...
  ```
- **Root Cause:** The `x_admin_key` header is accepted as a parameter in FastAPI but is never verified. No token comparison occurs. Anyone with network access to the API can trigger model hot-reloads.
- **Proposed Fix:**
  1. Add `ADMIN_API_KEY: Optional[str] = Field(default=None)` in `src/common/config.py`.
  2. Implement an administrative authentication dependency using constant-time comparison (`secrets.compare_digest`).
  3. In production (`ENV in ("production", "prod")`) or whenever `ADMIN_API_KEY` is configured, reject requests missing or presenting an invalid key with `HTTP 401 Unauthorized`.
  4. Ensure credentials are never hardcoded and read strictly from environment variables.

---

### BUG-04: Missing `POST /admin/retrain` Endpoint
- **Location:** `Architecture.md` (Table row: `POST /admin/retrain | Manual retrain trigger`) vs `src/serving/app.py`.
- **Root Cause:** The specification designated `POST /admin/retrain` as an administrative manual trigger, but the endpoint was never implemented in the FastAPI application.
- **Proposed Fix:**
  Add `POST /admin/retrain` to `src/serving/app.py` protected by the same administrative authentication dependency, invoking `check_retrain_trigger` or scheduling a background retraining run and returning trigger status.

---

### BUG-05: GNN Relational Risk Signal Inertness
- **Location:** `src/common/config.py` (`BLEND_ALPHA = 1.0`) & `src/training/blend.py`.
- **Root Cause:** `BLEND_ALPHA=1.0` was set to avoid precision collapse on cold-start test accounts (99.95% of test origin accounts are absent from the static training graph). However, this hardcoded 1.0 weight completely disables GNN risk signals even when an account *is* warm and present in the graph lookup.
- **Proposed Fix:**
  Implement conditional blending:
  - If `gnn_score is None` (cold-start account), use $\alpha = 1.0$ (pure XGBoost fallback).
  - If `gnn_score is not None` (warm account with historical network evidence), apply the calibrated blend weight $\alpha = 0.85$ (or `settings.BLEND_ALPHA`).

