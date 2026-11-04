# FraudShield Implementation & Engineering Fixes Log

**Audit & Remediation Window:** November 3, 2026  
**Auditor:** Senior AI/ML Engineer & Systems Quality Auditor  
**Status:** All Identified Bugs Remediated, GNN Blending Reconciled, CI Repaired, Verified with Regression Tests  

---

## 1. Overview of Remediations

During the complete codebase audit and post-audit verification, critical bugs, configuration discrepancies, and CI flaws were identified and resolved across the data, serving, monitoring, feedback, security, and MLOps layers. This log documents the root cause, exact code diffs, rationale, and verification for each change.

| Issue ID | Component | File Modified | Description | Verification Test |
| :--- | :--- | :--- | :--- | :--- |
| **BUG-01** | Feedback Loop | [`src/monitoring/retrain_trigger.py`](file:///d:/FraudShield/src/monitoring/retrain_trigger.py) | Verified analyst labels were filtered but never merged into `df_train` | `test_retraining_dataset_merges_safe_labels` |
| **BUG-02** | Dashboard | [`src/feedback/service.py`](file:///d:/FraudShield/src/feedback/service.py) | Hardcoded threshold in SQL query blocked alerts below 0.99 | `test_list_alerts_custom_min_score` |
| **BUG-03** | Serving API | [`src/serving/app.py`](file:///d:/FraudShield/src/serving/app.py) | `POST /admin/reload-model` accepted `X-Admin-Key` but never checked it | `test_admin_endpoint_security_unauthorized` |
| **BUG-04** | Architecture | [`src/serving/app.py`](file:///d:/FraudShield/src/serving/app.py) | Documented `POST /admin/retrain` endpoint was missing from FastAPI app | `test_admin_endpoint_security_authorized` |
| **CONFIG-01** | ML Configuration | [`src/common/config.py`](file:///d:/FraudShield/src/common/config.py), [`src/training/blend.py`](file:///d:/FraudShield/src/training/blend.py) | Restored XGBoost-only scoring (`BLEND_ALPHA=1.0`) based on measured test degradation of static GNN | `test_blend_alpha_one_results_in_xgboost_only_predictions`, `test_gnn_scores_cannot_alter_predictions_in_xgboost_only_config` |
| **CI-01** | CI / Actions | [`.github/workflows/containers.yml`](file:///d:/FraudShield/.github/workflows/containers.yml) | Removed invalid `yannh/kubeconform-action` step; pinned `v0.6.7` binary installation and execution | `test_kubernetes_manifest_validation_workflow_configured` |

---

## 2. Detailed Technical Fixes

### Fix 1: Merging Verified Feedback into Retraining Dataset (BUG-01)
- **File:** `src/monitoring/retrain_trigger.py` (`prepare_retraining_dataset`)
- **Problem:** In earlier versions, `get_labeled_cases()` was called and filtered for gate-set leakage (`safe_labels`), but the function concluded by returning `df_train` without incorporating `safe_labels`. This broke the continuous improvement feedback loop.
- **Solution:** 
  1. Maintained strict zero-leakage validation against the frozen gate set (`gate_txn_ids`).
  2. Identified existing transactions in `df_train` matching `safe_labels` and updated their ground truth `isFraud` column in-place.
  3. Identified new transactions not present in `df_train`. Extracted their features from the `transactions` table in `db_manager` (or directly from `safe_labels` in test fixtures), defaulted any missing features using `DEFAULT_FEATURE_VALUES`, assigned `isFraud`, and concatenated them into `df_train`.
- **Regression Test:** Added `test_retraining_dataset_merges_safe_labels` in `tests/unit/test_monitoring.py`.

---

### Fix 2: Dynamic Alert Score Filtering in Feedback Service (BUG-02)
- **File:** `src/feedback/service.py` (`list_alerts`)
- **Problem:** `list_alerts()` initialized SQL conditions with `s.blended_score >= settings.DECISION_THRESHOLD` (0.99). When `min_score` was passed from the UI slider (e.g., 0.50), it added `AND s.blended_score >= 0.50`. Because both were `AND`ed, scores below 0.99 were always filtered out.
- **Solution:** Changed logic to calculate `effective_min_score = min_score if min_score is not None else settings.DECISION_THRESHOLD`. Now, if a user specifies a lower threshold, that threshold takes precedence.
- **Regression Test:** Added `test_list_alerts_custom_min_score` in `tests/unit/test_feedback.py`.

---

### Fix 3: Constant-Time Authentication for Admin Endpoints (BUG-03)
- **File:** `src/serving/app.py` (`verify_admin_key`) and `src/common/config.py`
- **Problem:** `POST /admin/reload-model` accepted `x_admin_key: Optional[str] = Header(None)`, but the body of `reload_model()` never referenced `x_admin_key`. Any unauthenticated request could force model reloads.
- **Solution:**
  1. Added `ADMIN_API_KEY: str | None = os.getenv("ADMIN_API_KEY", None)` in `Settings`.
  2. Created FastAPI dependency `verify_admin_key`:
     - In production (`ENV in ('production', 'prod')`), mandates `ADMIN_API_KEY` configuration. If missing, returns 500 server misconfiguration. If header is missing or does not match, returns 401 Unauthorized.
     - Uses `secrets.compare_digest` to prevent timing attacks.
     - In development mode without `ADMIN_API_KEY`, emits an explicit audit log warning.
  3. Protected `POST /admin/reload-model` and `POST /admin/retrain` with `dependencies=[Depends(verify_admin_key)]`.
- **Regression Test:** Added `test_admin_endpoint_security_unauthorized` in `tests/unit/test_serving.py`.

---

### Fix 4: Implementation of Missing `/admin/retrain` Endpoint (BUG-04)
- **File:** `src/serving/app.py`
- **Problem:** `Architecture.md` specified administrative endpoints `/admin/reload-model` and `/admin/retrain`, but `/admin/retrain` was missing from `app.py`.
- **Solution:**
  1. Implemented `POST /admin/retrain` accepting optional `min_labels` and `drift_threshold` parameters.
  2. Calls `check_retrain_trigger()` and returns execution telemetry: `{"status": "evaluated", "trigger_fired": ..., "reason": ...}`.
  3. Synchronized `docs/api/openapi.json` using `app.openapi()` to guarantee schema parity.
- **Regression Test:** Added `test_admin_endpoint_security_authorized` in `tests/unit/test_serving.py` and verified schema sync in `test_openapi_schema_in_sync`.

---

### Fix 5: Restore XGBoost-Only Scoring (`BLEND_ALPHA=1.0`) (CONFIG-01)
- **Files:** `src/common/config.py`, `src/training/blend.py`
- **Problem:** While an initial audit explored `BLEND_ALPHA=0.85`, empirical evaluation on the warm-account hold-out test set ($N=389,097$) showed that static GNN scores monotonically degrade PR-AUC from 0.7757 down to 0.7500 (at $\alpha=0.85$) and 0.0022 (at $\alpha=0.0$). This occurs because 99.89% of origin accounts in PaySim are single-use burner accounts unseen in the training graph, and recipient accounts are legitimate merchants without fraud history.
- **Solution:**
  1. Set `BLEND_ALPHA = 1.0` in `src/common/config.py` for the active PaySim configuration.
  2. Hardened `compute_blend_score()` in `src/training/blend.py`: when `alpha >= 1.0`, it strictly returns `(float(xgb_score), False)`. GNN scores cannot accidentally shift the score or alter decision thresholds.
  3. Preserved all GNN model architectures, graph artifacts, and cold-start fallback interfaces for future dynamic streaming graph expansion.
- **Regression Tests:** Added `test_blend_alpha_one_results_in_xgboost_only_predictions`, `test_gnn_scores_cannot_alter_predictions_in_xgboost_only_config`, and `test_warm_and_cold_account_scoring_behavior` in `tests/unit/test_graph.py`.

---

### Fix 6: GitHub Actions Workflow Repair (CI-01)
- **File:** `.github/workflows/containers.yml`
- **Problem:** The `kubernetes-manifest-validation` job referenced `uses: yannh/kubeconform-action@v0.8.2`, which is not an action repository and failed on CI runners.
- **Solution:**
  1. Removed the invalid action reference block.
  2. Configured direct, pinned installation of `kubeconform` `v0.6.7` binary via curl and chmod.
  3. Executed `./kubeconform -summary -ignore-missing-schemas -schema-location default -schema-location '...' k8s/*.yaml`.
- **Regression Test:** Added `test_kubernetes_manifest_validation_workflow_configured` in `tests/unit/test_containers.py`.

---

## 3. Verification Summary

All 70 tests in the test suite pass with 0 failures:
- Unit tests: 70/70 PASS
- Code style & linting: `ruff check .` -> 0 errors
- Schema parity: `docs/api/openapi.json` -> 100% in sync
- CI workflow: Verified for Python 3.13, pinned Kubeconform v0.6.7, and multi-stage container validation
