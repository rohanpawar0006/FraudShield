# FraudShield — End-to-End Architecture Verification

**Generated**: 2026-11-03  
**Auditor**: Senior Backend & MLOps Architect  
**Objective**: Trace a financial transaction through all 11 stages of the FraudShield platform to verify integration readiness.

---

## 1. End-to-End Transaction Lifecycle Trace

```
                     [1. INGESTION]
               TransactionPayload (Pydantic)
                            │
                            ▼
          [2. FEATURE RETRIEVAL & ENRICHMENT]
         Feast Online Store (or Safe Defaults)
                            │
                            ▼
           [3. FASTAPI SCORING ENDPOINT /score]
                            │
                            ▼
               [4. XGBOOST MODEL INFERENCE]
              Raw Tabular Fraud Probability
                            │
                            ▼
              [5. RISK SCORE & ALERT DECISION]
         Fraud Score + Threshold Check (> 0.99)
             (BLEND_ALPHA = 1.0 Active Config)
              /                               \
             /                                 \
            ▼                                   ▼
 [6. ASYNC DB PERSISTENCE]             [7. STREAMING / DLQ]
 Background Queue -> SQLite            Kafka / Memory Channel
   (Scores & Cases Tables)             (Scores Topic / DLQ)
            │                                   │
            ▼                                   ▼
[8. DASHBOARD ALERT DISPLAY]           [Consumer Ingestion]
  Streamlit Triage Interface
            │
            ▼
[9. FORENSIC CASE INVESTIGATION]
    Account History & Details
            │
            ▼
[10. MODEL EXPLAINABILITY (SHAP)]
 Tree SHAP Feature Attributions
            │
            ▼
[11. FEEDBACK & RETRAINING LOOP]
  Analyst Disposition (0 or 1)
       -> Audit Log Entry
       -> Zero-Leakage Gate Retraining Set
```

---

## 2. Stage-by-Stage Verification Matrix

### Stage 1 — Transaction Ingestion
- **Implementation**: [`src/serving/schemas.py`](file:///d:/FraudShield/src/serving/schemas.py#L10-L77) (`TransactionPayload`).
- **Input Schema**:
  ```json
  {
    "txn_id": "string",
    "step": 1,
    "type": "TRANSFER",
    "amount": 950000.0,
    "nameOrig": "C2000000002",
    "oldbalanceOrg": 950000.0,
    "nameDest": "C3000000003",
    "oldbalanceDest": 0.0
  }
  ```
- **Constraints Enforced**:
  - `type` validated against `PAYMENT_TYPES` (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`).
  - `amount >= 0.0` validated with custom validator.
  - Required fields checked; aliases `account_id` and `dest_id` accepted.
- **Error Handling**: Missing required fields or negative amounts trigger FastAPI `RequestValidationError` with HTTP 422 response.
- **Verification Status**: **PASS** (Verified with valid, missing, negative, and invalid-type payloads).

---

### Stage 2 — Feature Retrieval & Transformation
- **Implementation**: [`src/serving/scoring.py`](file:///d:/FraudShield/src/serving/scoring.py#L90-L150) (`ScoringEngine._enrich_features`).
- **Feast Integration**: Queries `feature_store.get_online_features` for entity `account_id`.
- **Default Fallback**: When Feast misses or is unreachable, injects `DEFAULT_FEATURE_VALUES` (e.g., `avg_amount_7d`, `txn_count_24h`, `dest_fan_in_7d`).
- **Scikit-Learn Preprocessing**: Transforms raw and enriched features through `models/xgboost_preprocessor.joblib` to produce the dense model input matrix.
- **Verification Status**: **PASS** (Verified feature fallback flags and output dimension consistency).

---

### Stage 3 — FastAPI Scoring Endpoint (`/score`)
- **Implementation**: [`src/serving/app.py`](file:///d:/FraudShield/src/serving/app.py#L73-L128) (`score_transaction`).
- **Execution Model**: Asynchronous endpoint with off-request-path background task scheduling.
- **Response Schema** (`ScoreResponse`):
  ```json
  {
    "transaction_id": "txn_demo_fraud_001",
    "fraud_score": 0.999354,
    "is_alert": true,
    "latency_ms": 7.34,
    "feature_fallback": true,
    "model_version": "FraudShield-XGBoost:1",
    "threshold": 0.99,
    "xgb_score": 0.999354,
    "gnn_score": null,
    "blend_score": 0.999354,
    "blend_used": false
  }
  ```
- **Verification Status**: **PASS** (Verified end-to-end HTTP response and schema match).

---

### Stage 4 — XGBoost Inference
- **Implementation**: [`src/serving/scoring.py`](file:///d:/FraudShield/src/serving/scoring.py#L180-L240) (`ScoringEngine.score`).
- **Model Loading**: Champion model loaded from MLflow Model Registry (`models:/FraudShield-XGBoost/1`) with fallback to `models/xgboost_model.json`.
- **Thread Safety**: Model inference is configured with `n_jobs=1` to eliminate inter-thread contention.
- **Inference Latency**: Sub-10ms per transaction in-process.
- **Verification Status**: **PASS** (Champion booster correctly evaluates probabilities in $[0.0, 1.0]$).

---

### Stage 5 — Risk Score Generation & Alerting Decision
- **Implementation**: [`src/training/blend.py`](file:///d:/FraudShield/src/training/blend.py#L42-L52) (`compute_blend_score`).
- **Active Configuration**: `BLEND_ALPHA = 1.0`. Hardened so that when `alpha >= 1.0`, `blend_score` strictly equals `xgb_score` and `blend_used` is `False`.
- **Alert Decision**: Evaluated against `DECISION_THRESHOLD = 0.99`.
  - Normal Transaction (\$45.50 PAYMENT): Score $0.0011 \implies$ `is_alert: false`.
  - Suspicious Transaction (\$950,000 TRANSFER): Score $0.9994 \implies$ `is_alert: true`.
- **Verification Status**: **PASS** (Zero GNN score leakage or dilution into final scores; correct alerting).

---

### Stage 6 — Database Persistence
- **Implementation**: [`src/common/db.py`](file:///d:/FraudShield/src/common/db.py#L203-L250) (`DatabaseManager.save_transaction_and_score`).
- **Architecture**: Enqueues persistence payloads into a bounded in-memory worker queue (`_write_queue`). A background thread batches and writes records off the critical request path.
- **Tables Updated**:
  - `transactions`: Stores `txn_id`, `account_id`, `dest_id`, `amount`, `type`, `ts`, and `raw` JSON payload.
  - `scores`: Stores `txn_id`, `xgb_score`, `gnn_score`, `blended_score`, `model_version`, `latency_ms`.
  - `cases`: Auto-upserted when an alert is queried or reviewed.
- **Verification Status**: **PASS** (Verified background writer flush and persistent records in SQLite).

---

### Stage 7 — Streaming & Dead-Letter Queue (DLQ)
- **Implementation**: [`src/ingestion/producer.py`](file:///d:/FraudShield/src/ingestion/producer.py) and [`src/ingestion/consumer.py`](file:///d:/FraudShield/src/ingestion/consumer.py).
- **Dead-Letter Queue Behavior**:
  - Valid transaction $\to$ Validated, scored, persisted to database, score emitted to `scores` topic.
  - Malformed payload (negative amount, missing required keys) $\to$ Intercepted by Pydantic validation and routed to `transactions.dlq` with failure reason and timestamp.
- **Verification Status**: **PASS** (Verified with `MemoryStreamingChannel` test harness).

---

### Stage 8 — Dashboard Alert Display
- **Implementation**: [`dashboard/streamlit_app.py`](file:///d:/FraudShield/dashboard/streamlit_app.py) and [`src/feedback/service.py`](file:///d:/FraudShield/src/feedback/service.py#L31-L105) (`list_alerts`).
- **Features**: Real-time filtering by status (`New`, `Under Review`, `Confirmed Fraud`, `False Positive`), score bounds (`min_score`, `max_score`), and transaction type.
- **Header KPIs**: Calculated via `get_alerts_summary_kpis()` showing total alerts, percentage reviewed, and false positive rates.
- **Verification Status**: **PASS** (Alert triage queries verified against persisted transactions).

---

### Stage 9 — Case Forensic Investigation
- **Implementation**: [`dashboard/streamlit_app.py`](file:///d:/FraudShield/dashboard/streamlit_app.py#L227-L350) and [`src/feedback/service.py`](file:///d:/FraudShield/src/feedback/service.py#L150-L184) (`get_case`).
- **Forensic Detail**: Displays origin account balance before/after, destination balance before/after, transaction type, simulation step, and counterparty transaction history via `get_account_history()`.
- **Verification Status**: **PASS** (Deep-dive view correctly reconstructs full transaction context).

---

### Stage 10 — Model Explainability (Tree SHAP)
- **Implementation**: [`src/feedback/service.py`](file:///d:/FraudShield/src/feedback/service.py#L408-L463) (`get_feature_contributions`).
- **Mechanism**: Invokes XGBoost's native `booster.predict(dmat, pred_contribs=True)` to extract exact Tree SHAP values (log-odds shifts) for every feature.
- **Suspicious Transaction (\$950,000 TRANSFER) Output**:
  - `oldbalanceOrg`: $+3.3969$ (High balance being drained $\implies$ strong risk factor).
  - `amount`: $+1.7178$ (Outsized transaction amount $\implies$ risk factor).
  - `oldbalanceDest`: $+1.1417$ (Recipient balance context).
  - `type_CASH_OUT`: $-1.0624$.
  - `type_TRANSFER`: $+0.8321$.
- **Verification Status**: **PASS** (Features match XGBoost inference model directly without synthetic approximations).

---

### Stage 11 — Feedback Submission & Retraining Dataset Integration
- **Implementation**: [`src/feedback/service.py`](file:///d:/FraudShield/src/feedback/service.py#L185-L288) (`submit_feedback`) and [`src/monitoring/retrain_trigger.py`](file:///d:/FraudShield/src/monitoring/retrain_trigger.py#L100-L160) (`prepare_retraining_dataset`).
- **Audit Trail**: Submitting disposition appends an immutable record to the `labels` table (`disposition`, `analyst`, `comment`, `labeled_at`).
- **Retraining Integration**:
  - `get_labeled_cases()` returns verified ground-truth records (`disposition IN ('Confirmed Fraud', 'False Positive')`).
  - `prepare_retraining_dataset()` checks for leakage overlap against the frozen gate set ($N=15,000$).
  - Overlap with frozen gate set triggers an immediate fatal `ValueError`. Zero overlap allowed.
  - Safe labels are incorporated into the candidate training dataset.
- **Verification Status**: **PASS** (End-to-end feedback capture, audit log persistence, and retraining pool integration verified).

