# FraudShield — Repository Reconnaissance & Audit Report

**Auditor:** Senior AI/ML & MLOps Architect  
**Date:** 2026-11-03  
**Target Repository:** [https://github.com/rohanpawar0006/FraudShield](https://github.com/rohanpawar0006/FraudShield)  
**Host Environment:** Windows host, Python 3.13.14 x64, 16 GB RAM  

---

## 1. Executive Summary

This repository audit provides an exhaustive, code-level inventory and architecture verification of the **FraudShield** end-to-end financial transaction fraud prevention platform. Every claim in the root documentation (`PRD.md`, `Architecture.md`, `Phases.md`, `README.md`) was verified against the actual source code in `src/`, `docker/`, `k8s/`, and `tests/`.

The core tabular machine learning, Feast online/offline feature parity, FastAPI scoring microservice, Streamlit analyst dashboard, Evidently drift monitoring, and reproducible bootstrap workflows are robust and backed by an active 63-test pytest suite. However, this deep-dive inspection revealed four critical structural issues and several architectural gaps that require remediation to ensure production correctness, security, and closed-loop data flywheel integration.

---

## 2. Component Inventory

| Component / Layer | Primary Path(s) | Language / Framework | Implementation Status | Verified Capabilities |
|---|---|---|---|---|
| **Data Ingestion & Splits** | `src/ingestion/eda_and_split.py`, `download_data.py` | Python 3.13, Pandas, PyArrow | **PASS** | Temporal split (Train $\le 323$, Val $324..378$, Test $379..743$); zero target leakage. |
| **Feature Engineering & Feast** | `src/features/feature_pipeline.py`, `feature_repo/` | Python 3.13, Feast 0.67 | **PASS** | Point-in-time sliding window features; SQLite online store; verified offline-online parity test. |
| **Tabular ML Model** | `src/training/train_xgboost.py`, `data.py` | XGBoost 3.4.1, scikit-learn | **PASS** | Histogram-binned gradient boosting; class imbalance weighting (1224:1); PR-AUC 0.9239 on test. |
| **Graph Neural Network** | `src/graph/gnn_model.py`, `build_graph.py` | PyTorch Geometric 2.7.0, PyTorch 2.6 | **PARTIAL** | GraphSAGE edge classification and forwarding ring detection; extreme test cold-start degradation. |
| **Score Blending** | `src/training/blend.py`, `src/serving/scoring.py` | Python 3.13, NumPy | **PARTIAL** | Convex blend $S = \alpha S_{\text{xgb}} + (1-\alpha) S_{\text{gnn}}$; currently $\alpha=1.0$ (XGBoost identical). |
| **Streaming & Scoring API** | `src/serving/app.py`, `schemas.py`, `metrics.py` | FastAPI 0.128, Starlette, Pydantic v2 | **PARTIAL** | Real-time `POST /score` ($p_{95} \le 16\text{ms}$), async persistence queue, dead-letter queue; admin auth missing. |
| **Kafka Streaming Pipeline** | `src/ingestion/streaming.py`, `consumer.py`, `producer.py` | Python 3.13, Confluent Kafka | **PASS** | In-memory fallback channel and real Kafka streaming client; schema validation; dead-letter routing. |
| **Database Persistence** | `src/common/db.py`, `docker/init.sql` | PostgreSQL 16 / SQLite 3.45 | **PASS** | Dual-backend relational abstraction (`transactions`, `scores`, `cases`, `labels`), connection pooling. |
| **Analyst Dashboard** | `dashboard/streamlit_app.py`, `src/feedback/service.py` | Streamlit 1.55, Matplotlib | **PARTIAL** | Alert triage queue, tree SHAP contributions, case feedback; hardcoded threshold bug in filtering. |
| **Drift Monitoring** | `src/monitoring/drift.py` | Evidently AI 0.7.20 | **PASS** | Normalized Wasserstein distance and KS testing across 12 features; automated HTML/JSON reports. |
| **Retraining & Promotion** | `src/monitoring/retrain_trigger.py` | MLflow 3.17, XGBoost | **PARTIAL** | Frozen gate set ($N=15,000$), 1,000 bootstrap CI; safe labels are filtered but NOT merged into training set. |
| **Production Packaging** | `docker/`, `k8s/` | Docker Compose v2, K8s 1.30+ | **PASS** | Multi-stage slim Dockerfiles (Python 3.13), resource limits (< 4GB), StatefulSet, ConfigMaps, Secrets. |

---

## 3. Architecture & Entry Points

### Actual System Entry Points
1. **Scoring REST Service:** `src.serving.app:app` (via Uvicorn).
2. **Analyst Investigation Console:** `dashboard/streamlit_app.py` (via Streamlit).
3. **Kafka Streaming Consumer:** `src.ingestion.consumer:main` / `src.stream.consumer`.
4. **Kafka Replay Producer:** `src.ingestion.producer:main`.
5. **Scheduled Drift & Retrain Worker:** `src.monitoring.retrain_trigger:main` (CronJob / Worker).
6. **Platform Bootstrap & Setup:** `scripts/bootstrap.py` (CLI entry point with `--fast`, `--seed`, `--ci-mock`).

---

## 4. Critical Findings & Technical Debt

### Issue 1: Feedback Loop Retraining Disconnect (Critical Bug)
- **File:** `src/monitoring/retrain_trigger.py` (Lines 116–122).
- **Finding:** In `prepare_retraining_dataset()`, verified analyst labels from `src/feedback/service.py:get_labeled_cases()` are checked for gate-set leakage and filtered into `safe_labels`. However, `safe_labels` is **never merged or concatenated** into the returned training DataFrame `df_train`.
- **Architectural Impact:** The human-in-the-loop data flywheel is severed. Candidate models retrained in Phase 7 never learn from human analyst corrections.

### Issue 2: Dashboard Alert Filtering Obstruction (Critical Bug)
- **File:** `src/feedback/service.py` (Lines 55–68).
- **Finding:** `list_alerts()` hardcodes `conditions = ["s.blended_score >= " + ph]` initialized with `settings.DECISION_THRESHOLD` (0.99). When `min_score` is provided (e.g., analyst moves the slider down to 0.70), `conditions` appends a second clause, resulting in:
  `WHERE s.blended_score >= 0.99 AND s.blended_score >= 0.70`.
- **Architectural Impact:** Fraud analysts cannot triage lower-scoring or borderline transactions (scores 0.50–0.98), rendering the dashboard score slider inoperative.

### Issue 3: Unauthenticated Admin Operations (Security Vulnerability)
- **File:** `src/serving/app.py` (Lines 158–176).
- **Finding:** `POST /admin/reload-model` accepts `x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")`, but the parameter is completely unused. The endpoint never validates the key, allowing unauthenticated network callers to reload models arbitrarily.
- **Architectural Impact:** Severe operational security vulnerability in production environments.

### Issue 4: Documented but Missing `/admin/retrain` Endpoint (Specification Gap)
- **File:** `src/serving/app.py` & `Architecture.md` (Line 54).
- **Finding:** `Architecture.md` documents `POST /admin/retrain` for manual retraining triggering, but the endpoint does not exist in FastAPI `app.py`.
- **Architectural Impact:** API documentation drift and inability to trigger retraining checks via REST.

### Issue 5: Inert Graph Signal via Blind Global Alpha
- **Files:** `src/common/config.py`, `src/training/blend.py`.
- **Finding:** `BLEND_ALPHA = 1.0` was set globally to prevent test set precision dilution caused by extreme account turnover (99.95% cold-start in PaySim test split). While scientifically necessary for cold accounts, this completely suppresses GNN topology signals even when an account *is* known in the graph.
- **Architectural Impact:** PyG GNN features provide 0% contribution to serving decisions.

---

## 5. Dependency and Configuration Verification

- **Python Runtime:** Python 3.13.14 is the verified standard. All CI workflows and Docker base images are aligned to `python:3.13-slim-bookworm`.
- **Dependency Pinnings:** Key dependencies pinned in `requirements.txt`:
  - `numpy==2.5.4`
  - `pandas==2.2.3`
  - `xgboost==3.4.1`
  - `torch==2.6.0`
  - `torch-geometric==2.7.0`
  - `feast==0.67.0`
  - `mlflow==3.17.2`
  - `fastapi==0.128.0`
  - `evidently==0.7.20`
  - `streamlit==1.55.0`
- **Secrets Management:** Credentials in `k8s/02-secrets.yaml`, `.env.example`, and `docker-compose.yml` use local placeholders with 0 exposed production tokens.

