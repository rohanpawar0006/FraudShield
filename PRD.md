# PRD.md — Real-Time Financial Fraud Detection Platform

> Legend: **[USER]** = given in the assignment brief · **[ASSUMPTION]** = reasonable assumption · **[REC]** = recommendation · **[FUTURE]** = future possibility

## 1. Product Overview
- **Name:** FraudShield (working title) [ASSUMPTION]
- **One-liner:** A system that reads live transactions, scores each for fraud risk in milliseconds, links suspicious accounts into fraud rings, and shows analysts a dashboard to review cases. [USER]
- **Vision:** Give a fraud-operations team a trustworthy, explainable, continuously improving risk engine instead of static rules.
- **Problem:** Fraud is rare (<1% of transactions), fast, and organised across linked accounts/devices. Per-transaction rules miss rings; batch models are too slow; models silently degrade as behaviour drifts.
- **Solution:** Streaming scoring pipeline (Kafka → FastAPI) using consistent features (Feast), a tabular model (XGBoost) blended with a graph model (GNN), drift monitoring (Evidently) with retraining, and an analyst feedback loop.

## 2. Target Users
- **Primary:** Fraud analysts — review alerts, confirm/dismiss cases, label outcomes.
- **Secondary:** ML engineers (retraining, monitoring), engineering manager / intern-review panel (evaluating the project).
- **Goals:** Few false alarms, fast triage, see *why* a transaction was flagged, see connected accounts.
- **Pain points:** Alert fatigue, no context on linked accounts, stale models, no feedback path to the model.

## 3. Goals & Objectives
- **Primary:** End-to-end working pipeline with measured PR-AUC and scoring latency. [USER]
- **Secondary:** Demonstrate fraud-ring detection beyond the tabular baseline; demonstrate drift → retrain loop; analyst labels feed retraining.
- **Technical objectives:** Identical offline/online features (Feast); all runs tracked (MLflow); deployment on Kubernetes (Minikube). [USER]
- **Career objective:** Deliver above-minimum work: tests, real numbers, clean repo. [USER]

## 4. Core Features

### MVP
| # | Feature | Description / Why | Inputs → Outputs | Depends on |
|---|---|---|---|---|
| M1 | Data + EDA | Load dataset, quantify imbalance, leakage review | Raw CSV → cleaned parquet, EDA notebook | Dataset |
| M2 | Feature engineering + Feast | Velocity, amount vs user mean, device/location change; same definitions train/serve | Transactions → feature views (offline + online) | M1 |
| M3 | XGBoost model | Class-weighted classifier, PR-AUC, threshold chosen on validation | Features → fraud probability | M2 |
| M4 | MLflow tracking | Params, metrics, artifacts, model registry for every run | Training runs → registered model | M3 |
| M5 | Streaming scoring | Kafka producer replays transactions; FastAPI consumer fetches features, scores | Transaction event → score event + stored result | M2, M3 |
| M6 | Analyst dashboard | Alerts list, case status, feedback labels | Stored scores → UI; labels → DB | M5 |
| M7 | Deployment | Services on Minikube; docker-compose fallback | Images → running cluster | M5, M6 |

### Post-MVP (still required by brief)
| # | Feature | Description |
|---|---|---|
| P1 | Graph + GNN | Accounts/devices/cards graph, PyG GraphSAGE/GAT; blended with XGBoost [USER] |
| P2 | Drift + retraining | Evidently reports; automatic retraining trigger [USER] |
| P3 | Feedback loop | Analyst labels merged into next training set [REC] |

### Future [FUTURE]
Explainability (SHAP) panel, real-time graph updates, A/B model shadowing, authentication/roles, alert routing.

## 5. User Stories
- As an analyst, I want to see a ranked list of high-risk transactions, so that I review the riskiest first.
- As an analyst, I want to see accounts linked to a flagged one, so that I can spot fraud rings.
- As an analyst, I want to mark a case confirmed/false-positive, so that the model learns from my decision.
- As an ML engineer, I want every run tracked in MLflow, so that results are reproducible.
- As an ML engineer, I want to be alerted when data drifts, so that I can retrain before accuracy drops.
- As a reviewer, I want a one-command demo and documented metrics, so that I can verify the claims.

## 6. Functional Requirements
- FR1 Ingest transactions from a Kafka topic at a configurable replay rate.
- FR2 For each event, fetch online features and return a fraud score in [0,1] via FastAPI.
- FR3 Blend XGBoost and GNN scores with a configurable weight; fall back to XGBoost-only if the graph score is unavailable.
- FR4 Persist every score, features snapshot reference, model version, and timestamp.
- FR5 Dashboard lists alerts above threshold, supports status (New / In review / Confirmed fraud / False positive) and label feedback.
- FR6 Evidently drift report generated on a schedule/on demand; retraining triggered when drift threshold exceeded.
- FR7 Retraining pipeline includes analyst labels and registers a new model version in MLflow.

## 7. Non-Functional Requirements
- **Performance:** target p95 scoring latency < 100 ms (API, excluding Kafka lag) [ASSUMPTION; measured, not promised].
- **Reliability:** consumer must not crash on malformed events; dead-letter handling.
- **Reproducibility:** pinned dependencies, seeds, MLflow-tracked runs.
- **Security:** no secrets in repo; input validated with Pydantic.
- **Privacy:** dataset is synthetic/anonymised; no real PII.
- **Observability:** structured logs, `/health`, `/metrics` (latency, throughput, score distribution).
- **Maintainability:** typed Python, tests, documented API.

## 8. Success Criteria
- PR-AUC reported on a **time-based** hold-out set, XGBoost vs blended compared in a table.
- Recall at fixed precision (e.g., 90%) reported.
- p50/p95 scoring latency measured and reported.
- Feast offline/online parity test passes.
- Drift injected → Evidently flags → retrain runs → new model registered (demonstrated on video).
- Demo runs from README instructions on a clean machine.
- ≥ 1 test suite passing in CI.

## 9. Constraints
- **Time:** 14 days effort, deadline 20 Nov 2026. [USER]
- **Tech:** Required stack — Kafka, XGBoost, PyTorch Geometric, Feast, MLflow, FastAPI, Kubernetes, Evidently. [USER]
- **Hardware:** Student laptop; Minikube resources limited [ASSUMPTION] → keep dataset subsampled where needed.
- **Budget:** Free/open-source only [ASSUMPTION].
- **Dataset:** One of five options given [USER]; PaySim chosen as default [ASSUMPTION — confirm with mentor].

## 10. Out of Scope
Real payment-gateway integration, production-grade auth/RBAC, multi-region/HA deployment, real customer data, regulatory compliance (PCI-DSS), mobile app, full explainability suite.

## 11. Deliverables (from brief) [USER]
Working demo, GitHub repo (clean commits, README, setup), architecture diagram + API/data docs, test results and metrics, 10-minute presentation, individual reflection.
