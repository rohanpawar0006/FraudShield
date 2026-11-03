# Phases.md — Implementation Roadmap (14 working days)

Do not start Phase N+1 until Phase N's acceptance criteria pass.

## Phase 0 — Setup (Day 1)
**Objective:** Repo, environment, docs, dataset in place.
**Tasks:** - [x] Create repo + folder structure - [x] Pin dependencies; verify Python/PyG/Feast install - [x] Docker-compose with Kafka, Redis, Postgres, MLflow - [x] Download dataset, gitignore data - [x] CI skeleton (ruff + pytest)
**Acceptance:** `docker compose up` brings all infra services healthy (pending — Docker not installed on host); test suite passes (5/5 passed).
**Docs:** Memory.md, README.

## Phase 1 — Data & EDA (Days 1–2)
**Objective:** Understand data and imbalance; define splits.
**Tasks:** - [x] EDA notebook (class balance, amounts, types, time) - [x] Leakage review documented - [x] Time-based train/val/test split saved as parquet
**Testing:** split has no time overlap (verified in `tests/unit/test_splits.py`); fraud rate per split recorded.
**Acceptance:** EDA notebook (`notebooks/01_eda.ipynb`) + `docs/results/eda.md` with fraud rate (<1% noted): **PASSED**.

## Phase 2 — Features + Feast (Days 3–4)
**Tasks:** - [x] Velocity (txns/1h/24h per account) - [x] Amount vs account running mean - [x] Destination history flags (documented PaySim device/location limitations) - [x] Feast repo, offline store, materialization verified
**Testing:** **offline vs online parity test** on sampled accounts (`tests/unit/test_feast_parity.py` passed).
**Acceptance:** parity test passes; features retrievable online: **PASSED**.

## Phase 3 — XGBoost + MLflow (Days 5–6)
**Tasks:** - [x] Baseline (logistic regression) - [x] XGBoost with `scale_pos_weight = 1224.25`; sequential candidate tuning - [x] PR-AUC, recall@precision, threshold selection on val ($t = 0.99$) - [x] Register model in MLflow (`FraudShield-XGBoost` v1) - [x] Strictly preserved hold-out test evaluation
**Acceptance:** all runs visible in MLflow; PR-AUC on test (0.9238) beats baseline (0.1455); results table in `docs/results/models.md`: **PASSED**.

## Phase 4 — Streaming Scoring Service (Days 7–8)
**Tasks:** - [x] Producer replay with rate control - [x] FastAPI consumer + `/score`, `/health`, `/metrics` - [x] Persist scores to Postgres (with SQLite fallback) - [x] Dead-letter handling - [x] Latency benchmark script
**Testing:** API contract tests; malformed-event test; latency p50/p95 recorded (35/35 pytest unit tests passed).
**Acceptance:** replay of 10,000 hold-out events end-to-end with scores stored; 25 injected malformed payloads safely routed to DLQ; 0 crashes; scoring latency p50 = 10.12 ms and p95 = 20.82 ms (target < 100 ms met): **PASSED**. **(First demoable milestone — verified.)**

## Phase 5 — Graph + GNN + Blend (Days 9–10)
**Tasks:** - [x] Build account graph (training window only, $step \le 323$) - [x] GraphSAGE in PyG, class-weighted loss, MLflow registration (`FraudShield-GNN`) - [x] Evaluate GNN alone, XGBoost alone, blended - [x] Integrate GNN score into serving with XGBoost-only fallback - [x] Identify fraud rings and forwarding mule accounts
**Acceptance:** comparison table (`docs/results/phase5_comparison.md` + CSV); blend integrated into `POST /score`; GNN fallback tracked in `/metrics`; 42/42 tests passing; HTTP p50=29.0 ms, p95=96.4 ms (<100 ms met). Honest topological sparsity analysis documented: **PASSED**.

## Phase 6 — Dashboard + Feedback (Day 11)
**Tasks:** - [x] Alerts table, filters, case status - [x] Case detail with linked accounts & Tree SHAP feature contributions - [x] Label feedback → Postgres/SQLite with full audit trail - [x] Retraining job reads labels (`get_labeled_cases`)
**Acceptance:** analyst can open an alert, change status, submit label; label appears in DB and next training set; AppTest smoke test passed; 49/49 pytest tests passing; 0 ruff errors: **PASSED**.

## Phase 7 — Drift + Retraining (Day 12)
**Tasks:** - [x] Evidently data & prediction drift report (training reference vs recent slice) - [x] Dual retrain trigger (drift share threshold OR verified analyst labels) - [x] Frozen evaluation gate set (`data/processed/frozen_gate_set.parquet`) with strict zero-leakage enforcement - [x] Selection bias documentation (`docs/results/drift/selection_bias_notes.md`) - [x] Candidate retraining & promotion gate with champion alias + reload endpoint (`POST /admin/reload-model`) - [x] End-to-end drift injection and evaluation demo (`scripts/demo_drift.py`)
**Testing:** 6 new monitoring unit tests (drift detection, trigger logic, zero-leakage violation assertion, honest gate promotion/rejection, reload endpoint); full test suite 56/56 tests passing.
**Acceptance:** End-to-end demo executed: control window yields 0% feature drift; synthetic drift flags 33.3% feature drift, triggers retrain, trains candidate XGBoost, evaluates on 15,000-row frozen gate set, executes honest promotion gate decision ([REJECTED] due to champion superiority: 0.9607 vs 0.9484 PR-AUC), logs to `retrain_audit_log.json`: **PASSED**.

## Phase 8 — Production Packaging & Orchestration (Day 13)
**Tasks:** - [x] Multi-stage Dockerfiles for serving, dashboard, stream, drift (pinned `python:3.13-slim-bookworm`, non-root user `appuser`, zero PyTorch in serving) - [x] docker-compose with `core`, `stream`, and `full` resource-bounded profiles (< 4 GB RAM) - [x] Complete K8s manifests in `k8s/` (Namespace, ConfigMap, Secrets, Postgres StatefulSet + PVC, Serving Deployment replicas: 1, Dashboard Deployment, Drift CronJob, Ingress) - [x] Redis removed from compose and k8s; documented local Feast SQLite store - [x] CI pipeline (`.github/workflows/containers.yml`) - [x] Static validation and container unit tests (`tests/unit/test_containers.py`)
**Testing:** 6 container & k8s static unit tests; YAML validity, selector-to-label parity, resource limits, health probes verified. Full test suite passing (63/63 tests).
**Acceptance:** Written and statically validated; not executed locally. Unmet acceptance criteria: running Docker daemon, Minikube VM, live pod scheduling, and port-forwarding verification (host does not have Docker or Minikube installed).

## Phase 9 — Polish & Deliverables (Day 14)
**Tasks:** - [x] Production README with pitch, Mermaid architecture, quickstart, results table, tradeoffs, verification matrix - [x] Comprehensive architecture docs (`docs/ARCHITECTURE.md`) with 4 Mermaid diagrams - [x] REST API reference (`docs/api/API.md`) and OpenAPI JSON specification (`docs/api/openapi.json`) - [x] Automated results generator (`scripts/make_results_summary.py`) reading artifacts into `docs/results/RESULTS.md` - [x] High-resolution figures in `docs/figures/` (PR curves, model comparison, latency tail, drift share) - [x] Reproducible clean bootstrap script (`scripts/bootstrap.py`, `.ps1`, `.sh`) verified in fresh directory - [x] 12-slide executive presentation outline (`docs/presentation/DECK_OUTLINE.md`) with speaker notes and citations - [x] Live 5–7 minute walkthrough demo script (`docs/DEMO_SCRIPT.md`) with pre-flight checklist and fallback plan - [x] Code hygiene pass: 63/63 pytest tests passing, 0 ruff errors, 0 secret exposures
**Acceptance:** Fresh-clone bootstrap script executed and verified in temp directory in under 60 seconds; 63/63 unit and integration tests passing; all deliverables complete: **PASSED**.

## Phase 9.1 — Consistency, Statistical Honesty & Rigor (Follow-up Polish)
**Tasks:**
- [x] **Latency Disambiguation:** Isolated two rigorous benchmarks with single-core inference (`n_jobs=1`) on an idle machine: Benchmark 1 (in-process `ScoringEngine.score()`, Mean: 13.38 ms, p50: 12.94 ms, p95: 17.80 ms, p99: 21.10 ms) and Benchmark 2 (HTTP POST `/score` via TestClient, Mean: 11.83 ms, p50: 11.33 ms, p95: 15.80 ms, p99: 18.79 ms; Real Uvicorn over TCP loopback: p50: 9.19 ms, p95: 12.96 ms). Both comfortably meet the PRD performance target (< 100 ms). Saved raw artifacts to `docs/results/latency_in_process.json`, `latency_http.json`, and `latency_all_trials.json`.
- [x] **Retrain Gate & Statistical Power:** Added 1,000-resample bootstrap confidence interval on $\Delta\text{PR-AUC}$. Documented that at $N_{\text{fraud}}=230$ on the frozen gate set ($SE \approx 0.013$), small gains cross zero at 95% confidence. Verified dual scenarios: Scenario 1 rejected (candidate vs champion v1: $\Delta = +0.0040$, 95% CI $[-0.0038, +0.0127]$ crosses zero); Scenario 2 promoted (candidate vs weakened champion: $\Delta = +0.6568$, 95% CI $[+0.6022, +0.7095]$ strictly positive).
- [x] **Dataset Split Verification:** Confirmed verified split counts across the 6,362,620 PaySim universe: Training (steps 1..323) = 4,463,587 rows, Validation (steps 324..378) = 980,416 rows, Hold-out Test (steps 379..743) = 918,617 rows.
- [x] **Bootstrap Honesty:** Documented that `bootstrap.py --fast` verification validates an already-populated workspace (pre-existing parquet splits, Feast SQLite store, MLflow registry artifacts); a full from-scratch reproduction on raw data has not been run end to end. CI tests ran in `--ci-mock` mode with synthetic data.
- [x] **GNN Naming Consistency:** Renamed GNN references from "node embeddings alone" to "GNN Edge Risk Classifier alone", stating that the blended model with $\alpha=1.0$ is identical to XGBoost by construction.
- [x] **Secrets & File Hygiene:** Verified all credentials in `k8s/02-secrets.yaml`, `docker-compose.yml`, and `.env.example` are local placeholders. Verified `feature_repo/data/online_store.db` is gitignored.
**Acceptance:** All Phase 9.1 consistency and honesty requirements verified across code, docs, figures, and test suite: **PASSED**.

## Feature → Phase map
M1→P1 · M2→P2 · M3/M4→P3 · M5→P4 · P1 (GNN)→P5 · M6/P3→P6 · P2 (drift)→P7 · M7→P8
