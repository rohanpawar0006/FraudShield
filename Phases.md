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
**Tasks:** - [ ] Baseline (logistic regression) - [ ] XGBoost with `scale_pos_weight`; try SMOTE comparison - [ ] PR-AUC, recall@precision, threshold selection on val - [ ] Register model in MLflow
**Acceptance:** all runs visible in MLflow; PR-AUC on test beats baseline; results table in `docs/results/`.

## Phase 4 — Streaming Scoring Service (Days 7–8)
**Tasks:** - [ ] Producer replay with rate control - [ ] FastAPI consumer + `/score`, `/health`, `/metrics` - [ ] Persist scores to Postgres - [ ] Dead-letter handling - [ ] Latency benchmark script
**Testing:** API contract tests; malformed-event test; latency p50/p95 recorded.
**Acceptance:** replay of ≥10k events end-to-end with scores stored; latency numbers documented. **(First demoable milestone — protect this.)**

## Phase 5 — Graph + GNN + Blend (Days 9–10)
**Tasks:** - [ ] Build account/device/card graph (training window only) - [ ] GraphSAGE/GAT in PyG, class-weighted loss - [ ] Evaluate GNN alone, XGBoost alone, blended - [ ] Integrate GNN score into serving with XGBoost-only fallback
**Acceptance:** comparison table (PR-AUC, recall@precision); blend integrated; fallback tested. If GNN doesn't beat baseline, report honestly with analysis.

## Phase 6 — Dashboard + Feedback (Day 11)
**Tasks:** - [ ] Alerts table, filters, case status - [ ] Case detail with linked accounts - [ ] Label feedback → Postgres - [ ] Retraining job reads labels
**Acceptance:** analyst can open an alert, change status, submit label; label appears in DB and next training set.

## Phase 7 — Drift + Retraining (Day 12)
**Tasks:** - [ ] Evidently data-drift report (reference vs recent window) - [ ] Threshold → retrain trigger - [ ] Promote only if better on hold-out - [ ] Drift injection demo script
**Acceptance:** injected drift triggers report + retrain + new registered model, recorded on video.

## Phase 8 — Kubernetes Deployment (Day 13)
**Tasks:** - [ ] Dockerfiles for services - [ ] K8s manifests; Minikube deploy of serving, dashboard, Kafka, Redis, Postgres - [ ] docker-compose fallback verified
**Acceptance:** `kubectl get pods` all Running; demo works via port-forward.

## Phase 9 — Polish & Deliverables (Day 14)
**Tasks:** - [ ] README with setup instructions - [ ] Architecture diagram + API docs - [ ] Test results + metrics tables - [ ] Demo recording - [ ] 10-min slides (problem, design, challenges, results, next steps) - [ ] Individual reflection
**Acceptance:** fresh-clone setup works; all deliverables in the Drive folder.

## Feature → Phase map
M1→P1 · M2→P2 · M3/M4→P3 · M5→P4 · P1 (GNN)→P5 · M6/P3→P6 · P2 (drift)→P7 · M7→P8
