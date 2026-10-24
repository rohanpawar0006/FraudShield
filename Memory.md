# Memory.md — Project State (update after every meaningful task)

## Project Status
- Current phase: **Not started (documentation complete)**
- Overall completion: 0%
- Last completed task: Documentation system generated
- Current task: Phase 0 — Setup
- Next task: Create repo + docker-compose infra

## Completed Work
[✓] PRD, Architecture, Rules, Phases, Design, Memory drafted
[ ] Phase 0 Setup
[ ] Phase 1 Data & EDA
[ ] Phase 2 Features + Feast
[ ] Phase 3 XGBoost + MLflow
[ ] Phase 4 Streaming scoring
[ ] Phase 5 GNN + blend
[ ] Phase 6 Dashboard + feedback
[ ] Phase 7 Drift + retraining
[ ] Phase 8 Kubernetes
[ ] Phase 9 Polish & deliverables

## Important Decisions
| Decision | Reason | Alternatives |
|---|---|---|
| Dataset: PaySim [ASSUMPTION, confirm] | Account IDs enable graph + replay | IEEE-CIS (no clean account identity), Elliptic (graph-only) |
| Streamlit dashboard | Speed in 14 days | React |
| Postgres for cases/labels | Relational, simple | SQLite |
| GNN scores precomputed for serving | Keeps latency low | Online subgraph inference |
| docker-compose fallback beside Minikube | Demo safety | K8s only |

## Current Architecture
See Architecture.md (Kafka → FastAPI → Feast/XGB/GNN → Postgres → Streamlit; MLflow + Evidently alongside).

## Known Issues / Risks
| Issue | Severity | Status | Possible solution |
|---|---|---|---|
| Python 3.13 compatibility with PyG/Feast | Medium | Unverified | Use Python 3.11 in Docker |
| Minikube RAM limits with Kafka+Redis+Postgres | Medium | Unverified | Subsample; docker-compose fallback |
| PaySim label leakage via balance fields | High | To check in Phase 1 | Drop/transform, document |
| GNN may not beat XGBoost | Medium | Open | Report honestly; tune graph design |

## Technical Debt
None yet.

## Environment
- Services/ports (planned): Kafka 9092, Redis 6379, Postgres 5432, MLflow 5000, FastAPI 8000, Streamlit 8501.
- Env vars: via `.env` (not committed). **Never store secrets here.**
- Commands: `docker compose up -d`, `pytest`, `ruff check .` (to be confirmed in Phase 0)

## Recent Changes
- Docs created.

## Next Steps
1. Confirm dataset with mentor. 2. Phase 0 setup. 3. Phase 1 EDA.

## Context for the Next AI Session
Project: real-time fraud detection platform (internship deliverable, deadline 20 Nov 2026, ~14 days effort). Stack required: Kafka, XGBoost, PyG, Feast, MLflow, FastAPI, Kubernetes/Minikube, Evidently. Only documentation exists so far; no code. Start at Phase 0 in Phases.md. Follow Rules.md (time-based splits, PR-AUC, parity test, never claim untested work done). Protect the Phase 4 end-to-end milestone before attempting the GNN.
