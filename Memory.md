# Memory.md — Project State (update after every meaningful task)

## Project Status
- Current phase: **Phase 2 complete -> Ready for Phase 3 (XGBoost + MLflow)**
- Overall completion: 30%
- Last completed task: Phase 2 Features + Feast (Vectorized feature pipeline, Feast repository, train-serve parity test suite)
- Current task: Phase 3 — XGBoost + MLflow
- Next task: Baseline logistic regression, XGBoost with `scale_pos_weight`, PR-AUC & recall@precision evaluation on validation split, MLflow run tracking and model registry

## Completed Work
[✓] PRD, Architecture, Rules, Phases, Design, Memory drafted
[ ] Phase 0 Setup (Repo, dependencies, compose, CI done; Docker acceptance pending on host)
[✓] Phase 1 Data & EDA
[✓] Phase 2 Features + Feast
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
| Python 3.13 verified & adopted | PyG 2.8.1, Feast 0.67.0, XGBoost 3.4.1, Evidently 0.7.23, MLflow 3.17.2, and PyTorch 2.14.1 all install cleanly with wheels on Python 3.13 on Windows | Python 3.11 Docker |
| Dataset: PaySim adopted | Account IDs enable graph + replay; fraud rate 0.1291% (< 1% requirement met) | IEEE-CIS, Elliptic |
| Dropped balance fields | `newbalanceOrig` (drained to 0 in 98.05% of fraud) and `newbalanceDest` dropped to prevent severe post-settlement target leakage | Using raw balances |
| Dropped `isFlaggedFraud` | Naive simulator rule (> 200k transfer) that causes collinearity/target leakage | Using rule flag |
| Time-based splits by `step` | Train (1..323, 70.15%), Val (324..378, 15.41%), Test (379..743, 14.44%) strictly ordered with zero overlap | Random splits (forbidden) |
| Continuous feature state | Feature pipeline processes full chronological stream to preserve realistic historical state across train/val/test boundaries without boundary blindness | Independent split processing |
| Feast local SQLite for dev | Fast zero-skew verification and local CI execution; seamless switch to Redis in Docker/K8s | Redis-only |
| Streamlit dashboard | Speed in 14 days | React |
| Postgres for cases/labels | Relational, simple | SQLite |
| GNN scores precomputed for serving | Keeps latency low | Online subgraph inference |
| docker-compose fallback beside Minikube | Demo safety | K8s only |

## Current Architecture
See Architecture.md (Kafka -> FastAPI -> Feast/XGB/GNN -> Postgres -> Streamlit; MLflow + Evidently alongside).

## Known Issues / Risks
| Issue | Severity | Status | Possible solution |
|---|---|---|---|
| Docker CLI not installed on host machine | High | Open | User installs Docker Desktop / WSL2; mock or standalone fallback for testing |
| Python 3.13 compatibility | Low | Resolved | Verified: XGBoost, PyG, Feast, Evidently, MLflow, Torch import cleanly |
| Minikube RAM limits with Kafka+Redis+Postgres | Medium | Unverified | Subsample; docker-compose fallback |
| PaySim label leakage via balance fields | High | Resolved | Evaluated and resolved: `newbalanceOrig`, `newbalanceDest`, and `isFlaggedFraud` dropped |
| PaySim device/location absence | Low | Documented | No device/IP fields in PaySim; not invented artificially; addressed via graph connectivity in Phase 5 |
| GNN may not beat XGBoost | Medium | Open | Report honestly; tune graph design |

## Technical Debt
None.

## Environment
- Python: 3.13.14 (AMD64)
- Verified test suite: 16 passed in 16.20s (`tests/unit/test_common.py`, `tests/unit/test_download_data.py`, `tests/unit/test_feast_parity.py`, `tests/unit/test_features.py`, `tests/unit/test_sanity.py`, `tests/unit/test_splits.py`)
- Linter: `ruff check .` -> All checks passed!
- Processed feature artifacts:
  - `data/processed/train_features.parquet` (192.3 MB, 4,463,587 rows)
  - `data/processed/val_features.parquet` (42.2 MB, 980,416 rows)
  - `data/processed/test_features.parquet` (39.6 MB, 918,617 rows)
  - `docs/features.md` (Detailed feature documentation and Feast configuration)
  - `feature_repo/` (Configured Feast 0.67 repo with entities, sources, and feature views)

## Recent Changes
- Implemented `src/features/feature_pipeline.py` with point-in-time safe origin velocity (1h, 24h), amount-vs-mean ratio, time since previous transaction, and destination frequency features.
- Generated `train_features.parquet`, `val_features.parquet`, and `test_features.parquet` with int64 and float64 dtypes.
- Created `feature_repo/feature_store.yaml` and `feature_repo/features.py` with `nameOrig` and `nameDest` entities and feature views.
- Verified Feast repository initialization with `feast apply`.
- Built automated train-serve parity test suite in `tests/unit/test_feast_parity.py`.
- Built unit test suite for rolling window boundaries, cold-start handling, and duplicate timestamps in `tests/unit/test_features.py`.
- Authored comprehensive feature documentation in `docs/features.md`.

## Next Steps
1. Transition to Phase 3 (XGBoost + MLflow).
2. Build baseline logistic regression model.
3. Train XGBoost classifier with `scale_pos_weight` to address the 0.129% class imbalance.
4. Tune decision thresholds on the validation set targeting PR-AUC and recall@precision.
5. Track runs, metrics, parameters, and model artifacts in MLflow.

## Context for the Next AI Session
Phase 0, Phase 1, and Phase 2 are complete. Feature parquet files are saved in `data/processed/` with 7 engineered point-in-time features. Feast feature store is configured and train-serve parity verified with unit tests. Next is Phase 3 (XGBoost + MLflow). Follow `Rules.md` strictly (PR-AUC headline metric, time-based splits only, all runs logged to MLflow, fixed seeds).
