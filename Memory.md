# Memory.md — Project State (update after every meaningful task)

## Project Status
- Current phase: **Phase 1 complete -> Ready for Phase 2 (Features + Feast)**
- Overall completion: 20%
- Last completed task: Phase 1 Data & EDA (EDA pipeline, leakage review, Parquet splits, zero-overlap unit test)
- Current task: Phase 2 — Features + Feast
- Next task: Feature engineering in `src/features/` (velocity txns/1h/24h, amount vs mean, destination change flags), Feast repo setup, offline/online parity test

## Completed Work
[✓] PRD, Architecture, Rules, Phases, Design, Memory drafted
[ ] Phase 0 Setup (Repo, dependencies, compose, CI done; Docker acceptance pending on host)
[✓] Phase 1 Data & EDA
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
| Python 3.13 verified & adopted | PyG 2.8.1, Feast 0.67.0, XGBoost 3.4.1, Evidently 0.7.23, MLflow 3.17.2, and PyTorch 2.14.1 all install cleanly with wheels on Python 3.13 on Windows | Python 3.11 Docker |
| Dataset: PaySim adopted | Account IDs enable graph + replay; fraud rate 0.1291% (< 1% requirement met) | IEEE-CIS, Elliptic |
| Dropped balance fields | `newbalanceOrig` (drained to 0 in 98.05% of fraud) and `newbalanceDest` dropped to prevent severe post-settlement target leakage | Using raw balances |
| Dropped `isFlaggedFraud` | Naive simulator rule (> 200k transfer) that causes collinearity/target leakage | Using rule flag |
| Time-based splits by `step` | Train (1..323, 70.15%), Val (324..378, 15.41%), Test (379..743, 14.44%) strictly ordered with zero overlap | Random splits (forbidden) |
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
| GNN may not beat XGBoost | Medium | Open | Report honestly; tune graph design |

## Technical Debt
None.

## Environment
- Python: 3.13.14 (AMD64)
- Verified test suite: 8 passed in 1.10s (`tests/unit/test_common.py`, `tests/unit/test_download_data.py`, `tests/unit/test_sanity.py`, `tests/unit/test_splits.py`)
- Linter: `ruff check .` -> All checks passed!
- Processed artifacts:
  - `data/processed/train.parquet` (180.5 MB, 4,463,587 rows, 3,643 frauds, rate 0.0816%)
  - `data/processed/val.parquet` (39.8 MB, 980,416 rows, 564 frauds, rate 0.0575%)
  - `data/processed/test.parquet` (37.4 MB, 918,617 rows, 4,006 frauds, rate 0.4361%)
  - `docs/results/eda.md` (Comprehensive EDA & Leakage Review report)
  - `notebooks/01_eda.ipynb` (Self-contained EDA Jupyter Notebook)

## Recent Changes
- Housekeeping: Deleted duplicate `files/` directory; committed to git.
- Downloaded and unzipped PaySim dataset directly to `data/raw/PS_20174392719_1491204439457_log.csv` (470 MB).
- Implemented `src/ingestion/eda_and_split.py` to automate EDA metrics, leakage review, and time-based splitting.
- Generated `data/processed/train.parquet`, `val.parquet`, and `test.parquet`.
- Generated `docs/results/eda.md` and `notebooks/01_eda.ipynb`.
- Created unit tests in `tests/unit/test_splits.py` verifying non-overlapping time boundaries.
- Marked Python 3.13 and Balance Leakage risks resolved in Memory.md; kept Docker on host as open.

## Next Steps
1. Transition to Phase 2 (Features + Feast).
2. Define velocity and aggregation features in `src/features/`.
3. Configure Feast feature repository and test parity between offline Parquet and online Redis.

## Context for the Next AI Session
Phase 0 and Phase 1 are complete. Processed time-based splits exist in `data/processed/` as Parquet files with zero time overlap verified by unit tests. Docker CLI is missing on the host and remains an open issue. Start Phase 2 per `Phases.md`. Follow `Rules.md` strictly (shared feature logic between offline and online paths, parity unit test required).
