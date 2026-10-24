# Memory.md — Project State (update after every meaningful task)

## Project Status
- Current phase: **Phase 0 complete -> Phase 1 (Data & EDA in progress)**
- Overall completion: 15%
- Last completed task: Phase 0 Setup (Python 3.13 verification, pinned requirements, docker-compose, CI skeleton, git commits)
- Current task: Phase 1 — Data & EDA (Awaiting user Kaggle dataset download into data/raw/)
- Next task: Ingest PaySim CSV, run EDA analysis, document leakage review, create time-based train/val/test parquet splits

## Completed Work
[✓] PRD, Architecture, Rules, Phases, Design, Memory drafted
[✓] Phase 0 Setup (Folder structure, pinned requirements, compose file, smoke tests, CI)
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
| Python 3.13 verified & adopted | PyG 2.8.1, Feast 0.67.0, XGBoost 3.4.1, Evidently 0.7.23, MLflow 3.17.2, and PyTorch 2.14.1 all install cleanly with wheels on Python 3.13 on Windows | Python 3.11 Docker |
| Dataset: PaySim (assumed default) | Account IDs enable graph + replay; fraud rate < 0.2% | IEEE-CIS, Elliptic |
| Streamlit dashboard | Speed in 14 days | React |
| Postgres for cases/labels | Relational, simple | SQLite |
| GNN scores precomputed for serving | Keeps latency low | Online subgraph inference |
| docker-compose fallback beside Minikube | Demo safety | K8s only |

## Current Architecture
See Architecture.md (Kafka -> FastAPI -> Feast/XGB/GNN -> Postgres -> Streamlit; MLflow + Evidently alongside).

## Known Issues / Risks
| Issue | Severity | Status | Possible solution |
|---|---|---|---|
| Docker CLI not installed on host machine | High | Reported to user | User installs Docker Desktop / WSL2, or run services directly / mock mode for local tests |
| Python 3.13 compatibility | Low | Resolved | Verified: XGBoost, PyG, Feast, Evidently, MLflow, Torch import cleanly |
| Minikube RAM limits with Kafka+Redis+Postgres | Medium | Unverified | Subsample; docker-compose fallback |
| PaySim label leakage via balance fields | High | To check in Phase 1 | Drop/transform, document in eda.md |
| GNN may not beat XGBoost | Medium | Open | Report honestly; tune graph design |

## Technical Debt
None yet.

## Environment
- Python: 3.13.14 (AMD64)
- Services/ports (planned): Kafka 9092, Redis 6379, Postgres 5432, MLflow 5000, FastAPI 8000, Streamlit 8501.
- Env vars: via `.env` (configured in `.env.example`). Never store secrets in repo.
- Verified test commands:
  - `d:\FraudShield\.venv\Scripts\ruff.exe check .` -> Passed (0 errors)
  - `d:\FraudShield\.venv\Scripts\pytest.exe -v` -> Passed (5/5 tests in 0.89s)
  - `docker compose -f docker/docker-compose.yml up -d` -> Failed: Docker executable not found in PATH

## Recent Changes
- Created full project folder tree matching Architecture.md (.gitkeep in place).
- Generated `.env.example` with standard defaults and ports.
- Verified and pinned all dependencies in `requirements.txt` with one-line rationales.
- Verified PyTorch, PyTorch Geometric, Feast, XGBoost, Evidently, and MLflow on Python 3.13.
- Added GitHub Actions workflow `.github/workflows/ci.yml`.
- Initialized git repository and made small atomic commits (`docs:`, `chore:`, `feat:`).

## Next Steps
1. User downloads PaySim dataset from Kaggle (`PS_20174392719_1491204439457_log.csv`) and places it in `data/raw/`.
2. Generate Phase 1 EDA notebook (`notebooks/01_eda.ipynb` or script) covering class imbalance, amount/type distribution, and time structure.
3. Conduct leakage review and document dropped columns.
4. Export time-based train/val/test splits to `data/processed/` as Parquet, and record metrics in `docs/results/eda.md`.

## Context for the Next AI Session
Phase 0 is complete with all code committed to git. Python 3.13 has been verified with PyG, Feast, XGBoost, Evidently, and MLflow all installed cleanly in `.venv`. Docker CLI is missing on the host machine and has been reported to the user. Awaiting user dataset placement in `data/raw/` to run Phase 1. Follow Rules.md strictly (time-based splits only, PR-AUC headline metric, document leakage review).
