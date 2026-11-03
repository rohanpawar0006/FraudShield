# Memory.md — Project State (update after every meaningful task)

## Project Status
- Current phase: **Phase 9.1 complete — Consistency, Statistical Honesty & Deliverables Polished**
- Overall completion: 100% (Phases 0-7, 9, 9.1 fully verified locally; Phase 8 written, statically validated, not executed locally)
- Last completed task: Phase 9.1 — Latency disambiguation (two distinct benchmarks in `docs/results/latency_in_process.json` and `latency_http.json`), 1,000-resample bootstrap confidence intervals on promotion gate ($\Delta\text{PR-AUC}$), Kaggle PaySim bootstrap documentation and real-data verification, GNN naming alignment ("GNN Edge Risk Classifier alone"), secrets audit, and Phase 8 static validation status honesty.

## Completed Work
[✓] PRD, Architecture, Rules, Phases, Design, Memory drafted
[✓] Phase 0 Setup (Repo, dependencies, compose, CI done; Docker static acceptance verified)
[✓] Phase 1 Data & EDA
[✓] Phase 2 Features + Feast
[✓] Phase 3 XGBoost + MLflow
[✓] Phase 4 Streaming scoring
[✓] Phase 5 GNN + blend
[✓] Phase 6 Dashboard + feedback
[✓] Phase 7 Drift + retraining
[✓] Phase 8 Kubernetes & Containerization (Written & statically validated; not executed locally)
[✓] Phase 9 Polish & deliverables
[✓] Phase 9.1 Consistency, Statistical Honesty & Rigor (100% complete)

## Important Decisions
| Decision | Reason | Alternatives |
|---|---|---|
| Zero Torch in Serving Container | Serving uses fast tabular lookup for GNN scores; zero torch imports in serving code; keeps Docker image lean (~250 MB vs ~2.5 GB) | Bundling PyTorch in all containers |
| Multi-stage Docker Builds & Non-Root User | Security best practices: virtualenv built in builder stage, unprivileged `appuser` (UID 10001) in runner stage | Single-stage root containers |
| Docker Compose Profiles (`core`, `stream`, `full`) | Fits 8 GB RAM laptops by separating web/DB stack (`core`: ~2.8 GB) from streaming pipeline (`stream`: ~3.5 GB) | Monolithic docker-compose causing laptop OOM |
| Laptop-Sized Kubernetes Limits (Minikube) | Workloads request <= 250m CPU and <= 512Mi RAM each; total cluster requirements <= 4 CPU, <= 4 GB RAM | Production scale cloud clusters |
| Automated CI Container Verification (`containers.yml`) | Truthful local static validation + live container build and compose verification in GitHub Actions / Codespaces | Faking local Docker runs when tools are absent |
| Evidently 0.7.23 modern API pinned | Verified on Python 3.13; exports HTML & JSON summaries via `snapshot.save_html()` and `snapshot.save_json()` | Legacy evidently 0.2 API |
| Frozen Evaluation Gate Set (15,000 rows) | Fixed, representative sample from hold-out window (`data/processed/frozen_gate_set.parquet`) with strict zero-leakage assertion | Dynamic gate set (unstable baseline) |
| Selection Bias Documented | Analyst labels reflect high-scoring transactions ($S \ge 0.95$); documented in `docs/results/drift/selection_bias_notes.md` | Pretending analyst labels are i.i.d. |
| Honest Promotion Gate Rule | Candidate promoted to champion alias ONLY IF candidate PR-AUC $\ge$ champion PR-AUC + 0.0010 AND recall is not degraded | Unconditional candidate deployment |
| Serving Zero-Downtime Reload (`POST /admin/reload-model`) | Scoring service reloads newly promoted champion model in-place without restarting FastAPI process | Container restart |
| Benchmark DB Isolation | Benchmarks and replay smoke tests write to `data/fraudshield_benchmark.db`, eliminating score row count contamination in serving DB | Shared DB for tests and demo |
| DLQ Topic unified to `transactions.dlq` | Aligns code, config, compose, and tests strictly with prompt specification | Leaving mismatched `-` vs `.` |
| Precomputed GNN Risk Lookup | Per-account risk norm stored in `models/gnn_scores_lookup.parquet`; allows `< 1 ms` risk lookup during serving | Running PyG message passing per request |
| Off-path background DB persistence | `DatabaseManager` runs a background queue with batch inserts so persistence I/O never blocks scoring latency | Synchronous per-transaction DB commits |

## Current Architecture
See Architecture.md (Kafka -> FastAPI -> Feast online / XGBoost -> Postgres -> Streamlit; MLflow + Evidently alongside; Containerized via Docker & Kubernetes).

## Verification Status & Honest Tool Availability
| Tool / Component | Presence on Windows Host | Status | Verification Detail |
|---|---|---|---|
| `docker` | Not installed | STATICALLY VALIDATED | Dockerfiles validated via regex & multi-stage checks; CI via `.github/workflows/containers.yml` |
| `docker compose` | Not installed | STATICALLY VALIDATED | Valid YAML parsed, port collision verified, service dependency DAG checked |
| `kubectl` | Not installed | STATICALLY VALIDATED | Manifests validated via `yaml.safe_load_all`, selector matching verified, resource limits asserted |
| `minikube` | Not installed | NOT RUN - MISSING TOOL | Manifests sized for Minikube on laptop; step-by-step commands documented in `k8s/README.md` |
| Serving API | **Installed & Live** | **VERIFIED RUNNING LOCALLY** | Live unit tests, `/health` and `/score` endpoints verified; $p50 < 30\text{ ms}$ |
| Analyst Dashboard | **Installed & Live** | **VERIFIED RUNNING LOCALLY** | Streamlit AppTest passing; case investigation & feedback verified |
| Drift & Retraining | **Installed & Live** | **VERIFIED RUNNING LOCALLY** | Synthetic & natural drift detection verified; Candidate retraining & promotion gate verified |

## Test Suite Progress
- 63 tests passing across:
  - `test_common.py` (3)
  - `test_containers.py` (6)
  - `test_dashboard.py` (1)
  - `test_download_data.py` (1)
  - `test_feast_parity.py` (2)
  - `test_features.py` (6)
  - `test_feedback.py` (6)
  - `test_graph.py` (6)
  - `test_monitoring.py` (6)
  - `test_sanity.py` (1)
  - `test_schema_parity.py` (2)
  - `test_serving.py` (13)
  - `test_splits.py` (3)
  - `test_training.py` (7)
