# FraudShield — Enterprise Real-Time Fraud Detection & MLOps Platform

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-63%20passed-brightgreen.svg)](tests/)
[![PR-AUC](https://img.shields.io/badge/Hold--Out%20PR--AUC-0.9239-success.svg)](docs/results/RESULTS.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving%20Sub--20ms-teal.svg)](docs/api/API.md)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**FraudShield** is a production-grade, end-to-end real-time financial fraud detection and operations platform. It couples tabular gradient-boosted decision trees (XGBoost) and graph topology signals (PyTorch Geometric) over a low-latency streaming pipeline (Kafka, FastAPI, Feast, PostgreSQL/SQLite) to achieve **PR-AUC 0.9239** and **95.95% Precision at 73.27% Recall** on strictly held-out PaySim financial data. Engineered with zero future data leakage, FraudShield delivers sub-20ms p95 HTTP scoring, an interactive Streamlit analyst dashboard with human-in-the-loop feedback loops, automated Evidently drift monitoring, and MLflow model registry promotion gates that retrain and hot-reload champion models with zero downtime.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion & Streaming Layer"]
        Replay["Transaction Replay Producer<br/>(scripts/replay_transactions.py)"]
        KafkaTxn["Kafka Topic: transactions<br/>(KRaft Single Broker)"]
        KafkaDLQ["Kafka Topic: transactions.dlq<br/>(Dead Letter Queue)"]
    end

    subgraph Serving["Serving & Inference Layer (FastAPI)"]
        Consumer["Streaming Consumer Worker<br/>(src/streaming/consumer.py)"]
        FastAPI["FastAPI Scoring Service<br/>(POST /score, GET /health, /metrics)"]
        Engine["ScoringEngine<br/>(Single-threaded n_jobs=1)"]
        Preprocessor["Preprocessor Transformer<br/>(OneHotEncoder + Scaler)"]
        ChampionXGB["Champion Booster<br/>(MLflow Registered @champion)"]
        GNNLookup["GNN Mule / Ring Risk Table<br/>(Account ID Map)"]
    end

    subgraph FeatureStore["Feature Layer (Feast)"]
        FeastOffline["Parquet Offline Store<br/>(data/processed/)"]
        FeastOnline["SQLite Online Store<br/>(feature_repo/data/online_store.db)"]
    end

    subgraph Storage["Persistence & Registry"]
        MLflow["MLflow Tracking & Registry<br/>(sqlite:///mlruns.db)"]
        DB["PostgreSQL / SQLite Storage<br/>(transactions, scores, cases, labels)"]
    end

    subgraph Operations["Human-in-the-Loop & Monitoring"]
        Dashboard["Streamlit Analyst Dashboard<br/>(Alerts, Case Review, Feedback)"]
        Evidently["Evidently Drift Monitor<br/>(Wasserstein & KS Test Pipeline)"]
        RetrainTrigger["Retraining & Gate Set Evaluator<br/>(src/monitoring/retrain_trigger.py)"]
    end

    %% Streaming Flow
    Replay -->|JSON Events| KafkaTxn
    KafkaTxn -->|Consume Batch| Consumer
    Consumer -->|Process or Error| KafkaDLQ
    Consumer -->|Score Call| FastAPI

    %% Serving Dependencies
    FastAPI --> Engine
    Engine --> Preprocessor
    Engine --> ChampionXGB
    Engine --> GNNLookup
    Engine -->|Online Features| FeastOnline
    FastAPI -->|Async Persistence Queue| DB

    %% Model Tracking
    MLflow -->|Load Champion Booster| ChampionXGB

    %% Analyst & Monitoring
    DB -->|Read Alerts & Cases| Dashboard
    Dashboard -->|Submit Ground Truth Labels| DB
    DB -->|Read Recent Scores & Txns| Evidently
    Evidently -->|Trigger on Drift| RetrainTrigger
    DB -->|Read Labeled Data| RetrainTrigger
    RetrainTrigger -->|Register & Flip Alias| MLflow
    RetrainTrigger -->|POST /admin/reload-model| FastAPI
```

Detailed architectural diagrams, sequence diagrams, and split timelines are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## ⚡ Quickstart (Native Python)

FraudShield is designed to bootstrap completely from a fresh repository clone with zero external infrastructure required.

### 1. Prerequisites
- **Python:** 3.11, 3.12, or 3.13 (Python 3.13 recommended)
- **Git**

### 2. Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/FraudShield.git
cd FraudShield

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Automated One-Command System Bootstrap
Initialize the entire platform end-to-end (directories, feature store materialization, model validation, MLflow registry setup, SQLite database schema, OpenAPI schema, and optional demo analyst alerts):

> **Dataset & Bootstrap Honesty Note:**
> - **Workspace Validation:** The `bootstrap.py --fast` verification validates an already-populated workspace (pre-existing parquet splits, Feast SQLite store, MLflow registry artifacts); full from-scratch reproduction on real raw data has not been run end to end.
> - **Real PaySim Dataset:** To evaluate with full real data from scratch, obtain the 493.5 MB PaySim CSV from Kaggle (`kaggle datasets download -d ealaxi/paysim1`) and place it at `data/raw/PS_20174392719_1491204439457_log.csv`.
> - **Synthetic Fallback:** If the raw file is not present, `bootstrap.py` synthesizes realistic transactions adhering strictly to the PaySim schema and distribution.
> - **CI Testing:** In automated GitHub Actions CI, tests run with `--ci-mock` (5,000 synthetic records) to execute in under 30 seconds without large artifact transfers.

```bash
# Windows PowerShell wrapper:
.\scripts\bootstrap.ps1 --fast --seed

# Or Linux / macOS:
./scripts/bootstrap.sh --fast --seed

# Or direct Python invocation:
python scripts/bootstrap.py --fast --seed
```

### 4. Running Platform Services

#### A. Start the Scoring API Microservice
```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```
- API Docs (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
- OpenAPI Specification: [`docs/api/openapi.json`](docs/api/openapi.json)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

#### B. Launch the Streamlit Fraud Analyst Dashboard
In a separate terminal:
```bash
streamlit run src/dashboard/app.py --server.port 8501
```
Navigate to [http://localhost:8501](http://localhost:8501) to explore real-time alert triage, inspect account network rings, and submit analyst ground-truth labels.

#### C. Run Test Suite & Serving Benchmark
```bash
# Run all 63 unit, integration, and parity tests
pytest -q

# Run real-time serving replay benchmark (2,000 hold-out transactions)
python scripts/smoke_test_serving.py --benchmark
```

#### D. Run Automated Drift & Retraining Demo
Demonstrate the automated drift detection and retraining promotion lifecycle:
```bash
# Fast demonstration with subsampled dataset
python scripts/demo_drift.py --fast

# Full production run (trains candidate on 4,463,587 rows + analyst labels)
python scripts/demo_drift.py --full
```

---

## 📊 Key Results Summary

All metrics are harvested dynamically from project artifacts without hand-typed estimates. See the full report at [docs/results/RESULTS.md](docs/results/RESULTS.md).

| Architecture / Benchmark | PR-AUC (Avg Precision) | ROC-AUC | Precision @ t | Recall @ t | Recall @ P>=0.90 | Alerts Generated | Decision Threshold |
|---|---|---|---|---|---|---|---|
| **Baseline Logistic Regression** | 0.1455 | 0.9819 | 5.86% | 34.93% | 0.00% | 3,363 | t = 0.50 |
| **GNN Edge Risk Classifier alone** | 0.0031 | 0.4029 | 0.21% | 0.22% | 0.00% | 4,190 | t = 0.50 |
| **XGBoost Champion (`@champion`)** | **0.9239** | **0.9994** | **95.95%** | **73.27%** | **78.91%** | **3,059** | t = 0.99 |
| **Blended Scoring System ($\alpha=1.0$)\*** | **0.9239** | **0.9994** | **95.95%** | **73.27%** | **78.91%** | **3,059** | t = 0.99 |

*\*Note on Blend Equivalence: Blended system with $\alpha=1.0$ is identical to XGBoost by construction.*

![Model Benchmark Comparison](docs/figures/model_comparison.png)
![Precision-Recall Curves](docs/figures/pr_curves.png)

### Real-Time Serving Latency (Two Distinct Benchmarks)

Latency was measured over $\ge 2,000$ hold-out transactions on Python 3.13 (Windows host) with XGBoost restricted to single-core inference (`n_jobs=1`) on an idle system. Both benchmarks easily satisfy the PRD performance target of **p50 and p95 latency below 100 ms**. Results are saved to [`docs/results/latency_in_process.json`](docs/results/latency_in_process.json), [`docs/results/latency_http.json`](docs/results/latency_http.json), and [`docs/results/latency_all_trials.json`](docs/results/latency_all_trials.json).

| Benchmark Metric | Benchmark 1: In-Process (`ScoringEngine.score`) | Benchmark 2: HTTP Stack (`POST /score` TestClient) | Benchmark 3: Real Uvicorn (TCP Loopback) | PRD Performance Target |
|---|---|---|---|---|
| **Evaluated Requests** | 2,000 requests (100 warmup) | 2,000 requests (100 warmup) | 2,000 requests (100 warmup) | $\ge 2,000$ requests |
| **Mean Latency** | **13.38 ms** | **11.83 ms** | **9.63 ms** | < 100.0 ms |
| **Median (p50)** | **12.94 ms** | **11.33 ms** | **9.19 ms** | < 100.0 ms |
| **90th Percentile (p90)** | **16.27 ms** | **14.75 ms** | **11.83 ms** | < 100.0 ms |
| **95th Percentile (p95)** | **17.80 ms** | **15.80 ms** | **12.96 ms** | < 100.0 ms |
| **99th Tail (p99)** | **21.10 ms** | **18.79 ms** | **14.79 ms** | -- |

- **Benchmark 1 (In-Process Scoring Function):** Measures the pure algorithmic pipeline in memory: Feast online entity lookup from SQLite (4.2 ms), scikit-learn feature preprocessing (3.8 ms), XGBoost inference with `n_jobs=1` (3.9 ms), and GNN blend computation + background queue enqueue (0.1 ms). Total in-process p95 is **17.80 ms**, well below the PRD 100 ms target.
- **Benchmark 2 (HTTP POST /score via TestClient):** Measures the full microservice boundary: JSON client request serialization, Starlette HTTP routing, Pydantic type validation, `ScoringEngine.score()`, and asynchronous database write queue dispatch (p95 = **15.80 ms**).
- **Benchmark 3 (Real Uvicorn Server over TCP):** Measures real TCP loopback client-server communication using an external `httpx.Client` against a standalone Uvicorn process (p95 = **12.96 ms**), confirming zero synthetic TestClient distortion.

![Latency Tail Breakdown](docs/figures/latency_breakdown.png)

---

## 🔬 Core Engineering & Design Tradeoffs

### 1. The Metric Trap: Why Accuracy & ROC-AUC Fail in Fraud
In the PaySim financial dataset, fraud occurs in only 0.13% of transactions (1224:1 class imbalance). A trivial baseline predicting "Legitimate" for every transaction achieves **99.94% accuracy** while losing 100% of fraud capital. Similarly, Logistic Regression achieves a high ROC-AUC of **0.9819**, yet its PR-AUC is only **0.1455**, resulting in 140 false alarms for every genuine fraud detected. FraudShield optimizes strictly for **Precision-Recall AUC (PR-AUC)** and **Recall @ Precision $\ge 0.90$**, capping analyst fatigue.

### 2. Multi-Threading Contention: The `n_jobs=1` Decision
During Phase 4 load testing, benchmarking XGBoost with `n_jobs=4` inside FastAPI triggered severe OpenMP thread thrashing against Python's asynchronous event loop workers. While mean latency remained ~12 ms, tail latency (`p99`) spiked to **112.5 ms**. Restricting each booster prediction to `n_jobs=1` eliminated thread synchronization contention, dropping in-process `p99` latency down to **21.10 ms** (and HTTP `p99` to **18.79 ms**).

### 3. Graph Neural Networks vs Dynamic Cold-Start Turnover
While GNNs (PyG GraphSAGE) excelled at detecting money-laundering forwarding rings within the training window (validation PR-AUC 0.58), evaluating the GNN Edge Risk Classifier on future hold-out test transactions resulted in **PR-AUC 0.0031**. PaySim accounts have an extreme turnover rate: **only 0.05% of test origin accounts appeared in training steps $\le 323$**. FraudShield addresses this with a resilient convex blend formula ($S = \alpha S_{\text{xgb}} + (1 - \alpha) S_{\text{gnn}}$) and dynamically sets $\alpha = 1.0$ when graph signals lack historical coverage, guaranteeing zero degradation over tabular trees (empirically and mathematically identical to XGBoost).

### 4. Promotion Safeguards, Gate Set Integrity & Statistical Power
The automated retraining loop enforces a strict gate guardrail: candidate models are evaluated on a frozen, chronological gate set of 15,000 transactions (230 frauds) isolated from training. Promotion requires a minimum $+0.0010$ PR-AUC lift without degrading recall, and a 95% bootstrap confidence interval strictly above 0.
- **Statistical Power Limitation:** With only $N_{\text{fraud}}=230$ frauds in the gate set, standard error is $SE \approx \sqrt{\frac{0.96 \times 0.04}{230}} \approx 0.0129$. Small point estimate gains ($+0.0010$ to $+0.0040$) yield 95% bootstrap confidence intervals that cross zero (e.g. $[-0.0038, +0.0127]$). Thus, the gate set is statistically weak for sub-0.01 gains.
- **Dual Scenario Demonstration:**
  - **Scenario 1 (Rejection):** Candidate v8 achieves PR-AUC 0.9647 vs Champion v1 0.9607 ($\Delta = +0.0040$). Because the 95% bootstrap CI $[-0.0038, +0.0127]$ crosses zero (`ci_strictly_positive = False`), the gate marks it **`[REJECTED]`**, guarding against deploying noisy variations.
  - **Scenario 2 (Promotion):** Evaluated against a weakened champion (PR-AUC 0.3079), Candidate v8 achieves $\Delta = +0.6568$ with a 95% bootstrap CI of $[+0.6022, +0.7095]$ (`ci_strictly_positive = True`). The model is promoted to MLflow `@champion`, `/admin/reload-model` is called, and serving is cleanly reloaded with zero downtime. Serving is cleanly restored to Champion v1 afterward.

---

## 🔍 Verification Status & Host Environment

FraudShield maintains complete honesty regarding what was verified on the host machine versus static container definitions:

| Component / Workflow | Verification Status | Host Environment Details |
|---|---|---|
| **Full Python Pipeline (Phases 0-7)** | **VERIFIED LOCALLY** | 63 tests passing on Python 3.13.14 via pytest |
| **FastAPI REST API (`/score`, `/health`)** | **VERIFIED LOCALLY** | Verified via `TestClient` and in-process benchmarks (p50 12.8 ms) |
| **Feast Feature Store** | **VERIFIED LOCALLY** | Tested on SQLite online store (`feature_repo/data/online_store.db`) |
| **MLflow Tracking & Registry** | **VERIFIED LOCALLY** | Logged to local SQLite backend (`sqlite:///mlruns.db`) |
| **Streamlit Analyst UI** | **VERIFIED LOCALLY** | Schema parity and triage actions verified on SQLite fallback |
| **Evidently Drift Pipeline** | **VERIFIED LOCALLY** | Verified with Wasserstein distance & KS tests |
| **Phase 8 Container Infrastructure** | **WRITTEN, STATICALLY VALIDATED, NOT EXECUTED LOCALLY** | Validated Dockerfiles (pinned Python 3.13-slim) & compose syntax; Docker daemon not installed on Windows host |
| **Phase 8 Kubernetes Manifests** | **WRITTEN, STATICALLY VALIDATED, NOT EXECUTED LOCALLY** | Validated declarative syntax (k8s/*.yaml); Minikube / kubectl not installed on Windows host |
| **Kafka Live Broker Replay** | **STATICALLY VALIDATED** | Streaming consumer and DLQ logic unit-tested; broker runs in compose |

---

## 🛠️ Technology Stack

| Domain | Technology | Version | Purpose in Platform |
|---|---|---|---|
| **Language** | Python | 3.13.14 | Core development language across all services |
| **Tabular ML** | XGBoost | 3.4.1 | Primary fraud scoring booster with histogram binning |
| **Graph ML** | PyTorch Geometric | 2.7.0 | Graph construction, structural feature extraction, ring detection |
| **Feature Store** | Feast | 0.67.0 | Training/serving feature parity and entity state lookups |
| **Tracking & Registry** | MLflow | 3.17.2 | Run metrics tracking, artifact storage, model aliases (`@champion`) |
| **Serving Framework** | FastAPI | 0.128.0 | Asynchronous microservice exposing high-throughput REST API |
| **Monitoring** | Evidently | 0.7.20 | Distribution drift detection via Wasserstein metric |
| **Relational Database** | PostgreSQL / SQLite | 3.45 | Transactions, scores, cases, and feedback labels persistence |
| **Analyst UI** | Streamlit | 1.55.0 | Interactive alert triage, network ring viewer, labeling interface |
| **Containerization** | Docker Compose / K8s | v2 / 1.30+ | Multi-service orchestration specifications |

---

## 🗺️ Project Delivery Roadmap

- [x] **Phase 0: Environment & Baseline Infrastructure** — Pytest suite, ruff linting, Docker Compose definition, CI workflow.
- [x] **Phase 1: Ingestion & Temporal Splitting** — Raw PaySim download/generation, strict step $\le 323$ isolation, zero leakage.
- [x] **Phase 2: Feast Feature Store** — Entity views, historical features, SQLite online store, parity test.
- [x] **Phase 3: Tabular Model & MLflow Tracking** — Logistic regression vs XGBoost tuning, PR-AUC 0.9238, registered model v1.
- [x] **Phase 4: Streaming Scoring Service** — FastAPI `POST /score`, sub-20ms p95 HTTP latency, async persistence, dead-letter queue.
- [x] **Phase 5: Graph Modeling & Blending** — PyG graph construction, mule ring detection, convex blend scoring, cold-start fallback.
- [x] **Phase 6: Analyst Dashboard & Feedback** — Streamlit case management, transaction inspection, ground truth label feedback.
- [x] **Phase 7: Drift Monitoring & Automated Retraining** — Evidently drift pipeline, frozen evaluation gate set, MLflow promotion, zero-downtime hot reload.
- [x] **Phase 8: Production Packaging (Compose & K8s)** — Production multi-stage Dockerfiles (Python 3.13), Kubernetes manifests, Redis removal cleanup.
- [x] **Phase 9: Polish, Reproducibility & Deliverables** — `bootstrap.py` one-click setup, comprehensive architecture docs, automated results generation, executive deck outline, demo script.
