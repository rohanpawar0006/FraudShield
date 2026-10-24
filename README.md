# FraudShield — Real-Time Financial Fraud Detection Platform

FraudShield is an end-to-end real-time fraud detection and risk-scoring platform designed for high-throughput financial transactions. It couples tabular gradient boosting (XGBoost) with graph neural network ring detection (PyTorch Geometric) over a low-latency streaming pipeline (Kafka, FastAPI, Redis/Feast, PostgreSQL).

---

## 🏛️ Architecture Overview

```text
Producer (replay) ──► Kafka topic: transactions
                          │
                          ▼
                 FastAPI scoring consumer ──► Feast online store (Redis) ──► Features
                          │                 └► XGBoost model (MLflow registry)
                          │                 └► GNN score (precomputed risk embeddings)
                          ▼
                 Blend → score ──► Kafka topic: scores ──► Postgres (scores, cases, labels)
                                                               │
                                  Streamlit analyst dashboard ◄┘
```

- **Streaming Layer**: Kafka replay producer and FastAPI consumer.
- **Feature Store**: Feast (Parquet offline, Redis online) ensuring zero train/serve skew.
- **Machine Learning**: XGBoost (tabular) + PyG GraphSAGE/GAT (fraud rings) blended score.
- **Monitoring & Feedback**: Evidently AI drift detection triggering retraining; analyst triage in Streamlit updating Postgres feedback labels.
- **Deployment**: Docker Compose & Kubernetes (Minikube).

---

## 🚀 Quickstart & Setup (Phase 0)

### 1. Prerequisites
- Python 3.11+ / Python 3.13
- Git
- Docker & Docker Compose (optional for local mock testing, required for full infra)

### 2. Virtual Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Running Linting & Tests
```bash
# Lint with ruff
ruff check .

# Execute test suite
pytest
```

### 4. Running Infrastructure (Docker Compose)
To launch Kafka, Redis, PostgreSQL, and MLflow:
```bash
docker compose -f docker/docker-compose.yml up -d
```

Service endpoints:
- **Kafka**: `localhost:9092`
- **Redis**: `localhost:6379`
- **Postgres**: `localhost:5432` (`user: postgres`, `db: fraudshield`)
- **MLflow**: `http://localhost:5000`

---

## 📁 Repository Structure
```text
FraudShield/
├── data/                 # raw/ (gitignored) and processed/ parquet
├── notebooks/            # EDA, feature engineering, and graph experiments
├── feature_repo/         # Feast feature store definitions
├── src/
│   ├── ingestion/        # Transaction producer and schemas
│   ├── features/         # Feature logic shared offline and online
│   ├── training/         # XGBoost, GNN training, and blend logic
│   ├── graph/            # Graph construction and PyG models
│   ├── serving/          # FastAPI app, Kafka consumer, and scoring engine
│   ├── monitoring/       # Evidently drift reports and retrain triggers
│   └── common/           # Config, logging, and database connectors
├── dashboard/            # Streamlit analyst operations UI
├── k8s/                  # Kubernetes manifests (Minikube)
├── docker/               # Dockerfiles, init scripts, docker-compose.yml
├── tests/                # Unit and integration test suites
├── docs/                 # Documentation, API contracts, metrics
├── pyproject.toml        # Ruff and Pytest configuration
├── requirements.txt      # Dependency specification
└── README.md             # Project documentation
```

---

## 📜 Development Rules
1. **Time-Based Splits Only**: No random train/test splits; prevent future leakage.
2. **PR-AUC as Headline Metric**: Accuracy is never used alone on imbalanced datasets (<1% fraud).
3. **Train/Serve Parity**: Features are defined once and verified with parity unit tests.
4. **Reproducibility**: All models and hyperparameters logged to MLflow.
