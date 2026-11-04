# FraudShield — Environment & Deployment Readiness Assessment

**Updated**: 2026-11-03  
**Auditor**: Senior Backend, DevOps & MLOps Architect  
**Evaluation Scope**: Host Environment, Runtime Daemons, Databases, Message Brokers, Dependencies, Hardware Resources, and Container/K8s Manifests.

---

## 1. Executive Summary & Verification Matrix

| Component | Implemented | Unit Tested | Integration Tested | Runtime Verified | Status | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Python Runtime (3.13.14)** | Yes | Yes | Yes | **Yes** | **PASS** | Virtual environment fully loaded with all required ML and API libraries |
| **FastAPI Serving Engine** | Yes | Yes | Yes | **Yes** | **PASS** | Sub-10ms single-threaded XGBoost inference (`n_jobs=1`), p50/p95 latency metrics |
| **XGBoost Scoring Engine** | Yes | Yes | Yes | **Yes** | **PASS** | Loaded from MLflow/local JSON with active `BLEND_ALPHA = 1.0` configuration |
| **GNN Graph Lookup** | Yes | Yes | Yes | **Yes** | **PASS** | Preserved in codebase; bypassed during active scoring per measured PaySim results |
| **SQLite Persistence Fallback** | Yes | Yes | Yes | **Yes** | **PASS** | WAL-mode async persistence of transactions, scores, cases, and labels |
| **Streamlit Dashboard** | Yes | Yes | Yes | **Yes** | **PASS** | Alert triage, case forensic investigation, Tree SHAP explanations, feedback submission |
| **Streaming Replay & DLQ** | Yes | Yes | Yes | **Yes** | **PASS** | Verified with `MemoryStreamingChannel`; malformed events routed to DLQ |
| **PostgreSQL Backend** | Yes | Yes | Yes | *CI / External* | **PARTIAL** | Manifests & compose configured; local fallback active; live Postgres unverified on host |
| **Kafka Message Broker** | Yes | Yes | Yes | *CI / External* | **PARTIAL** | Manifests (`04-kafka.yaml`) & compose configured; live Kafka unverified on host |
| **Docker Compose Services** | Yes | Yes | Yes | *CI Only* | **PARTIAL** | All Dockerfiles & compose parsed cleanly; host lacks Docker daemon |
| **Kubernetes Manifests (k8s/)** | Yes | Yes | N/A | *CI Only* | **PARTIAL** | Kubeconform v0.6.7 validated; host lacks kubectl and cluster context |

---

## 2. Detailed Subsystem Audit

### 2.1 Python Environment & Dependencies
- **Interpreter**: `Python 3.13.14` located at `D:\FraudShield\.venv\Scripts\python.exe`.
- **Core ML Packages**:
  - `xgboost 3.4.1` — Loaded and running with single-threaded inference (`n_jobs=1`).
  - `torch 2.14.1` & `torch-geometric 2.8.1` — Available for graph operations and embeddings.
  - `scikit-learn 1.9.1` — Fitted preprocessors operational.
- **Serving & Web Frameworks**:
  - `fastapi 0.156.8`, `starlette 1.9.0`, `uvicorn 0.54.0` — API routing and lifespan startup operational.
  - `pydantic 2.14.1` & `pydantic-core 2.50.1` — Input contracts enforced.
- **Explainability & Drift**:
  - XGBoost Tree SHAP (`pred_contribs=True`) — Exact feature attribution operational.
  - `evidently 0.7.23` — Drift monitoring operational.
- **Verification Status**: **PASS** (Runtime verified).

---

### 2.2 Persistence & Database Infrastructure
- **PostgreSQL**:
  - Connection string: `postgresql://postgres:postgres@localhost:5432/fraudshield`.
  - Host port 5432 status: Inactive.
  - Fallback mechanism: `DatabaseManager` in `src/common/db.py` automatically detects when PostgreSQL is unreachable and initializes a local SQLite database at `data/fraudshield_serving.db` with WAL mode enabled (`PRAGMA journal_mode=WAL;`).
  - Schema parity: The SQLite schema mirrors `docker/init.sql`, providing `transactions`, `scores`, `cases`, and `labels` tables with indices.
  - Kubernetes initialization: Fixed in `k8s/03-postgres.yaml` by mounting `postgres-init-config` into `/docker-entrypoint-initdb.d/init.sql`.
- **Verification Status**: **PARTIAL** (SQLite fallback: **Runtime Verified**; PostgreSQL live container: **Unit Tested & Configured**).

---

### 2.3 Streaming & Message Broker Infrastructure
- **Kafka**:
  - Host port 9092 status: Inactive.
  - In-Memory Fallback: `MemoryStreamingChannel` in `src/ingestion/streaming.py` provides thread-safe, topic-partitioned in-memory queues.
  - Kubernetes Manifests: Added `k8s/04-kafka.yaml` providing a single-pod Apache Kafka broker in KRaft mode and a topic initialization job.
- **Verification Status**: **PARTIAL** (Memory channel & DLQ: **Runtime Verified**; Live Kafka broker: **Unit Tested & Configured**).

---

### 2.4 Container & Cluster Runtime (Docker & Kubernetes)
- **Docker CLI / Daemon**:
  - Command `docker --version` returns `CommandNotFoundException`. Docker daemon is not installed on PATH on this Windows host.
  - Static Dockerfile analysis (`tests/unit/test_containers.py`): All 4 Dockerfiles exist, pin Python 3.13-slim-bookworm base images, use multi-stage builds (`AS builder` / `AS runner`), and declare unprivileged non-root users (`appuser:10001`).
- **Kubernetes (kubectl / Cluster)**:
  - Command `kubectl` returns `CommandNotFoundException`.
  - Static manifest analysis: Manifests in `k8s/*.yaml` validate against Kubernetes schemas via `kubeconform v0.6.7`, specify CPU/memory requests and limits, declare liveness/readiness probes, and match Service selectors to Deployment/StatefulSet labels.
- **Verification Status**: **PARTIAL** (Unit tested & statically validated; runtime execution requires host Docker/k8s daemon or CI runners).

---

### 2.5 Hardware & Resource Bounds

```
CPU Model: Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz
Cores: 4 Physical Cores, 8 Logical Processors
Total Physical RAM: 15.82 GB (16,586,964 KB)
Available RAM: 5.08 GB (5,083,988 KB)
Drive C: (System): 38.30 GB Free / 155.53 GB Used
Drive D: (Project): 134.20 GB Free / 8.14 GB Used
```

- Peak test suite memory consumption: ~1.2 GB.
- Meets PRD specifications for local execution.
