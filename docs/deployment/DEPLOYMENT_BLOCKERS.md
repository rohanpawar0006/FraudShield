# FraudShield — Deployment Blockers & Resolution Audit

**Updated**: 2026-11-03  
**Auditor**: Senior Backend, DevOps & MLOps Architect  
**Scope**: Docker Compose, Kubernetes Manifests, Environment Configuration, Secrets, and Network Topologies.

---

## 1. Executive Status of Identified Blockers

| ID | Component | Severity | Description | Resolution Status | Verification State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BLK-01** | Docker / Compose | High | Incorrect module paths (`src.stream.*` instead of `src.ingestion.*`) and missing `--loop` flag | **RESOLVED** in codebase & compose | Unit tested & Integration tested |
| **BLK-02** | Docker / K8s Cron | High | CLI argument mismatch: Dockerfile & CronJob passed `--check-drift --check-labels` which failed `argparse` | **RESOLVED** in codebase | Unit tested & Integration tested |
| **BLK-03** | Kubernetes Manifests | **CRITICAL** | `k8s/03-postgres.yaml` failed to mount `init.sql`; PostgreSQL booted with empty schema, crashing serving & dashboard | **RESOLVED** via `postgres-init-config` ConfigMap & `/docker-entrypoint-initdb.d` mount | Unit tested (manifest schema & mounts) |
| **BLK-04** | Security / Config | Medium | `ADMIN_API_KEY` omitted from `k8s/02-secrets.yaml`, `k8s/05-serving.yaml`, and `docker-compose.yml` | **RESOLVED** across secrets, manifests, compose, and `.env.example` | Unit tested (secret injection & fail-closed auth) |
| **BLK-05** | Kubernetes Manifests | Medium | No Kafka or KRaft manifests in `k8s/`; assumed external broker without deployment path | **RESOLVED** via `k8s/04-kafka.yaml` (KRaft broker + topic init Job) | Unit tested (labels, limits, and probes) |
| **BLK-06** | Host Environment | Info | Windows development machine lacks Docker daemon and `kubectl` on PATH | **RESOLVED** for Docker | Docker Desktop 4.98.0 + WSL2 operational |
| **BLK-07** | Docker / Streaming | High | In-container consumer fails because `Dockerfile.stream` lacks ML dependencies (`xgboost`, `feast`, `models/`, `feature_repo/`) while `consumer.py` performs in-process scoring | **IN PROGRESS** | Option A: consumer to use `fraudshield-serving` image |

---

## 2. Technical Post-Mortem & Resolution Details

### BLK-01: Streaming Worker Module Name Mismatch (RESOLVED)
- **Root Cause**: `docker/docker-compose.yml` and `docker/Dockerfile.stream` specified `src.stream.producer` and `src.stream.consumer`. The Python package in the repository is `src.ingestion`.
- **Resolution**:
  1. Updated `docker/docker-compose.yml` to call `src.ingestion.producer` and `src.ingestion.consumer`.
  2. Updated `docker/Dockerfile.stream` default command to `src.ingestion.consumer`.
  3. Added `--loop` flag support to `src/ingestion/producer.py` to allow continuous streaming replay loops.
- **Verification**: Tested live via `MemoryStreamingChannel` and Python invocation. Unit tests passing.

---

### BLK-02: Retrain Trigger CLI Argument Mismatch (RESOLVED)
- **Root Cause**: `docker/Dockerfile.drift` and `k8s/07-drift-cronjob.yaml` passed `--check-drift --check-labels`, which failed `retrain_trigger.py`'s `argparse`.
- **Resolution**: Added `--check-drift` and `--check-labels` flags to `src/monitoring/retrain_trigger.py`.
- **Verification**: Tested live CLI execution (`python -m src.monitoring.retrain_trigger --check-drift --check-labels`).

---

### BLK-03: PostgreSQL Kubernetes Initialization (RESOLVED)
- **Root Cause**:
  In `k8s/03-postgres.yaml`, PostgreSQL was declared as a StatefulSet without mounting `init.sql`. In Kubernetes, PostgreSQL started with an empty schema, causing `UndefinedTable` exceptions on `transactions`, `scores`, `cases`, and `labels`.
- **Resolution**:
  1. Added `postgres-init-config` ConfigMap in `k8s/03-postgres.yaml` embedding the exact DDL from `docker/init.sql`.
  2. Mounted the ConfigMap into the PostgreSQL container under `/docker-entrypoint-initdb.d` as a read-only volume.
  3. **Data Integrity & Restart Guarantee**:
     - On first boot (empty PVC), the official `postgres:15-alpine` entrypoint executes `/docker-entrypoint-initdb.d/init.sql`.
     - On container restarts or pod rescheduling on an existing PVC, PostgreSQL detects the existing `PGDATA` cluster and **skips** `/docker-entrypoint-initdb.d/`, ensuring persistent user data is never overwritten.
     - DDL statements use `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` for complete idempotency.
- **Verification**: Regression test `test_postgres_kubernetes_initialization_configured` in `tests/unit/test_containers.py`.

---

### BLK-04: Admin API Key Secret Injection (RESOLVED)
- **Root Cause**:
  When `FRAUDSHIELD_ENV=production`, `src/serving/app.py` enforces fail-closed behavior: if `ADMIN_API_KEY` is unset, administrative operations (`/admin/reload-model`, `/admin/retrain`) return HTTP 500 (`Server security misconfiguration`). Neither `k8s/02-secrets.yaml` nor `docker-compose.yml` injected this key.
- **Resolution**:
  1. Added `ADMIN_API_KEY` placeholder to `k8s/02-secrets.yaml`.
  2. Injected `ADMIN_API_KEY` from `fraudshield-secrets` into `k8s/05-serving.yaml`.
  3. Added `ADMIN_API_KEY=${ADMIN_API_KEY:-}` to `serving` service in `docker/docker-compose.yml`.
  4. Documented `ADMIN_API_KEY` generation in `.env.example`.
- **Verification**: Regression test `test_admin_api_key_secret_injection_configured` in `tests/unit/test_containers.py`, and endpoint tests `test_admin_endpoint_security_unauthorized` / `test_admin_endpoint_security_authorized` in `tests/unit/test_serving.py`.

---

### BLK-05: Kafka Kubernetes Deployment (RESOLVED)
- **Root Cause**: The repository lacked Kubernetes manifests for Apache Kafka.
- **Resolution**:
  1. Created `k8s/04-kafka.yaml` declaring a single-node Apache Kafka broker in KRaft mode (`confluentinc/cp-kafka:7.5.0`), eliminating ZooKeeper complexity.
  2. Added PersistentVolumeClaim `kafka-pvc` (2Gi) for message retention.
  3. Added Service `kafka` (ClusterIP, ports 9092 client / 29092 internal).
  4. Added `kafka-init-topics` Job to automatically initialize `transactions`, `scores`, and `transactions.dlq`.
  5. Configured resource requests (`150m`/`384Mi`) and limits (`300m`/`768Mi`).
  6. **Production Strategy**: Documented that while `k8s/04-kafka.yaml` enables self-contained dev/demo deployments, production enterprise clusters should configure `KAFKA_BOOTSTRAP_SERVERS` in `k8s/01-configmap.yaml` to point to a managed Kafka service (AWS MSK or Confluent Cloud).
- **Verification**: Manifest parsing, selector matching, and resource limit assertions in `tests/unit/test_containers.py`.

---

### BLK-06: Host Runtime Limitations (RESOLVED FOR DOCKER)
- **Status**: Docker Desktop v4.98.0 (WSL2 engine v2, 4 GB RAM allocated) installed and operational.
- **Verification**: `docker version`, `docker compose version` (v5.6.1), PostgreSQL 15, and Kafka KRaft verified live.

---

### BLK-07: Streaming Consumer Architectural Dependency Gap (IN PROGRESS)
- **Root Cause**:
  `src/ingestion/consumer.py` imports `src.serving.scoring.scoring_engine` and performs model scoring and feature store queries in-process. However, `docker/Dockerfile.stream` was designed as a lean ingestion base containing only `requirements-stream.txt` (`kafka-python-ng`, `pandas`, `pyarrow`, `psycopg2-binary`) and copies only `src/`. It lacks `xgboost`, `scikit-learn`, `joblib`, `feast`, and omitted `models/` and `feature_repo/`. Attempting to start the consumer container with `Dockerfile.stream` immediately fails with `ModuleNotFoundError: No module named 'xgboost'`.
  Additionally, Kubernetes manifests in `k8s/` only define ConfigMap consumer groups (`fraudshield-scoring-group`), with no Deployment manifest for the streaming consumer.
- **Resolution Strategy**:
  - Adopt **Option A**: Run the streaming consumer on `fraudshield-serving:latest`, which already contains `xgboost`, `scikit-learn`, `feast`, `models/`, and `feature_repo/`.
  - Add `kafka-python-ng==2.2.3` to `docker/requirements-serving.txt`.
  - Update `consumer` definition in `docker/docker-compose.yml` to use `image: fraudshield-serving:latest` and mount required volumes.
- **Verification State**: Staged for execution in Phase 4 Stage C3.
