# FraudShield Integration & Runtime Verification Results

> **Verification Status**: Postgres and Kafka broker/topics = PASS (containerized, evidence below); serving container = PARTIAL; consumer/DLQ/recovery/dashboard/Kubernetes = NOT VERIFIED; Docker removed from the dev machine afterwards.

**Date**: 2026-11-04  
**Environment**: Windows 11 Pro / WSL2 Engine / Docker Desktop v4.98.0 / Docker Compose v5.6.1  
**Target Database**: PostgreSQL 15.19 Alpine (`fraudshield-postgres`)  
**Application Environment**: Python 3.13.14 (`.venv`), `USE_SQLITE_FALLBACK=false`  

---

## 1. Pre-Flight Environment Inspection

- **Free Physical RAM**: **3.92 GB** (4,018 MB) out of 15.82 GB.
- **Environment File**: `.env` created from `.env.example` with local development credentials (`USE_SQLITE_FALLBACK=false`).
- **Git Protection**: Confirmed `.env` matches `.gitignore` (untracked, excluded from repository).

---

## 2. Step 3: PostgreSQL Runtime Verification

### Item 1 — Container Startup, Healthcheck, and Schema Initialization

**Command Executed:**
```bash
docker compose -f docker/docker-compose.yml up -d postgres
docker compose -f docker/docker-compose.yml ps
docker logs fraudshield-postgres
```

**Terminal Evidence:**
```text
NAME                   IMAGE                COMMAND                  SERVICE    CREATED          STATUS                   PORTS
fraudshield-postgres   postgres:15-alpine   "docker-entrypoint.s…"   postgres   10 seconds ago   Up 9 seconds (healthy)   0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp
```

**Container Initialization Logs:**
```text
/usr/local/bin/docker-entrypoint.sh: running /docker-entrypoint-initdb.d/init.sql
CREATE TABLE
CREATE TABLE
CREATE INDEX
CREATE INDEX
CREATE INDEX
CREATE TABLE
CREATE INDEX
CREATE TABLE
CREATE INDEX
...
PostgreSQL init process complete; ready for start up.
2026-11-04 04:51:33.639 UTC [1] LOG:  starting PostgreSQL 15.19 on x86_64-pc-linux-musl, compiled by gcc (Alpine 15.2.0) 15.2.0, 64-bit
2026-11-04 04:51:33.639 UTC [1] LOG:  listening on IPv4 address "0.0.0.0", port 5432
2026-11-04 04:51:33.655 UTC [1] LOG:  database system is ready to accept connections
```

- **Verdict**: **PASS**

---

### Item 2 — Schema Validation & Codebase Table Comparison

**Command Executed:**
```bash
docker exec -i fraudshield-postgres psql -U postgres -d fraudshield -c "\dt"
```

**Terminal Evidence:**
```text
            List of relations
 Schema |     Name     | Type  |  Owner   
--------+--------------+-------+----------
 public | cases        | table | postgres
 public | labels       | table | postgres
 public | scores       | table | postgres
 public | transactions | table | postgres
(4 rows)
```

**Codebase Table Usage Audit (`src/` & `dashboard/`):**
- `src/common/db.py`: Reads and writes `transactions`, `scores`.
- `src/feedback/service.py`: Reads and writes `cases`, `labels`, `scores`, `transactions`.
- `src/monitoring/drift.py` & `retrain_trigger.py`: Reads `scores`, `transactions`, `labels`.
- `dashboard/streamlit_app.py`: Delegates queries to `src/feedback/service.py`.

**Reconciliation Result:**
- **Missing tables**: 0
- **Extra tables**: 0
- **Primary & Foreign Keys**: All tables declare primary keys (`transactions_pkey`, `scores_pkey`, `cases_pkey`, `labels_pkey`) and foreign key constraints with `ON DELETE CASCADE` referencing `transactions(txn_id)`.
- **B-Tree Indexes**: `idx_scores_blended_score`, `idx_scores_scored_at`, `idx_scores_txn_id`, `idx_cases_status`, `idx_labels_txn_id` are verified present.

- **Verdict**: **PASS**

---

### Item 3 — Application-Layer CRUD Verification

Executed through the actual Python application database client (`src.common.db.DatabaseManager` and `src.feedback.service`):

**Command Executed:**
```bash
python scripts/test_postgres_crud.py
```

**Terminal Evidence:**
```text
{"timestamp": "2026-11-04T04:54:30.458404+00:00", "level": "INFO", "logger": "src.common.db", "message": "Connected to PostgreSQL backend successfully"}
=== Running PostgreSQL Application CRUD Verification ===
PostgreSQL Host: localhost:5432
PostgreSQL DB:   fraudshield
Backend Active:  postgresql
Reachable:       True

[Step 2] Inserting transaction and score via db_manager.save_transaction_and_score()...
[Step 3] Reading back transaction and score...
 -> Found transaction: txn_id=test_crud_txn_001, account_id=C1234567890, amount=5432.1, type=TRANSFER
 -> Found score: txn_id=test_crud_txn_001, xgb_score=0.952, blended_score=0.952

[Step 4] Submitting analyst feedback via submit_feedback('Under Review')...
{"timestamp": "2026-11-04T04:54:31.762961+00:00", "level": "INFO", "logger": "src.feedback.service", "message": "Feedback recorded for txn test_crud_txn_001: 'Under Review' by analyst_alice (prev: 'New')"}
 -> Result: {'transaction_id': 'test_crud_txn_001', 'disposition': 'Under Review', 'previous_disposition': 'New', 'label': -1, 'analyst': 'analyst_alice', 'updated_at': '2026-11-04T04:54:31.727015+00:00'}

[Step 5] Reading back via get_case() and list_alerts()...
 -> Case status: Under Review, assigned_to: analyst_alice
 -> Audit trail count: 1
 -> Found alert in list_alerts: txn_id=test_crud_txn_001, status=Under Review

[Step 6] Updating analyst feedback via submit_feedback('Confirmed Fraud')...
{"timestamp": "2026-11-04T04:54:31.879357+00:00", "level": "INFO", "logger": "src.feedback.service", "message": "Feedback recorded for txn test_crud_txn_001: 'Confirmed Fraud' by analyst_alice (prev: 'Under Review')"}
 -> Result: {'transaction_id': 'test_crud_txn_001', 'disposition': 'Confirmed Fraud', 'previous_disposition': 'Under Review', 'label': 1, 'analyst': 'analyst_alice', 'updated_at': '2026-11-04T04:54:31.828031+00:00'}
 -> Updated case status: Confirmed Fraud
 -> Updated audit trail count: 2

[Step 7] Deleting test record (verifying ON DELETE CASCADE)...
 -> Table transactions remaining rows for test_crud_txn_001: 0
 -> Table scores remaining rows for test_crud_txn_001: 0
 -> Table cases remaining rows for test_crud_txn_001: 0
 -> Table labels remaining rows for test_crud_txn_001: 0

=== Application CRUD Verification PASSED (100% Success) ===
```

- **Verdict**: **PASS**

---

### Item 4 — Volume Data Persistence Across Restarts and Down/Up Cycles

Tested with synthetic marker record (`marker_persist_txn_999`).

**Step 4a: Insert Marker**
```text
[Persistence Test] Inserting marker transaction 'marker_persist_txn_999'...
 -> VERIFIED: Transaction marker_persist_txn_999 (amount: 9999.99, type: TRANSFER) and Score marker_persist_txn_999 (blended: 0.999) are PRESENT.
```

**Step 4b: Container Restart**
```bash
docker compose -f docker/docker-compose.yml restart postgres
python scripts/test_postgres_persistence.py verify
```
**Terminal Evidence:**
```text
Container fraudshield-postgres Restarting 
Container fraudshield-postgres Started 
 -> VERIFIED: Transaction marker_persist_txn_999 (amount: 9999.99, type: TRANSFER) and Score marker_persist_txn_999 (blended: 0.999) are PRESENT.
```

**Step 4c: Compose Down & Recreate (No Volume Pruning)**
```bash
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml up -d postgres
python scripts/test_postgres_persistence.py verify
```
**Terminal Evidence:**
```text
Container fraudshield-postgres Stopping 
Container fraudshield-postgres Stopped 
Container fraudshield-postgres Removing 
Container fraudshield-postgres Removed 
Network docker_fraudshield-net Removing 
Network docker_fraudshield-net Removed 

Network docker_fraudshield-net Creating 
Network docker_fraudshield-net Created 
Container fraudshield-postgres Creating 
Container fraudshield-postgres Created 
Container fraudshield-postgres Starting 
Container fraudshield-postgres Started 

 -> VERIFIED: Transaction marker_persist_txn_999 (amount: 9999.99, type: TRANSFER) and Score marker_persist_txn_999 (blended: 0.999) are PRESENT.
```

**Step 4d: Marker Cleanup**
```bash
python scripts/test_postgres_persistence.py cleanup
```
**Terminal Evidence:**
```text
 -> CLEANUP COMPLETE: Remaining rows for marker_persist_txn_999: 0
```

- **Verdict**: **PASS**

---

### Item 5 — Resource Usage & Performance

**Command Executed:**
```bash
docker stats --no-stream fraudshield-postgres
```

**Terminal Evidence:**
```text
CONTAINER ID   NAME                   CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O     PIDS
7e9ef477b926   fraudshield-postgres   0.03%     19.54MiB / 512MiB   3.82%     6.16kB / 5.18kB   0B / 57.3kB   6
```

- **Verdict**: **PASS**

---

## 3. Summary Scorecard

| Step / Verification Item | Target | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Pre-flight Checks** | Free RAM & `.env` configuration | 3.92 GB Free; `.env` created and gitignored | **PASS** |
| **Item 1: Container Startup** | `postgres:15-alpine` healthy & `init.sql` executed | Healthy in 9s; DDL created tables and indexes | **PASS** |
| **Item 2: Schema Audit** | Tables match `src/` and `dashboard/` | 4/4 tables matched; 0 missing, 0 extra | **PASS** |
| **Item 3: App Layer CRUD** | Persistence via `db_manager` and `service.py` | Insert, read, triage update, audit log & cascade delete verified | **PASS** |
| **Item 4: Persistence** | Data survival across restart and down/up | Marker record verified intact through both cycles | **PASS** |
| **Item 5: Resource Usage** | Memory budget limit 512 MB | Active usage 19.54 MiB (3.82% of limit, 0.03% CPU) | **PASS** |



---

## 3. Step 4: Kafka Runtime Verification — Stage A (Broker & Topic Initialization)

### Item 1 — Broker Launch & KRaft Health Check
- **Free Physical RAM before launch**: **1.40 GB** (1,437 MB).
- **Command**:
  ```bash
  docker compose -f docker/docker-compose.yml up -d kafka
  docker compose -f docker/docker-compose.yml ps
  ```

**Terminal Evidence:**
```text
NAME                   IMAGE                         COMMAND                  SERVICE    CREATED          STATUS                    PORTS
fraudshield-kafka      confluentinc/cp-kafka:7.5.0   "/etc/confluent/dock…"   kafka      26 seconds ago   Up 25 seconds (healthy)   0.0.0.0:9092->9092/tcp, 0.0.0.0:9094->9094/tcp
fraudshield-postgres   postgres:15-alpine            "docker-entrypoint.s…"   postgres   4 minutes ago    Up 4 minutes (healthy)    0.0.0.0:5432->5432/tcp
```

**KRaft Logs Evidence:**
```text
===> Using provided cluster id MkU3OEVBNTcwNTJENDM2Qk ...
[2026-11-04 04:59:45,411] INFO Awaiting socket connections on 0.0.0.0:29093. (kafka.network.DataPlaneAcceptor)
[2026-11-04 04:59:45,965] INFO Awaiting socket connections on 0.0.0.0:9092. (kafka.network.DataPlaneAcceptor)
[2026-11-04 04:59:45,968] INFO Awaiting socket connections on 0.0.0.0:29092. (kafka.network.DataPlaneAcceptor)
[2026-11-04 04:59:45,977] INFO [KafkaRaftServer nodeId=1] Kafka Server started (kafka.server.KafkaRaftServer)
```
- **Verdict**: **PASS**

---

### Item 2 — Topic Creation via `kafka-init`

**Command**:
```bash
docker compose -f docker/docker-compose.yml up kafka-init
docker exec -i fraudshield-kafka kafka-topics --bootstrap-server localhost:9092 --describe --topic transactions,scores,transactions.dlq
```

**Terminal Evidence (Execution & Exit 0):**
```text
fraudshield-kafka-init  | Creating Kafka topics...
fraudshield-kafka-init  | Created topic transactions.
fraudshield-kafka-init  | Created topic scores.
fraudshield-kafka-init  | Created topic transactions.dlq.
fraudshield-kafka-init  | Kafka topics successfully initialized:
fraudshield-kafka-init  | scores
fraudshield-kafka-init  | transactions
fraudshield-kafka-init  | transactions.dlq
fraudshield-kafka-init exited with code 0
```

**Partition & Replication Description:**
```text
Topic: transactions	TopicId: XERjh0pQQOyUSijRtaBSLg	PartitionCount: 3	ReplicationFactor: 1	Configs: 
	Topic: transactions	Partition: 0	Leader: 1	Replicas: 1	Isr: 1
	Topic: transactions	Partition: 1	Leader: 1	Replicas: 1	Isr: 1
	Topic: transactions	Partition: 2	Leader: 1	Replicas: 1	Isr: 1
Topic: transactions.dlq	TopicId: 3aZTp0vWSGO0tQC5YW7khg	PartitionCount: 1	ReplicationFactor: 1	Configs: 
	Topic: transactions.dlq	Partition: 0	Leader: 1	Replicas: 1	Isr: 1
Topic: scores	TopicId: HKEvRs_6RNKYaq3T4l9aZw	PartitionCount: 3	ReplicationFactor: 1	Configs: 
	Topic: scores	Partition: 0	Leader: 1	Replicas: 1	Isr: 1
	Topic: scores	Partition: 1	Leader: 1	Replicas: 1	Isr: 1
	Topic: scores	Partition: 2	Leader: 1	Replicas: 1	Isr: 1
```
- **Verdict**: **PASS**

---

### Item 3 — Memory Consumption & Health (`docker stats`)

**Command**:
```bash
docker stats --no-stream fraudshield-kafka fraudshield-postgres
```

**Terminal Evidence:**
```text
CONTAINER ID   NAME                   CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O     PIDS
a08851ffdb26   fraudshield-kafka      2.75%     298.9MiB / 768MiB   38.92%    7.03kB / 11.7kB   0B / 1.02MB   96
7e9ef477b926   fraudshield-postgres   0.03%     20.77MiB / 512MiB   4.06%     6.46kB / 5.18kB   0B / 582kB    6
```
- Memory consumed: **298.9 MiB** (38.92% of 768 MiB budget cap; >469 MiB safety headroom).
- OOM status: None.
- Host Free RAM: **1.64 GB** (1,682 MB).
- **Verdict**: **PASS**



---

## 4. Step 4: Kafka Runtime Verification — Stage B (Raw Produce / Consume Sanity)

### Item 1 & 2 — In-Container Scratch Topic Produce & Consume
- **Pre-check Host Free Physical RAM**: **1.35 GB** (1,380 MB).
- **Topic Created**: `scratch.sanity` (1 partition, replication factor 1).
- **Producer Command**:
  ```bash
  "SANITY_MSG_001`nSANITY_MSG_002`nSANITY_MSG_003" | docker exec -i fraudshield-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic scratch.sanity
  ```
- **Consumer Command**:
  ```bash
  docker exec -i fraudshield-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic scratch.sanity --from-beginning --max-messages 3 --timeout-ms 15000
  ```

**Terminal Evidence:**
```text
SANITY_MSG_001
SANITY_MSG_002
SANITY_MSG_003
Processed a total of 3 messages
```
- **Verdict**: **PASS**

---

### Item 3 — Host-Side Connection & Listener Verification
- **Host Bootstrap Address**: `localhost:9092` (matching `settings.KAFKA_BOOTSTRAP_SERVERS` and `.env`).
- **Container Listener Mappings**:
  - `9092`: `PLAINTEXT_HOST` advertised as `localhost:9092` for host clients.
  - `9094`: JMX management port (`KAFKA_JMX_PORT`), not a client broker listener.
  - `29092`: Internal Docker bridge listener (`PLAINTEXT`) advertised as `kafka:29092`.
- **Host Client Test Script**: Executed host-side produce and consume using `kafka-python` against `localhost:9092`.

**Terminal Evidence:**
```text
Testing HOST client connection to bootstrap: localhost:9092
Produce SUCCESS to partition 0 offset 3

Consumed 4 total messages from scratch.sanity via localhost:9092:
  - SANITY_MSG_001
  - SANITY_MSG_002
  - SANITY_MSG_003
  - {"source": "HOST_CLIENT", "msg": "HOST_SANITY_MSG_100"}
```
- **Reconciliation**: Zero listener address mismatches. Host connects to `localhost:9092`, in-container services connect to `kafka:29092`.
- **Verdict**: **PASS**

---

### Item 4 — Scratch Topic Cleanup & Verification
- **Command**:
  ```bash
  docker exec -i fraudshield-kafka kafka-topics --bootstrap-server localhost:9092 --delete --topic scratch.sanity
  docker exec -i fraudshield-kafka kafka-topics --bootstrap-server localhost:9092 --list
  ```

**Terminal Evidence:**
```text
__consumer_offsets
scores
transactions
transactions.dlq
```
- Confirmed `scratch.sanity` successfully deleted. Production topics `transactions`, `scores`, `transactions.dlq` remain intact.
- **Verdict**: **PASS**

---

### Item 5 — Resource Usage & Health After Stage B

**Command**:
```bash
docker stats --no-stream fraudshield-kafka fraudshield-postgres
```

**Terminal Evidence:**
```text
CONTAINER ID   NAME                   CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O     PIDS
a08851ffdb26   fraudshield-kafka      2.85%     319MiB / 768MiB     41.54%    12.9kB / 24.6kB   0B / 5.03MB   98
7e9ef477b926   fraudshield-postgres   0.00%     20.74MiB / 512MiB   4.05%     6.5kB / 5.18kB    0B / 582kB    6
```
- `fraudshield-kafka`: **319 MiB / 768 MiB (41.54%)**
- `fraudshield-postgres`: **20.74 MiB / 512 MiB (4.05%)**
- Host Free RAM: **1.68 GB** (1,724 MB).
- **Verdict**: **PASS**


---

## 5. Step 4: Streaming Verification — Stage C1 (Image Builds & Parity Audit)

### Item 1 — Image Builds & Inspect Verification
- **`fraudshield-serving:latest`**:
  - Image ID: `752e7cc6fd5f`
  - `docker image inspect` Size: **4,456,104,403 bytes** (4.15 GiB uncompressed virtual size).
  - Compressed Content Size: **1.05 GB** (export tarball format).
  - Breakdown (`docker history`):
    - Base Debian + Python 3.13: ~140 MB
    - `/opt/venv` dependencies: 1.5 GB
    - `COPY feature_repo/`: 789 MB (includes 752.7 MB SQLite `online_store.db`)
    - `COPY models/`: 88.5 MB
    - `RUN chown -R appuser:appgroup /app`: 879 MB (Docker duplicate layer from changing ownership of all copied files)
- **`fraudshield-stream:latest`**:
  - Image ID: `feae2f5a5ad3`
  - `docker image inspect` Size: **737,243,750 bytes** (703 MiB uncompressed virtual size).
  - Compressed Content Size: **172 MB**.
- **Verdict**: **PASS**

### Item 2 — Dependency & Version Parity Audit
Parity between host development environment (`.venv`) and container (`fraudshield-serving:latest`):
- `scikit-learn`: **1.9.1** (Host) vs **1.9.1** (Container) — 0 drift
- `xgboost`: **3.4.1** (Host) vs **3.4.1** (Container) — 0 drift
- `numpy`: **2.5.4** (Host) vs **2.5.4** (Container) — 0 drift
- `pandas`: **2.3.3** (Host) vs **2.3.3** (Container) — 0 drift
- **Verdict**: **PASS (0 Version Drift)**

---

## 6. Step 4: Streaming Verification — Stage C2 (`serving` Runtime Verification)

### Item 1 — Container Startup & Health
- **Pre-check Free RAM**: **2.42 GB** (>1.0 GB abort threshold).
- **Command**:
  ```bash
  docker compose -f docker/docker-compose.yml up -d --no-build serving
  docker compose -f docker/docker-compose.yml ps serving
  ```
- **Terminal Evidence**:
  ```text
  NAME                  IMAGE                        COMMAND                  SERVICE   CREATED         STATUS                   PORTS
  fraudshield-serving   fraudshield-serving:latest   "uvicorn src.serving…"   serving   5 minutes ago   Up 5 minutes (healthy)   0.0.0.0:8000->8000/tcp
  ```
- **Verdict**: **PASS**

### Item 2 — Model Loading & Fallback Evidence
**Startup Log Evidence (`docker logs fraudshield-serving`)**:
```text
{"timestamp": "2026-11-04T06:38:17.213777+00:00", "level": "INFO", "logger": "src.serving.scoring", "message": "Initializing FraudShield ScoringEngine..."}
{"timestamp": "2026-11-04T06:38:17.244603+00:00", "level": "INFO", "logger": "src.serving.scoring", "message": "Loaded fitted preprocessor from /app/models/xgboost_preprocessor.joblib"}
{"timestamp": "2026-11-04T06:38:18.188903+00:00", "level": "INFO", "logger": "src.serving.scoring", "message": "Attempting to load model from MLflow: models:/FraudShield-XGBoost/1"}
{"timestamp": "2026-11-04T06:43:00.881647+00:00", "level": "INFO", "logger": "src.serving.scoring", "message": "Could not load from MLflow registry (API request to http://mlflow:5000/api/2.0/mlflow/registered-models/get failed with exception HTTPConnectionPool(host='mlflow', port=5000): Max retries exceeded with url: /api/2.0/mlflow/registered-models/get?name=FraudShield-XGBoost (Caused by NameResolutionError(\"HTTPConnection(host='mlflow', port=5000): Failed to resolve 'mlflow' ([Errno -2] Name or service not known)\"))). Falling back to local artifact."}
{"timestamp": "2026-11-04T06:43:01.031063+00:00", "level": "INFO", "logger": "src.serving.scoring", "message": "Successfully loaded model from /app/models/xgboost_model.json with n_jobs=1"}
{"timestamp": "2026-11-04T06:43:10.212036+00:00", "level": "INFO", "logger": "src.training.blend", "message": "Loaded GNN risk lookup for 2468895 accounts from models/gnn_scores_lookup.parquet"}
{"timestamp": "2026-11-04T06:43:10.278788+00:00", "level": "INFO", "logger": "src.serving.scoring", "message": "Loaded GNN risk lookup for fast serving."}
{"timestamp": "2026-11-04T06:43:13.242059+00:00", "level": "INFO", "logger": "src.serving.scoring", "message": "Connected to Feast FeatureStore at /app/feature_repo"}
{"timestamp": "2026-11-04T06:43:13.242171+00:00", "level": "INFO", "logger": "src.serving.scoring", "message": "ScoringEngine initialization complete."}
{"timestamp": "2026-11-04T06:43:13.322869+00:00", "level": "INFO", "logger": "src.serving.app", "message": "Service started. Model loaded: True, Feast: True, DB: True (postgresql)"}
```
- **Local Model Path Loaded**: `/app/models/xgboost_model.json`
- **Exact MLflow Fallback Line**:
  `"Could not load from MLflow registry (API request to http://mlflow:5000/api/2.0/mlflow/registered-models/get failed with exception HTTPConnectionPool(host='mlflow', port=5000): Max retries exceeded with url: /api/2.0/mlflow/registered-models/get?name=FraudShield-XGBoost (Caused by NameResolutionError(\"HTTPConnection(host='mlflow', port=5000): Failed to resolve 'mlflow' ([Errno -2] Name or service not known)\"))). Falling back to local artifact."`
- **Feast Online Store Fallback Status**: Did **NOT** fall back to default empty feature dictionaries; successfully connected: `"Connected to Feast FeatureStore at /app/feature_repo"`.
- **Verdict**: **PASS**

### Item 3 — Endpoint Verification (`/health` & `/score`)
- **GET `http://localhost:8000/health`**:
  ```json
  {
      "status": "healthy",
      "model_loaded": true,
      "feast_reachable": true,
      "db_reachable": true
  }
  ```

- **Parity Comparison (Container FastAPI Endpoint vs Host In-Process Engine)**:
  - **Normal Transaction Payload** (`amount: 45.50`, `type: PAYMENT`):
    - Container Output: `fraud_score: 0.00111500`, `is_alert: false`
    - Host Output: `fraud_score: 0.00111500`
    - Parity Delta: **`0.000000000000`**
  - **Suspicious Transaction Payload** (`amount: 950000.00`, `type: TRANSFER`, balance emptied):
    - Container Output: `fraud_score: 0.99935400`, `is_alert: true` (threshold: 0.99)
    - Host Output: `fraud_score: 0.99935400`
    - Parity Delta: **`0.000000000000`**
- **Verdict**: **PASS (Zero Parity Delta)**

### Item 4 — Resource Usage & Memory Headroom (`docker stats`)
- **Command**:
  ```bash
  docker stats --no-stream fraudshield-serving fraudshield-kafka fraudshield-postgres
  ```
- **Terminal Evidence**:
  ```text
  CONTAINER ID   NAME                   CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
  49c31d2ffe2f   fraudshield-serving    0.48%     491.1MiB / 512MiB   95.93%    35kB / 30.8kB     1.37GB / 1.47GB   22
  a08851ffdb26   fraudshield-kafka      10.85%    376.5MiB / 768MiB   49.02%    13.2kB / 24.6kB   95.1MB / 84.4MB   98
  7e9ef477b926   fraudshield-postgres   26.72%    33.1MiB / 512MiB    6.46%     35.4kB / 36.8kB   47.9MB / 1.92MB   6
  ```
- **Analysis**:
  - `fraudshield-serving` consumes **491.1 MiB / 512 MiB (95.93%)**, leaving only 20.9 MiB of headroom. While not OOMKilled on single requests, multi-event bursts during Stage C3 will exceed 512 MiB.
  - Raising memory limit from **512M to 768M** is recommended to ensure stability during batch scoring.
- **Verdict**: **PASS (Operational, Recommend 768M Cap Increase)**
