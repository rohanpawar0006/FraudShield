# FraudShield — Deployment & Operations Runbook

**Generated**: 2026-11-03  
**Auditor**: Senior DevOps & MLOps Architect  
**Audience**: DevOps Engineers, Platform Engineers, and Incident Responders  
**Scope**: Docker Compose, Kubernetes Clusters, Health Verification, and Safe Rollback Procedures.

---

## 1. Architecture & Deployment Targets

FraudShield supports two containerized operational deployment models in addition to local development execution:

```
[Target 1: Local / Standalone]  ──> Python .venv + SQLite Fallback + In-Memory Streaming
[Target 2: Docker Compose]       ──> Multi-Container Core/Stream Stacks (Postgres, MLflow, Kafka, FastAPI, Streamlit)
[Target 3: Kubernetes Cluster]   ──> Namespace fraudshield with StatefulSets, Deployments, CronJobs, and ConfigMaps
```

---

## 2. Target 1 — Local Python Development Execution

*Verification Status*: **Runtime Verified** on Windows host.

### 2.1 Start FastAPI Serving Service
```powershell
# In terminal 1:
$env:FRAUDSHIELD_ENV="development"
$env:USE_SQLITE_FALLBACK="true"
.venv\Scripts\uvicorn.exe src.serving.app:app --host 0.0.0.0 --port 8000
```

### 2.2 Start Streamlit Analyst Console
```powershell
# In terminal 2:
$env:USE_SQLITE_FALLBACK="true"
.venv\Scripts\streamlit.exe run dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### 2.3 Verification Commands
```powershell
# Health check
curl -X GET http://127.0.0.1:8000/health

# Operational telemetry
curl -X GET http://127.0.0.1:8000/metrics

# Streamlit status
curl -X GET http://127.0.0.1:8501/_stcore/health
```

---

## 3. Target 2 — Docker Compose Deployment

*Verification Status*: **Unit Tested & Compose Parsed**; requires Docker daemon.

### 3.1 Pre-Deployment Secret & Configuration Setup
1. Copy the environment configuration template:
   ```bash
   cp .env.example .env
   ```
2. Generate a secure `ADMIN_API_KEY`:
   ```bash
   # Linux/macOS:
   export ADMIN_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
   sed -i "s/dev_admin_secret_key_change_in_prod/$ADMIN_KEY/g" .env

   # Windows PowerShell:
   $adminKey = python -c "import secrets; print(secrets.token_urlsafe(32))"
   (Get-Content .env) -replace "dev_admin_secret_key_change_in_prod", $adminKey | Set-Content .env
   ```
3. Set secure database credentials in `.env`:
   ```ini
   POSTGRES_USER=fraudshield_admin
   POSTGRES_PASSWORD=generated_secure_db_password_123!
   POSTGRES_DB=fraudshield
   ```

### 3.2 Build & Launch Profiles
Docker Compose defines three modular profiles:
- `core`: PostgreSQL, MLflow, FastAPI Serving, Streamlit Dashboard.
- `stream`: Core + Kafka KRaft, Topic Initializer, Replay Producer, Scoring Consumer.
- `full`: Stream + Drift Monitoring Worker.

```bash
# Option A: Core Serving & Dashboard stack
docker compose --profile core up -d --build

# Option B: Full Streaming, Ingestion, and Monitoring stack
docker compose --profile full up -d --build
```

### 3.3 Health Verification Commands
```bash
# 1. Verify container status
docker compose ps

# 2. Check Serving health (waits for startup)
curl -f http://localhost:8000/health

# 3. Check Dashboard health
curl -f http://localhost:8501/_stcore/health

# 4. Check MLflow UI
curl -f http://localhost:5000/

# 5. Check Kafka topics
docker exec -it fraudshield-kafka kafka-topics --bootstrap-server localhost:9092 --list
```

### 3.4 Stopping & Tear-Down
```bash
# Stop containers while preserving database volume
docker compose down

# Stop containers and remove volumes (DESTRUCTIVE - wipes database)
docker compose down -v
```

---

## 4. Target 3 — Kubernetes Cluster Deployment

*Verification Status*: **Unit Tested via Kubeconform v0.6.7**; ready for deployment to Minikube, Kind, or production EKS/GKE/AKS.

### 4.1 Secret Generation (Kubernetes Secrets)
Never commit production secrets to Git. Apply secrets directly using `kubectl` or SealedSecrets/Vault:

```bash
# 1. Generate secrets imperatively:
kubectl create namespace fraudshield --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic fraudshield-secrets \
  --namespace fraudshield \
  --from-literal=POSTGRES_USER="fraudshield_app" \
  --from-literal=POSTGRES_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')" \
  --from-literal=ADMIN_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  --dry-run=client -o yaml | kubectl apply -f -
```

Alternatively, edit placeholders in `k8s/02-secrets.yaml` with your secret manager before applying.

### 4.2 Apply Manifests in Sequence
```bash
# 1. Namespace
kubectl apply -f k8s/00-namespace.yaml

# 2. ConfigMap & Secrets
kubectl apply -f k8s/01-configmap.yaml
kubectl apply -f k8s/02-secrets.yaml

# 3. PostgreSQL Database (ConfigMap DDL + PVC + Service + StatefulSet)
kubectl apply -f k8s/03-postgres.yaml

# Wait for PostgreSQL pod readiness
kubectl wait --namespace fraudshield --for=condition=ready pod -l app.kubernetes.io/name=postgres --timeout=90s

# 4. Apache Kafka (KRaft Single-Pod + Topic Init Job for dev/demo clusters)
kubectl apply -f k8s/04-kafka.yaml

# 5. FastAPI Serving Engine (Deployment + ClusterIP Service)
kubectl apply -f k8s/05-serving.yaml

# 6. Streamlit Dashboard (Deployment + NodePort Service)
kubectl apply -f k8s/06-dashboard.yaml

# 7. Drift Monitoring & Retraining CronJob
kubectl apply -f k8s/07-drift-cronjob.yaml
```

### 4.3 Cluster Health & Readiness Verification
```bash
# 1. Verify all pods running and ready
kubectl get pods -n fraudshield -o wide

# 2. Verify Services and endpoints
kubectl get svc -n fraudshield

# 3. Port-forward Serving service to test API
kubectl port-forward -n fraudshield svc/serving 8000:8000 &
curl http://localhost:8000/health

# 4. Port-forward Dashboard service
kubectl port-forward -n fraudshield svc/dashboard 8501:8501 &
curl http://localhost:8501/_stcore/health

# 5. Verify PostgreSQL DDL initialization ran
kubectl exec -it -n fraudshield statefulset/postgres -- psql -U postgres -d fraudshield -c "\dt"
```
**Expected Output from `\dt`**:
```
              List of relations
 Schema |     Name      | Type  |  Owner   
--------+---------------+-------+----------
 public | cases         | table | postgres
 public | labels        | table | postgres
 public | scores        | table | postgres
 public | transactions  | table | postgres
```

---

## 5. Rollback Procedures

### 5.1 FastAPI Serving Rollback
If a newly deployed model version or container image causes regressions:

```bash
# Check rollout history
kubectl rollout history deployment/serving -n fraudshield

# Roll back to previous deployment revision immediately (zero-downtime)
kubectl rollout undo deployment/serving -n fraudshield

# Verify rollback completion
kubectl rollout status deployment/serving -n fraudshield
```

### 5.2 Model Champion Rollback via Administrative API
If a retrained model candidate fails post-deployment verification without requiring a container rollback:

```bash
# 1. Inspect current champion alias in MLflow
python -c "
import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient('http://localhost:5000')
print('Current champion:', client.get_model_version_by_alias('FraudShield-XGBoost', 'champion').version)
"

# 2. Demote candidate and re-point champion alias to version 1
python -c "
from mlflow.tracking import MlflowClient
client = MlflowClient('http://localhost:5000')
client.set_registered_model_alias('FraudShield-XGBoost', 'champion', '1')
print('Champion alias rolled back to version 1')
"

# 3. Trigger live serving model reload without restarting containers
curl -X POST http://localhost:8000/admin/reload-model \
  -H "X-Admin-Key: $ADMIN_API_KEY"
```

### 5.3 Database Rollback & Recovery
- PostgreSQL persistent storage is backed by `PersistentVolumeClaim: postgres-pvc`.
- Container rollouts or pod restarts do **not** wipe or re-initialize data.
- If schema rollback is required, run targeted SQL migrations against the active database:
  ```bash
  kubectl exec -i -n fraudshield statefulset/postgres -- psql -U postgres -d fraudshield < scripts/rollback_schema.sql
  ```

