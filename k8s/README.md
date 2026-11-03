# FraudShield Kubernetes Deployment Guide (Minikube)

> [!IMPORTANT]
> **Status: Manifests written and statically validated, not run locally.** Docker, Minikube, and kubectl are not installed on the local Windows host. Full validation is executed in CI via `.github/workflows/containers.yml` and verifiable in GitHub Codespaces.
>
> **Scope Note:** Kafka, the streaming replay producer, consumer, and the MLflow tracking server are orchestrated via `docker-compose.yml` for local stream workflows. Kubernetes manifests provide productionized deployments for PostgreSQL storage, the FastAPI Scoring Service (`serving`, single replica safe for SQLite Feast store), the Streamlit Analyst Dashboard (`dashboard`), and the scheduled drift monitoring CronJob (`drift-monitor`).

This guide documents the complete procedure for deploying FraudShield on a local Minikube cluster sized for standard laptops (<= 4 CPU cores, <= 4 GB RAM).

---

## 1. Prerequisites & Minikube Startup

Ensure `minikube` and `kubectl` are installed on your host machine.

```bash
# Start Minikube with resources tailored for FraudShield
minikube start --cpus=4 --memory=4096 --driver=docker

# Point your local Docker CLI to Minikube's internal Docker daemon
# Linux/macOS:
eval $(minikube docker-env)
# Windows PowerShell:
& minikube -p minikube docker-env --shell powershell | Invoke-Expression

# Build container images directly inside Minikube's Docker daemon
docker build -f docker/Dockerfile.serving -t fraudshield-serving:latest .
docker build -f docker/Dockerfile.dashboard -t fraudshield-dashboard:latest .
docker build -f docker/Dockerfile.drift -t fraudshield-drift:latest .
```

---

## 2. Apply Manifests in Dependency Order

Apply Kubernetes manifests in sequence so that dependencies (Namespace, Secrets, ConfigMaps, Storage) exist before workloads initialize:

```bash
# Step 1: Create Namespace
kubectl apply -f k8s/00-namespace.yaml

# Step 2: Create ConfigMap and Secrets
kubectl apply -f k8s/01-configmap.yaml
kubectl apply -f k8s/02-secrets.yaml

# Step 3: Deploy Database (PostgreSQL StatefulSet + PVC)
kubectl apply -f k8s/03-postgres.yaml

# Step 4: Deploy Core Workloads (FastAPI Scoring & Streamlit Dashboard)
kubectl apply -f k8s/05-serving.yaml
kubectl apply -f k8s/06-dashboard.yaml

# Step 5: Deploy Scheduled Drift Monitoring CronJob
kubectl apply -f k8s/07-drift-cronjob.yaml
```

---

## 3. Wait for Rollout Status

Verify all pods and services reach `Running` and `Ready` status:

```bash
# Check pod startup and readiness
kubectl get pods -n fraudshield -w

# Wait explicitly for serving and dashboard rollouts
kubectl rollout status deployment/serving -n fraudshield --timeout=120s
kubectl rollout status deployment/dashboard -n fraudshield --timeout=120s
kubectl rollout status statefulset/postgres -n fraudshield --timeout=120s
```

---

## 4. Port-Forward Services

Expose the internal ClusterIP services to your host machine:

```bash
# Terminal 1: Port-forward FastAPI Serving Service (Port 8000)
kubectl port-forward svc/serving 8000:8000 -n fraudshield

# Terminal 2: Port-forward Streamlit Dashboard (Port 8501)
kubectl port-forward svc/dashboard 8501:8501 -n fraudshield
```

Alternatively, open the Streamlit dashboard via Minikube's NodePort service:
```bash
minikube service dashboard -n fraudshield
```

---

## 5. Send Test Scoring Request

Verify end-to-end scoring via `curl`:

```bash
# Check service health
curl -s http://localhost:8000/health | jq .

# Send transaction for fraud scoring
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "step": 743,
    "type": "TRANSFER",
    "amount": 950000.0,
    "nameOrig": "C1234567890",
    "oldbalanceOrg": 950000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C9876543210",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 950000.0
  }' | jq .
```

Expected Response:
```json
{
  "transaction_id": "...",
  "fraud_score": 0.9998,
  "is_alert": true,
  "latency_ms": 11.2,
  "feature_fallback": false
}
```

---

## 6. Trigger Drift Check Manually (Optional)

Test the drift monitoring CronJob on-demand:

```bash
kubectl create job --from=cronjob/drift-monitor manual-drift-check -n fraudshield
kubectl logs -f job/manual-drift-check -n fraudshield
```

---

## 7. Cluster Teardown

Clean up all FraudShield resources and terminate Minikube:

```bash
# Delete all FraudShield Kubernetes resources
kubectl delete namespace fraudshield

# Stop Minikube cluster
minikube stop

# (Optional) Delete Minikube cluster VM
minikube delete
```

