# FraudShield Deployment & Verification Guide

This document provides instructions for verifying and executing the FraudShield containerized stack and Kubernetes deployments across different environments, with specific instructions for GitHub Codespaces.

---

## 1. Running in GitHub Codespaces (Zero Setup, Pre-installed Docker)

GitHub Codespaces provides an instant, isolated Linux VM with Docker and Docker Compose pre-installed out of the box. To launch and verify the full FraudShield stack in Codespaces:
1. Open the FraudShield repository in GitHub and click **Code > Codespaces > Create codespace on main**.
2. Once the terminal opens, launch the core profile (FastAPI scoring + PostgreSQL + Redis + MLflow + Streamlit) by running:
   ```bash
   docker compose -f docker/docker-compose.yml --profile core up --build -d
   ```
3. Check service health with `docker compose -f docker/docker-compose.yml ps`. Ports 8000 (FastAPI API) and 8501 (Analyst Dashboard) will be automatically forwarded by Codespaces and visible under the **Ports** tab in VS Code.
4. Test the scoring endpoint directly via curl:
   ```bash
   curl -s http://localhost:8000/health
   curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{"step": 1, "type": "TRANSFER", "amount": 1000.0, "nameOrig": "C1", "oldbalanceOrg": 1000.0, "newbalanceOrig": 0.0, "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 1000.0}'
   ```
5. To shut down the environment, run:
   ```bash
   docker compose -f docker/docker-compose.yml down -v
   ```

---

## 2. Local Minikube Deployment Instructions

See [`k8s/README.md`](../k8s/README.md) for full commands to run on Minikube locally.

---

## 3. Environment Verification Status

| Tool | Presence on Current Host | Status / Verification Method |
|---|---|---|
| `docker` | Not installed | Static Dockerfile lint & unit tests; CI pipeline in `.github/workflows/containers.yml` |
| `docker compose` | Not installed | Static YAML parsing, port collision checks, and service dependency graphs |
| `kubectl` | Not installed | Manifest schema validation, selector matching, resource sizing checks |
| `minikube` | Not installed | Manifests sized for laptop constraints (<= 4 CPU, <= 4 GB RAM); documented commands |
| FastAPI Serving | **Verified** | Live unit tests (56/56 passing), `/health` and `/score` endpoints verified |
| Model & Features | **Verified** | Zero-skew feature parity, Feast offline store, MLflow registry loading |

