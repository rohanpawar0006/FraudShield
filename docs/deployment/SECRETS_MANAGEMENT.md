# FraudShield — Secrets Management & Security Architecture

**Generated**: 2026-11-03  
**Auditor**: Senior Backend, DevOps & MLOps Architect  
**Scope**: API Keys, Database Credentials, Message Broker Auth, Kubernetes Secrets, and Fail-Closed Policies.

---

## 1. Secrets Inventory & Component Matrix

| Secret Name | Intended Usage | Where Injected | Production Policy | Dev Policy | Verification State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ADMIN_API_KEY`** | Protects `/admin/reload-model` and `/admin/retrain` | Serving pod via `k8s/02-secrets.yaml` / `.env` | **Fail-Closed**: Unset key crashes/rejects with HTTP 500; invalid key returns 401 | Allowed with warning; auth bypassed if unset | **Unit Tested & Integration Tested** |
| **`POSTGRES_USER`** | PostgreSQL DB admin user | Postgres & Serving pods via Secret | High-entropy alphanumeric string | Defaults to `postgres` | **Unit Tested** |
| **`POSTGRES_PASSWORD`**| PostgreSQL DB password | Postgres & Serving pods via Secret | High-entropy string ($\ge 24$ chars); rotation via Vault | Defaults to `postgres` | **Unit Tested** |
| **`KAFKA_SASL_PASSWORD`**| SASL/SCRAM Kafka auth (production) | Producer & Consumer pods | Enforced for external/managed MSK/Confluent brokers | Plaintext (KRaft dev mode) | **Architectural / Documented** |

---

## 2. Administrative Security & Fail-Closed Enforcement

### 2.1 The Problem
Administrative endpoints perform operational actions:
- `/admin/reload-model`: Pulls latest champion from MLflow/disk and hot-swaps model in memory.
- `/admin/retrain`: Evaluates drift/feedback triggers and initiates training & promotion gate checks.

In production environments, exposing these endpoints without strict authentication exposes the system to Denial-of-Service or unauthorized model swap attacks.

### 2.2 Security Implementation (`src/serving/app.py`)
In [`src/serving/app.py`](file:///d:/FraudShield/src/serving/app.py#L167-L201), `verify_admin_key()` enforces:

```python
def verify_admin_key(
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> None:
    admin_key = config_module.settings.ADMIN_API_KEY
    is_prod = config_module.settings.ENV.lower() in ("production", "prod")

    # 1. Fail-closed guard: In production, missing configuration is a server error
    if is_prod and not admin_key:
        logger.error("ADMIN_API_KEY must be configured in production!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server security misconfiguration: ADMIN_API_KEY not set",
        )

    # 2. Key verification using constant-time comparison against timing attacks
    if admin_key:
        if not x_admin_key or not secrets.compare_digest(x_admin_key, admin_key):
            logger.warning("Unauthorized administrative access attempt.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Missing or invalid X-Admin-Key header",
            )
    elif is_prod:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Admin access disabled without API key",
        )
    else:
        # Development-mode only: log audit warning
        logger.warning("Administrative operation invoked without ADMIN_API_KEY configured (dev-only).")
```

---

## 3. Secret Generation & Ingestion Guidelines

### 3.1 Generating Cryptographically Secure Secrets
Never invent passwords or use default strings like `admin123` or `password`. Generate cryptographic entropy using Python's standard `secrets` library:

```bash
# Generate 32-byte URL-safe base64 string for ADMIN_API_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Output example: Yk9wXQ7vP8sM_eN2fT5hR1uC0zA6bJ3xE4lV8gW9oI0

# Generate 24-byte secure alphanumeric string for POSTGRES_PASSWORD
python3 -c "import secrets; print(secrets.token_urlsafe(24))"
# Output example: K9_mZ2xL8vP1qW7tE4rU6yI0
```

---

### 3.2 Setting Secrets in Docker Compose
1. Ensure `.env` is listed in [`.gitignore`](file:///d:/FraudShield/.gitignore). **Never commit `.env` to Git.**
2. Set generated values in `.env`:
   ```ini
   FRAUDSHIELD_ENV=production
   ADMIN_API_KEY=Yk9wXQ7vP8sM_eN2fT5hR1uC0zA6bJ3xE4lV8gW9oI0
   POSTGRES_USER=fraudshield_admin
   POSTGRES_PASSWORD=K9_mZ2xL8vP1qW7tE4rU6yI0
   POSTGRES_DB=fraudshield
   ```

---

### 3.3 Setting Secrets in Kubernetes

#### Option A: Imperative Secret Creation (Recommended for CI/CD pipelines)
```bash
kubectl create secret generic fraudshield-secrets \
  --namespace fraudshield \
  --from-literal=POSTGRES_USER="fraudshield_admin" \
  --from-literal=POSTGRES_PASSWORD="<GENERATED_POSTGRES_PASSWORD>" \
  --from-literal=ADMIN_API_KEY="<GENERATED_ADMIN_API_KEY>" \
  --dry-run=client -o yaml | kubectl apply -f -
```

#### Option B: External Secrets Operator / HashiCorp Vault (Recommended for Enterprise)
In enterprise clusters, integrate `external-secrets.io` with AWS Secrets Manager or HashiCorp Vault. Map the remote secret vault to `Secret: fraudshield-secrets` in namespace `fraudshield`.

---

## 4. Secret Verification Commands

Test that administrative authentication functions correctly:

```bash
# 1. Test request without header (Should return HTTP 401 Unauthorized)
curl -i -X POST http://localhost:8000/admin/reload-model

# 2. Test request with wrong key (Should return HTTP 401 Unauthorized)
curl -i -X POST http://localhost:8000/admin/reload-model \
  -H "X-Admin-Key: invalid-key-attempt"

# 3. Test request with valid key (Should return HTTP 200 OK)
curl -i -X POST http://localhost:8000/admin/reload-model \
  -H "X-Admin-Key: $ADMIN_API_KEY"
```

---

## 5. Secret Rotation Procedure

1. **Rotating `ADMIN_API_KEY`**:
   - Update Kubernetes secret:
     ```bash
     kubectl patch secret fraudshield-secrets -n fraudshield \
       --type='json' -p='[{"op": "replace", "path": "/data/ADMIN_API_KEY", "value": "'$(echo -n "$NEW_KEY" | base64)'"}]'
     ```
   - Restart the serving deployment to reload the environment variable:
     ```bash
     kubectl rollout restart deployment/serving -n fraudshield
     ```
2. **Rotating `POSTGRES_PASSWORD`**:
   - Change password in PostgreSQL engine first (`ALTER USER postgres WITH PASSWORD 'new_pass';`).
   - Update `fraudshield-secrets`.
   - Perform rolling restart of dependent deployments (`serving`, `dashboard`, `drift-worker`).

