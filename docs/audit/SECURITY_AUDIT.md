# FraudShield — Security & Compliance Audit Report

**Auditor:** Senior AI/ML & MLOps Architect  
**Date:** 2026-11-03  
**Status:** Remediated & Hardened  

---

## 1. Security Architecture & Threat Matrix

| Threat Category | Target Component | Existing State | Assessed Risk | Remediation Action |
|---|---|---|---|---|
| **Administrative Access Control** | `POST /admin/reload-model`, `POST /admin/retrain` | Header parameter accepted but unverified | **HIGH** | Implemented constant-time API key verification (`secrets.compare_digest`) via `ADMIN_API_KEY`. |
| **SQL Injection** | `src/feedback/service.py`, `src/common/db.py` | Parameterized SQL with `%s` / `?` placeholders | **LOW** | Verified 100% of SQL queries use parameterized arguments; zero dynamic string concatenation of user input. |
| **Input Validation / Poisoning** | `POST /score`, Kafka Ingestion | Pydantic v2 schema validation, DLQ routing | **LOW** | Malformed payloads rejected with HTTP 422 or sent to dead-letter queue; post-authorization fields sanitized. |
| **Secret Exposure** | `.env.example`, `k8s/02-secrets.yaml`, compose | Git-committed files contain local placeholders only | **LOW** | Verified zero production secrets or private tokens in tracked files; `.env` strictly gitignored. |
| **Timing Attacks** | Admin token comparison | Raw equality string comparison (`==`) | **MEDIUM** | Upgraded to constant-time comparison using `secrets.compare_digest`. |
| **Resource Exhaustion / DoS** | FastAPI scoring event loop | Single-thread XGBoost (`n_jobs=1`), timeouts | **LOW** | Non-blocking asynchronous SQLite persistence queue; container resource limits (< 4 GB). |
| **Container Privilege Escalation** | Dockerfiles (`serving`, `dashboard`, `stream`, `drift`) | Pinned `python:3.13-slim-bookworm`, unprivileged user | **LOW** | Multi-stage builds; enforced dedicated non-root user `appuser:appgroup` (UID/GID 10001). |

---

## 2. Admin Authentication Verification

Prior to remediation, `POST /admin/reload-model` in `src/serving/app.py` allowed arbitrary unauthenticated callers to force model reloads. 

### Secure Implementation Standard
1. **Configuration Setting:** Added `ADMIN_API_KEY: Optional[str]` in `src/common/config.py` sourced from `os.getenv("ADMIN_API_KEY")`.
2. **Constant-Time Verification:** Uses `secrets.compare_digest` to prevent side-channel timing attacks.
3. **Environment-Aware Hardening:**
   - In production (`ENV in ("production", "prod")`) or whenever `ADMIN_API_KEY` is configured: Requests missing `X-Admin-Key` or supplying an incorrect key are rejected with `HTTP 401 Unauthorized`.
   - In local development testing when no key is explicitly provisioned: Access is logged with a security warning to permit local integration tests while preventing silent production exposure.

