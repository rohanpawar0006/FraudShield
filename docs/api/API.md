# FraudShield REST API Reference

The FraudShield scoring service exposes a high-performance, asynchronous REST API powered by FastAPI.

- **Base URL:** `http://localhost:8000` (or `http://serving:8000` in Docker/Kubernetes)
- **OpenAPI JSON Specification:** [`docs/api/openapi.json`](openapi.json)
- **Interactive Documentation (Swagger UI):** `http://localhost:8000/docs`
- **Alternative Documentation (ReDoc):** `http://localhost:8000/redoc`

---

## 1. Endpoints

### 1.1 Score Transaction (`POST /score`)
Scores a single incoming transaction for fraudulent activity using the blended XGBoost + GNN inference engine.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Body
```json
{
  "txn_id": "txn_sample_1001",
  "step": 743,
  "type": "TRANSFER",
  "amount": 250000.0,
  "nameOrig": "C1234567890",
  "oldbalanceOrg": 250000.0,
  "newbalanceOrig": 0.0,
  "nameDest": "C0987654321",
  "oldbalanceDest": 0.0,
  "newbalanceDest": 250000.0
}
```

| Field | Type | Description |
|---|---|---|
| `txn_id` | string (optional) | Unique client transaction ID. If omitted, a UUID is automatically generated. |
| `step` | integer (required) | Hourly timestamp index ($1 \le step \le 743$). |
| `type` | string (required) | Transaction type (`TRANSFER`, `CASH_OUT`, `CASH_IN`, `PAYMENT`, `DEBIT`). |
| `amount` | float (required) | Transaction amount in USD ($> 0$). |
| `nameOrig` | string (required) | Origin customer account identifier. |
| `oldbalanceOrg` | float (required) | Origin balance prior to transaction. |
| `newbalanceOrig` | float (required) | Origin balance post transaction (monitored, dropped prior to model input to avoid label leakage). |
| `nameDest` | string (required) | Destination customer/merchant account identifier. |
| `oldbalanceDest` | float (required) | Destination balance prior to transaction. |
| `newbalanceDest` | float (required) | Destination balance post transaction (monitored, dropped prior to model input). |

#### Response (`200 OK`)
```json
{
  "transaction_id": "txn_sample_1001",
  "fraud_score": 0.9984,
  "xgboost_score": 0.9984,
  "gnn_score": 0.0512,
  "blend_score": 0.9984,
  "blend_used": true,
  "is_alert": true,
  "decision_threshold": 0.99,
  "feature_fallback": false,
  "latency_ms": 11.45
}
```

| Field | Type | Description |
|---|---|---|
| `transaction_id` | string | ID of the scored transaction. |
| `fraud_score` | float | Final assigned risk score $[0.0, 1.0]$. |
| `xgboost_score` | float | Raw XGBoost gradient boosted decision tree risk probability. |
| `gnn_score` | float | Graph neural network topological risk score (or 0.0 if cold-start). |
| `blend_score` | float | Linearly weighted blend ($S = \alpha \cdot S_{\text{xgb}} + (1-\alpha) \cdot S_{\text{gnn}}$). |
| `blend_used` | boolean | True if GNN edge/node risk was successfully factored into the score. |
| `is_alert` | boolean | True if `fraud_score` $\ge$ `DECISION_THRESHOLD` (0.99). |
| `decision_threshold` | float | Active threshold applied to generate the alert. |
| `feature_fallback` | boolean | True if Feast online lookup missed and safe default features were substituted. |
| `latency_ms` | float | Server-side end-to-end scoring elapsed time in milliseconds. |

#### Error Responses
- **`422 Unprocessable Entity`**: Payload schema validation failure (e.g., negative amount, invalid type string). In streaming mode, malformed payloads are rejected and routed to `transactions.dlq`.
- **`503 Service Unavailable`**: Scoring model or preprocessor uninitialized at startup.

---

### 1.2 Health Check (`GET /health`)
Evaluates the status of all underlying dependencies (model in memory, Feast feature store reachability, and database connectivity).

#### Response (`200 OK` or `503 Service Unavailable`)
```json
{
  "status": "healthy",
  "components": {
    "model_loaded": true,
    "feast_online_store": "reachable",
    "database": "reachable"
  },
  "model_version": "FraudShield-XGBoost:1",
  "timestamp": "2026-11-03T11:35:00.000000Z"
}
```

---

### 1.3 Service Telemetry Metrics (`GET /metrics`)
Provides real-time telemetry over a rolling window (default: 1,000 requests).

#### Response (`200 OK`)
```json
{
  "total_requests": 10542,
  "total_errors": 0,
  "fallback_requests": 142,
  "throughput_rps": 64.2,
  "latency": {
    "p50_ms": 15.4,
    "p95_ms": 19.3,
    "p99_ms": 27.6,
    "mean_ms": 15.8
  },
  "window_size": 1000
}
```

---

### 1.4 Dynamic Champion Model Reload (`POST /admin/reload-model`)
Performs a zero-downtime, in-place reload of the champion model and preprocessor without restarting the FastAPI service process. Triggered automatically by the automated retraining gate when a candidate model is promoted.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Body (Optional)
```json
{
  "model_path": "models/xgboost_model.json",
  "preprocessor_path": "models/xgboost_preprocessor.joblib"
}
```

#### Response (`200 OK`)
```json
{
  "status": "success",
  "message": "Champion model reloaded successfully in-place",
  "reloaded_at": "2026-11-03T11:35:10.123456Z"
}
```

