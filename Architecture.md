# Architecture.md — FraudShield

## 1. Overview
```
Producer (replay) ──► Kafka topic: transactions
                          │
                          ▼
                 FastAPI scoring consumer ──► Feast online store (Redis) ─► features
                          │                 └► XGBoost model (MLflow registry)
                          │                 └► GNN score (precomputed/refreshed node scores)
                          ▼
                 Blend → score ──► Kafka topic: scores ──► Postgres (scores, cases, labels)
                                                              │
                                  Streamlit analyst dashboard ◄┘
Training: Feast offline (parquet) → XGBoost/GNN → MLflow
Monitoring: Postgres scores + reference data → Evidently → drift flag → retrain job
```

## 2. Technology Stack
| Layer | Tech | Why | Alternatives |
|---|---|---|---|
| Streaming | Kafka (single-broker, KRaft) | Required; replay + decoupling | Redpanda |
| Features | Feast (file offline + Redis online) | Required; train/serve parity | Hand-rolled store |
| Tabular model | XGBoost | Required; strong on imbalanced tabular | LightGBM |
| Graph model | PyTorch Geometric (GraphSAGE/GAT) | Required; ring detection | DGL |
| Tracking | MLflow (tracking + registry) | Required | W&B |
| Serving | FastAPI + Pydantic + uvicorn | Required; async, typed | Flask |
| Drift | Evidently AI | Required | Alibi Detect |
| Store | PostgreSQL (SQLite for dev) | Cases/labels need relational store | SQLite only |
| Dashboard | Streamlit | Fastest for 14 days [REC] | React |
| Orchestration | Docker, Minikube, docker-compose fallback | Required K8s; fallback for safety | kind |
| Testing/CI | pytest, GitHub Actions, ruff | Standard | — |

## 3. Components
- **Producer:** reads dataset ordered by time, publishes JSON events at configurable rate.
- **Scoring service:** consumes events, validates, fetches features, scores, publishes/stores result. Also exposes REST for single-transaction scoring.
- **Feature repo:** Feast entity (account), feature views (velocity, amount ratios, device/location change).
- **Training pipeline:** `train_xgb.py`, `train_gnn.py`, `blend.py`, all logged to MLflow.
- **Graph builder:** builds account/device/card graph from training window; GNN produces node-level risk used at serving time.
- **Drift monitor:** scheduled job; Evidently report + threshold → retrain trigger.
- **Dashboard:** alerts, case detail, linked-account view, feedback form.

## 4. Data Flow
Raw CSV → clean/split by time → Feast offline → train → MLflow registry → serving loads `Production` model → event in → features → score → persisted → analyst label → Postgres → next training set.

## 5. API Architecture (REST, JSON)
| Endpoint | Purpose |
|---|---|
| `POST /score` | Score one transaction → `{txn_id, xgb_score, gnn_score, blended_score, model_version, is_alert}` |
| `GET /alerts?status=&limit=` | List alerts |
| `GET /cases/{txn_id}` | Case detail incl. linked accounts |
| `POST /cases/{txn_id}/label` | Analyst label `{label, comment}` |
| `GET /health`, `GET /metrics` | Ops |
| `POST /admin/retrain` | Manual retrain trigger |
- **Validation:** Pydantic models; 422 on bad input. **Errors:** structured `{error, detail, request_id}`; malformed Kafka events → dead-letter topic.
- **Auth:** none in MVP (local demo) — documented limitation; optional API key header.

## 6. Database (Postgres)
- `transactions(txn_id PK, account_id, dest_id, amount, type, ts, raw jsonb)`
- `scores(txn_id FK, xgb_score, gnn_score, blended_score, model_version, scored_at, latency_ms)` — index on `(blended_score desc)`, `(scored_at)`
- `cases(txn_id FK, status, assigned_to, updated_at)`
- `labels(txn_id FK, label, comment, labeled_at)`
- **Lifecycle:** scores retained for drift windows; labels permanent (training data).

## 7. Folder Structure
```text
fraudshield/
├── data/                 # raw/ (gitignored), processed/
├── notebooks/            # 01_eda.ipynb, 02_features.ipynb, 03_gnn.ipynb
├── feature_repo/         # feature_store.yaml, features.py
├── src/
│   ├── ingestion/        # producer.py, schemas.py
│   ├── features/         # feature logic shared offline/online
│   ├── training/         # train_xgb.py, train_gnn.py, blend.py, evaluate.py
│   ├── graph/            # build_graph.py, gnn_model.py
│   ├── serving/          # app.py, consumer.py, scoring.py
│   ├── monitoring/       # drift.py, retrain_trigger.py
│   └── common/           # config.py, logging.py, db.py
├── dashboard/            # streamlit_app.py
├── k8s/                  # deployments, services, configmaps
├── docker/               # Dockerfiles, docker-compose.yml
├── tests/                # unit/, integration/
├── docs/                 # architecture diagram, API docs, results
├── PRD.md Architecture.md Rules.md Phases.md Design.md Memory.md
└── README.md
```

## 8. Security
No secrets in repo (`.env`, K8s Secrets); validate all input; synthetic data only; rate limiting optional via middleware; dashboard is local-only in MVP.

## 9. Scalability
Kafka partitions keyed by account_id + multiple consumer replicas; Redis online store scales reads; model loaded in memory per pod; GNN scores precomputed to keep latency low.

## 10. Failure Points
| Risk | Handling |
|---|---|
| Feast online miss (new account) | Default/cold-start features + flag |
| GNN score unavailable/stale | Fallback to XGBoost-only |
| Kafka consumer lag | Metrics + autoscale note; backpressure via replay rate |
| Bad event | Dead-letter topic + log |
| Minikube resource limits | docker-compose fallback; subsample |
| Retrain produces worse model | Promote only if PR-AUC ≥ current on hold-out |
