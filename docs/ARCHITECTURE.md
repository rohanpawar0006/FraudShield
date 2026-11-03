# FraudShield System Architecture

This document describes the architectural topology, data pipelines, scoring workflows, and deployment topology of **FraudShield**, an end-to-end real-time fraud detection and operations platform.

---

## 1. System Component Architecture

FraudShield integrates feature engineering, machine learning inference, real-time event streaming, operational relational persistence, human-in-the-loop analyst triage, and continuous drift monitoring into a unified ecosystem.

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion & Streaming Layer"]
        Replay["Transaction Replay Producer<br/>(scripts/replay_transactions.py)"]
        KafkaTxn["Kafka Topic: transactions<br/>(KRaft Single Broker)"]
        KafkaDLQ["Kafka Topic: transactions.dlq<br/>(Dead Letter Queue)"]
    end

    subgraph Serving["Serving & Inference Layer (FastAPI)"]
        Consumer["Streaming Consumer Worker<br/>(src/streaming/consumer.py)"]
        FastAPI["FastAPI Scoring Service<br/>(POST /score, GET /health, /metrics)"]
        Engine["ScoringEngine<br/>(Single-threaded n_jobs=1)"]
        Preprocessor["Preprocessor Transformer<br/>(OneHotEncoder + Scaler)"]
        ChampionXGB["Champion Booster<br/>(MLflow Registered @champion)"]
        GNNLookup["GNN Mule / Ring Risk Table<br/>(Account ID Map)"]
    end

    subgraph FeatureStore["Feature Layer (Feast)"]
        FeastOffline["Parquet Offline Store<br/>(data/processed/)"]
        FeastOnline["SQLite Online Store<br/>(feature_repo/data/online_store.db)"]
    end

    subgraph Storage["Persistence & Registry"]
        MLflow["MLflow Tracking & Registry<br/>(sqlite:///mlruns.db)"]
        DB["PostgreSQL / SQLite Storage<br/>(transactions, scores, cases, labels)"]
    end

    subgraph Operations["Human-in-the-Loop & Monitoring"]
        Dashboard["Streamlit Analyst Dashboard<br/>(Alerts, Case Review, Feedback)"]
        Evidently["Evidently Drift Monitor<br/>(Wasserstein & KS Test Pipeline)"]
        RetrainTrigger["Retraining & Gate Set Evaluator<br/>(src/monitoring/retrain_trigger.py)"]
    end

    %% Streaming Flow
    Replay -->|JSON Events| KafkaTxn
    KafkaTxn -->|Consume Batch| Consumer
    Consumer -->|Process or Error| KafkaDLQ
    Consumer -->|Score Call| FastAPI

    %% Serving Dependencies
    FastAPI --> Engine
    Engine --> Preprocessor
    Engine --> ChampionXGB
    Engine --> GNNLookup
    Engine -->|Online Features| FeastOnline
    FastAPI -->|Async Persistence Queue| DB

    %% Model Tracking
    MLflow -->|Load Champion Booster| ChampionXGB

    %% Analyst & Monitoring
    DB -->|Read Alerts & Cases| Dashboard
    Dashboard -->|Submit Ground Truth Labels| DB
    DB -->|Read Recent Scores & Txns| Evidently
    Evidently -->|Trigger on Drift| RetrainTrigger
    DB -->|Read Labeled Data| RetrainTrigger
    RetrainTrigger -->|Register & Flip Alias| MLflow
    RetrainTrigger -->|POST /admin/reload-model| FastAPI
```

---

## 2. Request & Feedback Loop Flow

The scoring path decouples high-throughput, low-latency transaction evaluation from relational database persistence. Background worker queues guarantee sub-50ms execution while securing transactional audit trails.

```mermaid
sequenceDiagram
    autonumber
    actor Client as Payment Client / Kafka Consumer
    participant API as FastAPI (POST /score)
    participant Feast as Feast Online Store (SQLite)
    participant Engine as ScoringEngine (n_jobs=1)
    participant Queue as Async DB Persistence Queue
    participant DB as Relational DB (Postgres/SQLite)
    actor Analyst as Fraud Analyst (Streamlit)
    participant MLflow as MLflow Model Registry

    Client->>API: POST /score (Transaction Payload JSON)
    API->>API: Pydantic Schema Validation
    API->>Feast: Retrieve Historical Aggregates (orig_txn_count_30d, avg_amt)
    alt Feast Cache Miss (Cold-Start Account)
        Feast-->>API: Return Safe Default Features (feature_fallback=True)
    else Feast Cache Hit
        Feast-->>API: Return Historical Aggregate Features
    end
    API->>Engine: Pass Feature Vector (Tabular + Feast Features)
    Engine->>Engine: Recompute Derived Ratios (orig_amount_vs_mean)
    Engine->>Engine: Run XGBoost Predict Proba + Query GNN Mule Risk Table
    Engine->>Engine: Compute S_blend = alpha * S_xgb + (1 - alpha) * S_gnn
    Engine-->>API: Return Fraud Score & Alert Flag (Score >= Threshold)
    API->>Queue: Push (txn, score, alert) to Background Worker Queue
    API-->>Client: HTTP 200 JSON Response (Latency < 20ms)
    
    Note over Queue,DB: Asynchronous Non-Blocking Persistence
    Queue->>DB: Batch Insert transactions, scores, and auto-open cases

    Note over DB,Analyst: Human-in-the-Loop Analyst Triage
    Analyst->>DB: Query Open Cases (Filter by Alert Score)
    Analyst->>Analyst: Review Network Ring & History
    Analyst->>DB: Submit Confirmed Label (FRAUD / LEGITIMATE)
    
    Note over DB,MLflow: Automated Retraining & Gate Set Evaluation
    DB->>MLflow: Retrain on Full Data + Eligible Analyst Labels
    MLflow->>API: POST /admin/reload-model (Zero-Downtime Reload)
```

---

## 3. Temporal Splitting & Leakage Isolation

To enforce strict production realism and prevent time-travel data snooping, FraudShield partitions the PaySim dataset strictly along chronological boundaries.

```mermaid
gantt
    title PaySim Chronological Data Partitioning Timeline (1 Step = 1 Hour)
    dateFormat X
    axisFormat Step %s
    
    section Training Window
    Train Set (Steps 1 to 323, 4.46M txns, 3,643 frauds) :active, train, 1, 323
    Graph Topology Construction (Steps 1 to 323)        :train_g, 1, 323
    Feast Offline Features Fit (Steps 1 to 323)          :train_f, 1, 323
    
    section Validation Window
    Validation Set (Steps 324 to 378, 980k txns, 564 frauds) :crit, val, 324, 378
    Hyperparameter Tuning & Threshold Sweep (t=0.99)         :val_t, 324, 378
    Control Drift Baseline Window                             :val_d, 324, 340
    
    section Hold-Out Test Window
    Frozen Evaluation Gate Set (15,000 txns, 115 frauds)     :gate, 379, 410
    Hold-Out Test Split (Steps 379 to 743, 918k txns, 4,006 frauds) :test, 379, 743
    Real-Time Replay & Smoke Verification                     :replay, 379, 743
```

### Architectural Guardrails Against Data Leakage
1. **Target Leakage Fields Dropped:** `newbalanceOrig` (drained to $0.00$ during fraud) and `newbalanceDest` are excluded prior to model input.
2. **Account Entity Separation:** Account identifiers (`nameOrig`, `nameDest`) are utilized solely as entity join keys for Feast feature store lookups and graph adjacency lookups, never passed directly to tree splitters.
3. **Out-of-Sample Preprocessing:** Scalers, category encoders, and mean-amount lookups are fitted strictly on steps $\le 323$ and applied out-of-sample to validation and test steps.
4. **Frozen Gate Set Isolation:** The automated retrain validation gate set is frozen in time from hold-out data. Candidate models must improve PR-AUC by $\ge +0.0010$ and preserve recall on this immutable slice before receiving traffic.

---

## 4. Deployment Topology (Docker Compose vs Kubernetes)

FraudShield provides dual deployment specifications engineered for operational flexibility:

```mermaid
flowchart TB
    subgraph HostEnv["Host Operating Environment"]
        Dev["Native Python Development<br/>(.venv, Python 3.13)"]
    end

    subgraph ComposeTopology["Docker Compose Multi-Container Stack (docker/docker-compose.yml)"]
        direction TB
        C_Zk["zookeeper: 2181"]
        C_Kafka["kafka: 9092<br/>(transactions, scores, dlq)"]
        C_DB["postgres: 5432<br/>(fraudshield_db)"]
        C_MLflow["mlflow: 5000<br/>(Tracking Server)"]
        C_Serving["serving: 8000<br/>(FastAPI Scoring API)"]
        C_Producer["producer: worker<br/>(Stream Replay)"]
        C_Consumer["consumer: worker<br/>(Kafka Stream Processor)"]
        C_Dashboard["dashboard: 8501<br/>(Streamlit Analyst UI)"]
        C_Drift["drift-monitor: cron<br/>(Scheduled Drift Detector)"]

        C_Zk --- C_Kafka
        C_Kafka --- C_Serving
        C_Kafka --- C_Consumer
        C_Kafka --- C_Producer
        C_DB --- C_Serving
        C_DB --- C_Consumer
        C_DB --- C_Dashboard
        C_MLflow --- C_Serving
    end

    subgraph K8sTopology["Kubernetes Cluster Manifests (k8s/)"]
        direction TB
        K_Namespace["Namespace: fraudshield"]
        K_Config["ConfigMap & Secret<br/>(01-configmap.yaml, 02-secret.yaml)"]
        K_DB["PostgreSQL StatefulSet & PVC<br/>(03-postgres.yaml)"]
        K_Serving["Serving Deployment (Replicas: 1)<br/>(05-serving.yaml)"]
        K_Dashboard["Dashboard Deployment<br/>(06-dashboard.yaml)"]
        K_Cron["Evidently Drift CronJob<br/>(07-drift-cronjob.yaml)"]
        K_Ingress["NGINX Ingress & Services<br/>(08-ingress.yaml)"]

        K_Namespace --> K_Config
        K_Config --> K_DB
        K_Config --> K_Serving
        K_Config --> K_Dashboard
        K_Config --> K_Cron
        K_Serving --- K_Ingress
        K_Dashboard --- K_Ingress
    end

    Dev -.->|Containerize| ComposeTopology
    ComposeTopology -.->|Cloud-Native Spec| K8sTopology
```

### Deployment Scope & Verification Status
- **Docker Compose Stack:** Complete multi-service definition with healthchecks, persistent volumes, and inter-service dependencies.
- **Kubernetes Production Manifests:** Fully defined declarative specifications (StatefulSets, Deployments, ClusterIP Services, CronJobs, Ingress rules). `k8s/05-serving.yaml` is configured with `replicas: 1` to guarantee thread-safe SQLite online store access.
- **Host Execution Transparency:** Both Docker Compose and Kubernetes manifests are written and statically verified. Container daemons and Minikube virtual clusters were not provisioned on this host, preserving zero-dependency local Python reproducibility.

