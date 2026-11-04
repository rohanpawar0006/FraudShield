# FraudShield — End-to-End Data Flow & Architecture Verification

**Auditor:** Senior AI/ML & MLOps Architect  
**Date:** 2026-11-03  
**Document Status:** Verified Against Live Implementation  

---

## 1. Verified End-to-End Data Flow

The following lifecycle traces a transaction through the complete FraudShield infrastructure:

```mermaid
flowchart TD
    subgraph S1["Stage 1: Raw Ingestion & Temporal Splitting"]
        RAW["Raw PaySim CSV<br/>(493.5 MB, 6.36M rows)"] --> SPLIT["eda_and_split.py<br/>Strict Temporal Split (Step <= 323)"]
        SPLIT --> TRAIN_RAW["Train Raw (4,463,587 rows)"]
        SPLIT --> VAL_RAW["Val Raw (980,416 rows)"]
        SPLIT --> TEST_RAW["Test Raw (918,617 rows)"]
    end

    subgraph S2["Stage 2: Feature Pipeline & Feature Store"]
        TRAIN_RAW --> FEAT_ENG["feature_pipeline.py<br/>Sliding Windows (1h, 24h),<br/>Amount vs Mean, Velocity"]
        FEAT_ENG --> TRAIN_PARQUET["train_features.parquet"]
        FEAT_ENG --> FEAST_SQLITE["Feast FeatureStore<br/>(feature_repo/data/online_store.db)"]
    end

    subgraph S3["Stage 3: Offline Model Training & Registry"]
        TRAIN_PARQUET --> TRAIN_XGB["train_xgboost.py<br/>Hist-binned XGBClassifier<br/>(scale_pos_weight=1224:1)"]
        TRAIN_PARQUET --> BUILD_GRAPH["build_graph.py & gnn_model.py<br/>PyG GraphSAGE<br/>(Mule Ring Topology)"]
        TRAIN_XGB --> MLFLOW["MLflow Model Registry<br/>(@champion alias)"]
        BUILD_GRAPH --> GNN_LOOKUP["gnn_scores_lookup.parquet<br/>(Account Risk Map)"]
    end

    subgraph S4["Stage 4: Real-Time Serving & Streaming"]
        PAYLOAD["Transaction Event<br/>(JSON HTTP or Kafka)"] --> FASTAPI["FastAPI /score<br/>(src/serving/app.py)"]
        FASTAPI --> FEAST_FETCH["Feast get_online_features()<br/>(account_id key)"]
        FEAST_SQLITE -.-> FEAST_FETCH
        FASTAPI --> PREPROC["Fitted Preprocessor<br/>(ColumnTransformer)"]
        PREPROC --> XGB_SCORE["XGBoost Inference (n_jobs=1)"]
        MLFLOW -.-> XGB_SCORE
        FASTAPI --> GNN_FETCH["GNNRiskLookup.get_score()"]
        GNN_LOOKUP -.-> GNN_FETCH
        XGB_SCORE & GNN_FETCH --> BLEND["compute_blend_score()<br/>(Convex Combination)"]
        BLEND --> RESP["JSON Response<br/>(fraud_score, is_alert, latency_ms)"]
        BLEND --> ASYNC_Q["Background Persistence Queue<br/>(Non-blocking Write)"]
        PAYLOAD -.->|Streaming Broker| KAFKA["Kafka Topic: transactions<br/>(or In-Memory Channel)"]
        KAFKA --> CONSUMER["ScoringConsumer"]
        CONSUMER --> ASYNC_Q
        CONSUMER -.->|Malformed / Schema Error| DLQ["Topic: transactions-dlq"]
    end

    subgraph S5["Stage 5: Database & Analyst Operations"]
        ASYNC_Q --> DB[("PostgreSQL / SQLite<br/>(transactions, scores, cases, labels)")]
        DB --> STREAMLIT["Streamlit Analyst Dashboard<br/>(dashboard/streamlit_app.py)"]
        STREAMLIT --> TRIAGE["Alert Queue Triage & SHAP Inspector"]
        TRIAGE --> FEEDBACK["Analyst Feedback<br/>('Confirmed Fraud', 'False Positive')"]
        FEEDBACK --> LABELS_TABLE["labels Table (Audit Trail)"]
    end

    subgraph S6["Stage 6: Drift Monitoring & Automated Retraining"]
        DB --> EVIDENTLY["Evidently Drift Analysis<br/>(Wasserstein & KS Metric)"]
        EVIDENTLY --> TRIGGER["check_retrain_trigger()<br/>(Drift Share >= 0.30 OR Labels >= 10)"]
        LABELS_TABLE --> RETRAIN_PREP["prepare_retraining_dataset()"]
        DB --> RETRAIN_PREP
        RETRAIN_PREP --> RETRAIN_XGB["train_candidate_model()"]
        RETRAIN_XGB --> GATE_SET["Frozen Gate Set<br/>(15,000 rows, 230 frauds)"]
        GATE_SET --> GATE_EVAL["evaluate_promotion_gate()<br/>(1,000 Bootstrap CI on Delta PR-AUC)"]
        GATE_EVAL -->|Promoted| PROMOTE["Update MLflow @champion &<br/>POST /admin/reload-model"]
        GATE_EVAL -->|Rejected| AUDIT_LOG["retrain_audit_log.json<br/>(Immutable Record)"]
    end
```

---

## 2. Stage-by-Stage Verification Table

| Stage | Input Schema | Output Schema | Error Handling & Fallback | Data Leakage Guardrail | Verification Verdict |
|---|---|---|---|---|---|
| **1. Splitting** | PaySim 11-column CSV (`step`, `type`, `amount`, `nameOrig`, etc.) | Temporal splits: train ($\le 323$), val ($324..378$), test ($379..743$) | Missing file raises `FileNotFoundError`; type enforcement. | Strict step cutoff; zero future information in training. | **PASS** |
| **2. Feast** | Processed train DataFrame with `event_timestamp` and `created` | SQLite feature tables (`online_store.db`) | SQLite locking handling; fallback defaults for missing entities. | Entities stamped at chronological transaction occurrence. | **PASS** |
| **3. Modeling** | 11 feature columns (`type`, `amount`, `oldbalanceOrg`, etc.) | XGBoost booster JSON + joblib preprocessor + GNN risk parquet | Cold-start accounts map to `None`; MLflow fallback to local JSON. | Target column `isFraud` strictly excluded from features; `newbalanceOrig/Dest` removed. | **PASS** |
| **4. Serving** | `TransactionPayload` Pydantic model (`txn_id`, `type`, `amount`, etc.) | `ScoringResponse` (`fraud_score`, `is_alert`, `latency_ms`) | Schema rejection $\rightarrow$ 422; missing Feast $\rightarrow$ `DEFAULT_FEATURE_VALUES`; DLQ on poison events. | Post-auth balances ignored; single-thread booster prevents OpenMP thrashing. | **PASS** |
| **5. Database** | Tuple of `(txn_id, account_id, amount, type, ts, raw_json)` | Relational tables: `transactions`, `scores`, `cases`, `labels` | Primary key conflicts use `ON CONFLICT DO NOTHING / UPDATE`. | Read/write separation; async queue avoids serving timeouts. | **PASS** |
| **6. Dashboard** | Query parameters (`min_score`, `status`, `txn_type`, pagination) | Tabular alerts, SHAP waterfall, graph context, feedback confirmation | Dispositions validated against `{'Confirmed Fraud', 'False Positive', 'Under Review'}`. | Parameterized SQL prevents SQL injection. | **PARTIAL** (Fixed score filter bug). |
| **7. Monitoring & Retraining** | Inference window vs frozen reference window | Drift summary JSON, HTML report, candidate model, promotion gate verdict | Underpowered gate set accounted for via 1,000 bootstrap CI. | Frozen gate set excluded from training via set disjointness assertion. | **PARTIAL** (Fixed label incorporation bug). |

