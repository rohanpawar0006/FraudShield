# Feature Engineering & Feast Architecture Documentation

## 1. Feature Definitions & Time-Window Semantics

All features are engineered strictly point-in-time safe (causal): at transaction time $t$, **only events strictly prior to $t$ are accessible**. The current transaction is never included in historical aggregations, guaranteeing zero future leakage.

| Feature Name | Type | Entity | Window / Scope | Description & Edge Case Handling |
|---|---|---|---|---|
| `orig_txn_count_1h` | `int64` | `nameOrig` | Prior 1 hour $[t-1, t)$ | Count of previous outbound transactions by the same origin account in the preceding 1 hour. Default = 0 for first transaction. |
| `orig_txn_count_24h` | `int64` | `nameOrig` | Prior 24 hours $[t-24, t)$ | Count of previous outbound transactions by the same origin account in the preceding 24 hours. Default = 0 for first transaction. |
| `orig_amount_vs_mean` | `float64` | `nameOrig` | All prior history $[0, t)$ | Ratio of current transaction amount to historical mean amount: `amount / (historical_mean + 1e-5)`. Default = 1.0 for first transaction. |
| `orig_time_since_last_txn` | `float64` | `nameOrig` | Prior transaction | Elapsed time in hours since the immediate preceding transaction: $t - t_{prev}$. Default = -1.0 for first transaction. |
| `orig_has_prev_txn` | `int64` | `nameOrig` | Lifetime $[0, t)$ | Binary indicator: 1 if `nameOrig` has conducted at least one prior transaction; 0 otherwise. |
| `dest_txn_count_24h` | `int64` | `nameDest` | Prior 24 hours $[t-24, t)$ | Count of previous inbound transactions received by `nameDest` in the preceding 24 hours. Default = 0 for first arrival. |
| `dest_has_prev_txn` | `int64` | `nameDest` | Lifetime $[0, t)$ | Binary indicator: 1 if `nameDest` has received at least one prior transaction; 0 otherwise. |

---

## 2. PaySim Domain Characteristics & Limitations

1. **Origin Account Frequency:**
   In PaySim, origin accounts (`nameOrig`) represent individual consumers. Across 6.36M transactions, 99.8% of origin accounts conduct only 1 transaction, and none conduct more than 3. Consequently, deterministic handling of cold-start / unseen accounts is foundational to model stability.
2. **Destination Account Frequency:**
   Destination accounts (`nameDest`) represent merchants (`M...`) or recipient accounts (`C...`) and frequently receive repeat transactions (up to 108 in the training split). High inbound destination velocity (`dest_txn_count_24h`) is a strong indicator of cash-out laundering hubs.
3. **Absence of Device & Geo Telemetry:**
   PaySim does not contain device IDs, IP addresses, browser fingerprints, or GPS coordinates. Per `Rules.md`, we do not invent synthetic device or location fields. Device and geo changes are documented as dataset limitations and will be addressed in future phases via graph connectivity patterns instead.

---

## 3. Feast Feature Store Configuration

- **Configuration:** [feature_repo/feature_store.yaml](file:///d:/FraudShield/feature_repo/feature_store.yaml)
  - **Project:** `fraudshield`
  - **Provider:** `local`
  - **Registry:** `feature_repo/data/registry.db`
  - **Offline Store:** `file` (Parquet files in `data/processed/`)
  - **Online Store:** `sqlite` (`feature_repo/data/online_store.db`)
- **Entities:**
  - `nameOrig`: Join key for origin account feature views.
  - `nameDest`: Join key for destination account feature views.
- **Feature Views:**
  - `origin_account_features`: TTL = 30 days, backed by `data/processed/train_features.parquet`.
  - `destination_features`: TTL = 30 days, backed by `data/processed/train_features.parquet`.

> **Architecture Note on SQLite:** Local SQLite is used exclusively for local development, CI testing, and fast zero-skew verification. In production/Kubernetes deployment (Phase 8), the online store provider configuration seamlessly transitions to Redis as defined in `docker/docker-compose.yml`.

---

## 4. Train-Serve Parity & Skew Prevention

To guarantee zero offline/online skew:
1. **Single Source of Truth:** Feature definitions are written once in `src/features/feature_pipeline.py`.
2. **Automated Parity Suite:** [tests/unit/test_feast_parity.py](file:///d:/FraudShield/tests/unit/test_feast_parity.py) materializes offline features to Feast online store and asserts exact matching against online `fs.get_online_features()` with numerical tolerance $\le 10^{-4}$.
3. **Unseen Entity Contract:** Querying Feast for an unseen entity deterministically returns `None`, which our serving pipeline maps to predefined cold-start defaults (`has_prev=0`, `time_since=-1.0`, `count=0`, `amount_vs_mean=1.0`).
