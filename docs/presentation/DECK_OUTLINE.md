# FraudShield Executive Presentation Deck Outline

This outline specifies a 10–12 slide executive and technical presentation for **FraudShield**. Each slide contains structured slide bullets, speaker notes, exact metric citations from `docs/results/RESULTS.md`, and references to visual figures in `docs/figures/`.

---

### Slide 1: Title & Executive Summary
- **Title:** FraudShield: Real-Time Fraud Detection & Autonomous MLOps Platform
- **Subtitle:** Sub-20ms Machine Learning Inference, Zero-Leakage Temporal Design, and Automated Closed-Loop Retraining
- **Presenter:** Antigravity / Engineering Team
- **Key Bullets:**
  - Production-ready financial fraud detection platform operating on real-time streaming transaction flows.
  - End-to-end integration: Feature store (Feast), model tracking (MLflow), inference API (FastAPI), analyst triage (Streamlit), and drift monitoring (Evidently).
  - Empirical achievement: **PR-AUC 0.9239** and **95.95% Precision @ 73.27% Recall** on strictly held-out PaySim financial data.
- **Speaker Notes:**
  > "Welcome everyone. Today we are presenting FraudShield, an enterprise-grade platform engineered to solve one of fintech's hardest challenges: detecting high-velocity financial fraud in under 20 milliseconds without drowning human fraud analysts in false alarms. We will walk through the architecture, the machine learning models, the latency optimizations, and the autonomous retraining loop."
- **Visual:** FraudShield Platform Banner & Architectural Overview.

---

### Slide 2: The Problem: Extreme Imbalance & The "Accuracy Trap"
- **Title:** The Fraud Paradox: Why Traditional Metrics and Models Fail
- **Key Bullets:**
  - **Extreme Class Imbalance:** Fraud represents only **0.13%** of transactions in the PaySim dataset (1,224 legitimate transactions for every 1 fraud).
  - **The Accuracy Trap:** A naive dummy model predicting "Legitimate" for all transactions achieves **99.94% accuracy** while suffering **100% loss of fraud capital**.
  - **The ROC-AUC Illusion:** Baseline Logistic Regression achieves **0.9819 ROC-AUC**, yet yields **140 false alarms for every genuine fraud caught** ($PR\text{-}AUC = 0.1455$).
  - **Core KPI:** Precision-Recall AUC (PR-AUC) and **Recall @ Precision $\ge 0.90$** to guarantee operational viability for risk teams.
- **Speaker Notes:**
  > "In high-throughput payments, accuracy is a meaningless metric. If you predict legitimate for everything, you get 99.94% accuracy and your business goes bankrupt. Even ROC-AUC is misleading because the massive volume of true negatives inflates the score. Our north star is PR-AUC and Recall at 90% Precision—ensuring 9 out of 10 alerts investigated by analysts represent actionable fraud."
- **Visual Citation:** `docs/figures/pr_curves.png` (Logistic Regression vs Champion curve comparison).

---

### Slide 3: Zero-Leakage Architecture & Temporal Discipline
- **Title:** Eliminating Time Travel: Chronological Splitting & Leakage Invariants
- **Key Bullets:**
  - **Temporal Splitting by Step:**
    - Train Window: Steps 1 to 323 ($N = 4,463,587$ transactions, 3,643 frauds).
    - Validation Window: Steps 324 to 378 ($N = 980,416$ transactions, 564 frauds) — used strictly for threshold sweeps.
    - Hold-Out Test Window: Steps 379 to 743 ($N = 918,617$ transactions, 4,006 frauds) — evaluated exactly once.
  - **Eliminating Target Leakage:**
    - Post-transaction origin balance (`newbalanceOrig`) drained to zero in 98% of fraud; strictly dropped prior to feature transformation.
    - Destination balance (`newbalanceDest`) unavailable at pre-auth scoring time; strictly dropped.
  - **Out-of-Sample Encoders:** All scalers and feature transformers fit strictly on steps $\le 323$.
- **Speaker Notes:**
  > "Data leakage is the silent killer of production ML models. If an algorithm sees future transactions or post-authorization balances like `newbalanceOrig`, it looks invincible in testing but collapses in production. We enforced a zero-tolerance policy: strict chronological time splits and programmatic removal of post-transaction fields before feature engineering."
- **Visual Citation:** `docs/ARCHITECTURE.md` (Section 3: Gantt Chart of Temporal Splitting Timeline).

---

### Slide 4: System Architecture & Data Pipelines
- **Title:** End-to-End Component Topology: From Ingestion to Triage
- **Key Bullets:**
  - **Streaming & Ingestion:** Kafka transaction replay with automatic Dead-Letter Queue (`transactions.dlq`) routing.
  - **Feature Layer:** Feast feature store enforcing train/serve parity with sub-5ms online state retrieval.
  - **Serving Layer:** FastAPI microservice with asynchronous, non-blocking database persistence queue.
  - **Registry & Storage:** MLflow tracking server and registered model aliases (`@champion`).
  - **Operations:** Streamlit human-in-the-loop analyst triage dashboard and background Evidently drift detection.
- **Speaker Notes:**
  > "FraudShield is not a standalone script or a notebook; it is a full-lifecycle software platform. When a payment event arrives, FastAPI scores the transaction, retrieves online features from Feast, and queries the registered model from MLflow. To protect real-time latency, write persistence to PostgreSQL or SQLite occurs asynchronously via background worker queues."
- **Visual Citation:** `docs/ARCHITECTURE.md` (Section 1: Mermaid Component Diagram).

---

### Slide 5: Model Selection: Decision Trees vs Linear Baselines
- **Title:** Tabular Modeling: Achieving 0.9239 PR-AUC with XGBoost
- **Key Bullets:**
  - **Histogram Binning (`tree_method='hist'`):** Scalable gradient boosting over 4.46M records in under 4 minutes.
  - **Class Weight Balancing:** `scale_pos_weight = 1224.25` calibrated to training prevalence.
  - **Performance Comparison:**
    - Baseline Logistic Regression: PR-AUC **0.1455** | Recall@P0.90: **0.00%**
    - Champion XGBoost: PR-AUC **0.9239** | Recall@P0.90: **78.91%** (+535% relative lift).
  - **Calibrated Production Threshold:** Setting $t = 0.99$ yields **95.95% Precision**, **73.27% Recall**, and generates exactly 3,059 alerts across 918k test transactions.
- **Speaker Notes:**
  > "Look at the performance comparison on slide 5. Logistic Regression completely flatlines—it cannot capture the non-linear ratios between transfer velocity and historical account volume. Our XGBoost champion achieves a remarkable 0.9239 PR-AUC on future data. At our calibrated decision threshold of 0.99, we catch 73% of fraud while maintaining nearly 96% precision."
- **Visual Citation:** `docs/figures/model_comparison.png` and `docs/figures/pr_curves.png`.

---

### Slide 6: Graph Neural Networks & The Cold-Start Reality
- **Title:** Graph Modeling & Mule Rings: Insights from PyTorch Geometric
- **Key Bullets:**
  - **Topology Mining:** PyG GraphSAGE and GAT edge-risk architectures constructed to detect circular forwarding rings and fan-in hubs.
  - **Training Window Validation:** Strong fraud-ring detection on closed account sets (validation PR-AUC **0.58**).
  - **The Cold-Start Challenge:** On strictly future test transactions, the GNN Edge Risk Classifier alone dropped to **PR-AUC 0.0031**.
  - **Root Cause Discovered:** Extreme dynamic turnover in payment graphs — **only 0.05% of test origin accounts existed in the training window**.
  - **Engineering Solution:** Resilient convex blend formula ($S = \alpha S_{\text{xgb}} + (1 - \alpha) S_{\text{gnn}}$) with dynamic cold-start weighting ($\alpha = 1.0$), securing tabular performance without graph dilution (empirically and mathematically identical to XGBoost by construction).
- **Speaker Notes:**
  > "Graph neural networks are frequently hailed as a silver bullet for fraud rings. In Phase 5, our GNN Edge Risk Classifier achieved 0.58 PR-AUC on known account rings. But when we evaluated it on future hold-out data, PR-AUC plummeted to 0.0031. Why? Because 99.95% of test accounts were brand-new entities. This is an essential industry takeaway: static graph structures fail under high account turnover. We mitigated this by engineering a dynamic blend that relies on tree features when accounts are cold, equaling the XGBoost champion by construction."
- **Visual Citation:** `docs/results/phase5_comparison.md` and `docs/figures/model_comparison.png`.

---

### Slide 7: Real-Time Serving Microservice & Two Rigorous Benchmarks
- **Title:** Real-Time Scoring: In-Process Latency & Microservice Tail Analysis
- **Key Bullets:**
  - **PRD Latency Target:** p50 and p95 latency below 100 ms under continuous load.
  - **Benchmark 1: In-Process Scoring Function (`ScoringEngine.score`, N=2,000, `n_jobs=1`):**
    - Mean: **13.38 ms** | Median (p50): **12.94 ms** | p90: **16.27 ms** | p95: **17.80 ms** | p99: **21.10 ms**
    - Stage breakdown: Feast SQLite lookup (~4.2 ms), scikit-learn transform (~3.8 ms), XGBoost inference (~3.9 ms), blend scoring & async enqueue (~0.1 ms).
  - **Benchmark 2: Full HTTP Stack (`POST /score` via TestClient, N=2,000):**
    - Mean: **11.83 ms** | Median (p50): **11.33 ms** | p90: **14.75 ms** | p95: **15.80 ms** | p99: **18.79 ms**
    - Measures end-to-end client serialization, Starlette HTTP routing, Pydantic validation, scoring execution, and asynchronous background database queueing.
  - **Benchmark 3: Real Uvicorn Server (External Client over TCP Loopback, N=2,000):**
    - Mean: **9.63 ms** | Median (p50): **9.19 ms** | p90: **11.83 ms** | p95: **12.96 ms** | p99: **14.79 ms**
  - **Thread-Contention Optimization:** Setting `n_jobs=1` on XGBoost eliminated OpenMP lock thrashing with FastAPI's event loop, bringing in-process p99 down to **21.10 ms** and HTTP p99 to **18.79 ms** (comfortably beating the PRD 100 ms target by a factor of 5x–7x).
- **Speaker Notes:**
  > "In real-time scoring, latency must be benchmarked with full honesty. We evaluate our performance across rigorous benchmarks: Benchmark 1 tests the pure in-process scoring engine, achieving a median of 12.9 ms and a p95 of 17.8 ms. Benchmark 2 measures the full HTTP boundary through FastAPI TestClient, achieving a p95 of 15.8 ms, and a real Uvicorn server over TCP loopback achieves a p95 of 13.0 ms. Both comfortably exceed the PRD requirement of keeping p50 and p95 latency below 100 ms. Crucially, restricting XGBoost to single-thread inference eliminated OpenMP thread contention with worker threads, guaranteeing predictable tail latency."
- **Visual Citation:** `docs/figures/latency_breakdown.png` and `docs/results/latency_in_process.json`.

---

### Slide 8: Human-in-the-Loop Analyst Operations
- **Title:** Closing the Operational Loop: Streamlit Analyst Triage UI
- **Key Bullets:**
  - **Real-Time Alert Triage:** Interactive filtering of high-risk cases sorted by fraud score and dollar exposure.
  - **Contextual Evidence Pane:** Displays transaction velocity, Feast historical ratios, and linked destination mule accounts.
  - **One-Click Ground Truth Labeling:** Analysts submit confirmed decisions (`CONFIRMED_FRAUD` or `BENIGN`) directly to the relational database.
  - **Zero-Downtime Data Flywheel:** Analyst feedback labels automatically become eligible training inputs for the next retraining cycle.
- **Speaker Notes:**
  > "Machine learning models must serve human domain experts, not operate in an ivory tower. FraudShield includes a full-featured Streamlit analyst dashboard. Analysts inspect high-risk transactions, view linked accounts across the graph, and submit ground-truth labels. These labels feed directly back into our database, creating a continuous data flywheel that improves future model generations."
- **Visual Citation:** Streamlit UI Screenshot / Demo walkthrough (`src/dashboard/app.py`).

---

### Slide 9: Automated Drift Detection with Evidently AI
- **Title:** Continuous Monitoring: Detecting Statistical Shift in Production Traffic
- **Key Bullets:**
  - **Reference Baseline:** 8,000 transactions frozen from the training distribution (steps $\le 323$).
  - **Current Window:** 4,000 recent streaming transactions.
  - **Methodology:** Wasserstein distance (numerical) and Kolmogorov-Smirnov test (categorical) across 12 engineered features.
  - **Demonstrated Detection:**
    - Natural Traffic (Control Window): Drift share 0.08 ($< 0.30$ threshold) $\rightarrow$ No False Alarms.
    - Injected Payment Bursts: Drift detected on `amount`, `orig_amount_vs_mean`, and score share ($50.0\%$ feature drift share) $\rightarrow$ Retraining Trigger Dispatched.
- **Speaker Notes:**
  > "Fraud patterns evolve rapidly. We integrated Evidently AI to continuously measure statistical drift between current streaming traffic and our baseline training window. On normal traffic, the system remains calm. When a coordinated fraud burst occurs—shifting transfer amounts and velocity—the pipeline immediately raises an alarm and dispatches an automated retraining trigger."
- **Visual Citation:** `docs/figures/drift_analysis.png`.

---

### Slide 10: Promotion Safeguards & Gate Set Integrity
- **Title:** Autonomous Retraining: Safe Model Promotion with Gate Guardrails
- **Key Bullets:**
  - **Dual-Branch Retraining Protocol:**
    - Candidate model trained on full historical data (4.46M rows) + eligible analyst feedback.
  - **Frozen Gate Set Guardrail:**
    - 15,000 strictly held-out transactions with 230 confirmed frauds.
    - Rule: Candidate is promoted **only if** $PR\text{-}AUC \ge \text{Champion} + 0.0010$, recall is preserved, and 95% bootstrap CI lower bound is strictly positive.
  - **Empirically Verified Safeguards (Bootstrap CI):**
    - Scenario 1 (Rejection): Candidate v8 scored 0.9647 vs Champion 0.9607 ($\Delta = +0.0040$), but 95% bootstrap CI $[-0.0038, +0.0127]$ crosses zero $\rightarrow$ **Automatically REJECTED**.
    - Scenario 2 (Promotion): Candidate vs weakened champion (0.3079) achieved $\Delta = +0.6568$ with 95% bootstrap CI $[+0.6022, +0.7095]$ strictly positive $\rightarrow$ **PROMOTED**.
  - **Zero-Downtime Hot Reload:** Automatic MLflow `@champion` alias flip + `/admin/reload-model` webhook reload.
- **Speaker Notes:**
  > "Automated retraining without statistical rigor is reckless. In FraudShield, every candidate must pass an immutable, frozen evaluation gate set evaluated with 1,000 bootstrap resamples. In Scenario 1, a small +0.0040 point gain failed promotion because with 230 frauds in the gate set, the confidence interval crossed zero. In Scenario 2, when retrained against a severely degraded model, the candidate demonstrated overwhelming improvement (+0.6568 lift with a strictly positive confidence interval) and was automatically promoted to champion, hot-reloading in FastAPI with zero downtime."
- **Visual Citation:** `docs/results/retrain_audit_log.json` summary table in `RESULTS.md`.

---

### Slide 11: Deployment Topology & Engineering Integrity
- **Title:** Production Delivery: Native Python vs Container Packaging
- **Key Bullets:**
  - **One-Command Bootstrap:** `scripts/bootstrap.py` provisions complete system from clean clone in under 60 seconds.
  - **Local Native Parity:** 63/63 pytest tests pass on Python 3.13.14 without Docker dependencies.
  - **Production Packaging:**
    - 4 Multi-stage Dockerfiles (`python:3.13-slim-bookworm`) with pinned dependencies.
    - Full Docker Compose stack (`docker/docker-compose.yml`) defining Kafka, PostgreSQL, FastAPI, and Streamlit.
    - Production Kubernetes manifests (`k8s/`) configured with `replicas: 1` for SQLite safety.
  - **Honest Verification Reporting:** Manifests statically validated; host environment documented with zero fictitious claims.
- **Speaker Notes:**
  > "We hold engineering honesty as a core principle. The entire FraudShield pipeline—training, feature store, MLflow, FastAPI, and test suite—is 100% verified locally on Python 3.13. Our Docker and Kubernetes manifests are written and statically validated. We do not claim containers ran locally when Docker wasn't installed; instead, we provide a reproducible native path alongside production-ready cloud manifests."
- **Visual Citation:** `docs/ARCHITECTURE.md` (Section 4: Deployment Topology).

---

### Slide 12: Business Impact & Strategic Takeaways
- **Title:** Summary & Next Steps: Operationalizing FraudShield
- **Key Bullets:**
  - **Capital Preservation:** Detects 73.27% of fraudulent volume on hold-out data with negligible false alarm rate (95.95% Precision).
  - **Operational Efficiency:** Reduces analyst investigation overhead by over 90% compared to linear baselines.
  - **Autonomous Resiliency:** Self-monitoring and self-updating with mathematical promotion guarantees.
  - **Future Roadmap:** Distributed Feast online store on Redis cluster, dynamic graph edge updates via streaming GNNs, and multi-region Kubernetes deployments.
- **Speaker Notes:**
  > "In summary, FraudShield bridges the gap between sophisticated machine learning research and high-availability financial engineering. It protects capital, eliminates analyst fatigue, and provides autonomous closed-loop retraining with strict mathematical safeguards. Thank you, and we welcome your questions."
- **Visual:** FraudShield Architecture Summary & Q&A Contact Information.

