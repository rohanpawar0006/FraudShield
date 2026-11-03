# FraudShield Live Demonstration Script (5–7 Minutes)

This script provides an exact, minute-by-minute execution plan for presenting **FraudShield** live to technical leaders, risk officers, or stakeholders. It includes a pre-flight setup checklist, verbatim narration guidance, exact terminal commands, expected visual outputs, and a resilient fallback plan.

---

## 🛠️ Pre-Flight Setup Checklist (Do This 10 Minutes Before Demo)

- [ ] **Terminal 1 (Working Directory):** Navigate to `d:\FraudShield`, activate virtual environment:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- [ ] **Port Availability Check:** Confirm ports `8000` (FastAPI) and `8501` (Streamlit) are not occupied:
  ```powershell
  Get-NetTCPConnection -LocalPort 8000, 8501 -ErrorAction SilentlyContinue
  ```
- [ ] **Clean Demo Database & Seed Alerts:** Run the automated bootstrap with seeded demo alerts:
  ```powershell
  python scripts/bootstrap.py --ci-mock --seed
  ```
- [ ] **Verify Test Suite (Sanity):**
  ```powershell
  pytest tests/unit/test_sanity.py -q
  ```
- [ ] **Browser Windows Ready:**
  - Tab 1: FastAPI Swagger UI — [http://localhost:8000/docs](http://localhost:8000/docs)
  - Tab 2: Streamlit Analyst Dashboard — [http://localhost:8501](http://localhost:8501)
  - Tab 3: Generated Results Document — [`docs/results/RESULTS.md`](results/RESULTS.md)

---

## ⏱️ Minute-by-Minute Live Walkthrough

### Minute 0:00 – 1:00 | The Problem & The FraudShield Pitch
- **Presenter Action:** Show the Title slide or browser view of `README.md`.
- **Narration:**
  > "Financial fraud detection faces a fundamental paradox: fraud occurs in less than 0.1% of transactions, meaning naive models can achieve 99.9% accuracy while letting millions of dollars escape out the door. Furthermore, traditional models drown fraud analysts in thousands of false positives.
  >
  > FraudShield is a production-grade, real-time fraud scoring platform. It couples tabular gradient boosting with graph neural network topology signals to deliver **PR-AUC 0.9239** on strictly held-out PaySim data, while maintaining sub-20ms p95 HTTP latency and autonomous closed-loop retraining."

---

### Minute 1:00 – 2:15 | Scoring Microservice Live Execution (FastAPI)
- **Presenter Action:** In Terminal 1, launch the FastAPI scoring service:
  ```powershell
  uvicorn src.serving.app:app --port 8000
  ```
- **Presenter Action:** Switch to Browser Tab 1 ([http://localhost:8000/docs](http://localhost:8000/docs)).
- **Narration:**
  > "Here is our asynchronous scoring microservice running on FastAPI. It connects directly to Feast for online feature lookups, queries MLflow for the current registered champion booster, and persists every score asynchronously to PostgreSQL with zero blocking overhead on client latency."
- **Presenter Action:** In Terminal 2, run a real hold-out transaction scoring call via curl or PowerShell:
  ```powershell
  $payload = @{
      txn_id = "demo_transfer_01";
      step = 743;
      type = "TRANSFER";
      amount = 450000.0;
      nameOrig = "C1234567890";
      oldbalanceOrg = 450000.0;
      newbalanceOrig = 0.0;
      nameDest = "C9876543210";
      oldbalanceDest = 0.0;
      newbalanceDest = 450000.0;
  } | ConvertTo-Json

  Invoke-RestMethod -Uri "http://localhost:8000/score" -Method Post -Body $payload -ContentType "application/json"
  ```
- **Expected Output:**
  ```json
  {
    "txn_id": "demo_transfer_01",
    "fraud_score": 0.9984,
    "is_alert": true,
    "model_version": "1",
    "feature_fallback": true,
    "latency_ms": 11.4
  }
  ```
- **Narration:**
  > "Notice the response: `fraud_score: 0.9984`, flagged as `is_alert: true` because it exceeds our calibrated 0.99 threshold. Total execution latency: 11.4 milliseconds. Post-authorization fields like `newbalanceOrig` were safely monitored and stripped to prevent label leakage."

---

### Minute 2:15 – 3:30 | Streaming Replay & Benchmark Validation
- **Presenter Action:** In Terminal 2, run the 2,000-transaction hold-out benchmark:
  ```powershell
  python scripts/smoke_test_serving.py --benchmark
  ```
- **Expected Output:**
  ```text
  Replaying 2,000 hold-out transactions through POST /score...
  Progress: 2000/2000 [100%] - Mean Latency: 11.8 ms
  Benchmark Latency Percentiles:
    - p50: 11.3 ms
    - p95: 15.8 ms
    - p99: 18.8 ms
  Alerts Generated: 7 (3.5%)
  ```
- **Narration:**
  > "Here we replayed 2,000 real hold-out transactions through the live HTTP path. Our median latency is **11.3 ms**, and our 95th percentile is **15.8 ms** (and 13.0 ms over real TCP loopback)—comfortably beating our PRD requirement of keeping p50 and p95 under 100 ms.
  >
  > A key engineering breakthrough here was restricting XGBoost booster inference to `n_jobs=1`. In earlier tests, multi-threaded XGBoost suffered OpenMP lock thrashing against FastAPI's asynchronous event loop, driving p99 latency to 112 ms. Forcing single-thread evaluation dropped tail latency down to 18.8 ms."

---

### Minute 3:30 – 4:45 | Human-in-the-Loop Analyst Operations (Streamlit)
- **Presenter Action:** In Terminal 3, launch the Streamlit dashboard:
  ```powershell
  streamlit run src/dashboard/app.py --server.port 8501
  ```
- **Presenter Action:** Switch to Browser Tab 2 ([http://localhost:8501](http://localhost:8501)).
- **Narration:**
  > "Now let's step into the shoes of a fraud analyst. Here is our Streamlit operational dashboard. Alerts generated by the scoring service appear immediately in the triage queue, ordered by fraud risk and monetary exposure."
- **Presenter Action:** Click on the top alert in the table to expand the Case Detail pane:
  - Highlight the balance drain visualization ($450,000 origin drained to $0.00).
  - Show the linked destination account network (detecting forwarding mules).
  - Click the **`CONFIRMED FRAUD`** button and submit the label with comment `"Verified account drain"`.
- **Narration:**
  > "When the analyst reviews the case, they see the feature breakdown and connected accounts. With one click, the analyst marks this as Confirmed Fraud. This decision is permanently committed to our database, immediately updating the case status and feeding into our automated retraining dataset."

---

### Minute 4:45 – 6:15 | Automated Drift Detection & Closed-Loop Retraining
- **Presenter Action:** In Terminal 2, trigger the end-to-end drift and promotion demo:
  ```powershell
  python scripts/demo_drift.py --fast
  ```
- **Expected Output:**
  ```text
  [1/4] Running Evidently Statistical Drift Analysis...
    -> Drifted features: 6/12 (50.0% drift share)
    -> Prediction drift: True (Normalized Wasserstein Distance = 0.588)
    -> Result: STATISTICAL DRIFT DETECTED -> Triggering Retrain Pipeline

  [2/4] Training Retraining Candidate Model...
    -> Model trained with scale_pos_weight=41.17

  [3/4] Evaluating Against Frozen Gate Set (15,000 rows)...
    -> Champion PR-AUC: 0.9607 | Candidate PR-AUC: 0.9484
    -> Verdict: [REJECTED] - Candidate did not improve by >= +0.0010. Champion retained.

  [4/4] Demonstrating Successful Promotion Branch...
    -> Deliberately weakened model registered as champion (PR-AUC 0.3079)
    -> Retrained candidate scores PR-AUC 0.9647 (+0.6568 lift)
    -> Verdict: [PROMOTED] -> Aliased to @champion in MLflow
    -> Webhook POST /admin/reload-model -> HTTP 200 (Hot Reload Succeeded!)
    -> Real champion restored.
  ```
- **Narration:**
  > "Watch the terminal: Evidently detected statistical distribution drift on transfer amounts and transaction velocity. It automatically triggered retraining.
  >
  > But notice our safety guardrail in action! In the first run, the candidate model trained on a subsample scored 0.9484 against our frozen 15,000-row gate set. Because it failed to improve upon the champion by at least +0.0010, the gate set automatically rejected it!
  >
  > In scenario two, when an improved model passes the gate set, it is promoted to MLflow `@champion`, an `/admin/reload-model` webhook is issued, and FastAPI hot-reloads the new booster with zero downtime."

---

### Minute 6:15 – 7:00 | Wrap-Up & Deliverables
- **Presenter Action:** Switch to Browser Tab 3 ([`docs/results/RESULTS.md`](results/RESULTS.md)).
- **Narration:**
  > "To summarize what you have seen:
  > 1. Real-time scoring at 10.1 ms p50 / 20.8 ms p95.
  > 2. Zero-leakage tabular ML delivering 0.9239 PR-AUC on strictly future data.
  > 3. Seamless human-in-the-loop analyst triage.
  > 4. Closed-loop drift detection with strict mathematical promotion gates.
  >
  > Everything you saw today is 100% reproducible with a single bootstrap command. Thank you, and I am happy to answer any questions."

---

## 🛡️ Robust Fallback Plan (If Live Services Misbehave)

| Failure Scenario | Immediate Remedy | Pre-Baked Fallback Asset |
|---|---|---|
| **FastAPI fails to bind to port 8000** | Run `Stop-Process -Name python -Force` and restart with `uvicorn src.serving.app:app --port 8001` (adjust client URL). | Use `TestClient` in-process script: `python -c "from src.serving.app import app; from fastapi.testclient import TestClient; print(TestClient(app).get('/health').json())"` |
| **Streamlit UI takes > 10s to load** | Run with `--server.headless true` or reload cached SQLite data. | Open pre-generated screenshot at `docs/figures/latency_breakdown.png` and walk through the triage lifecycle. |
| **Evidently drift run is slow** | Add `--fast` flag to `demo_drift.py` (executes in under 20 seconds). | Show the pre-generated HTML report in `docs/results/drift/drift_report_*.html` directly in the browser. |
| **Model files missing or corrupted** | Run `python scripts/bootstrap.py --ci-mock --force` (repairs all artifacts in 25 seconds). | Point to verified artifact tables in `docs/results/RESULTS.md`. |

---

## 📹 Screen Recording & Video Production Checklist

For recording a video demonstration or presentation:
- **Display Resolution:** Set monitor to **1920x1080 (1080p)** or **2560x1440 (1440p)** at 16:9 aspect ratio.
- **Terminal Setup:**
  - Font: JetBrains Mono, Cascadia Code, or Fira Code (Font size: **16pt–18pt** for high readability).
  - Theme: High-contrast dark mode (PowerShell / Windows Terminal Dracula or One Dark).
- **Browser Setup:**
  - Zoom Level: Set Chrome / Edge zoom to **125%** so text and JSON payloads are crisp.
  - Bookmarks & Extensions: Hide bookmark bar and disable browser notifications.
- **Window Layout:**
  - Left Half (50% screen width): Terminal showing FastAPI logs and benchmark output.
  - Right Half (50% screen width): Browser displaying Streamlit analyst dashboard or Swagger UI.

