# FraudShield — End-to-End Demonstration Guide

**Generated**: 2026-11-03  
**Auditor**: Senior Backend & MLOps Architect  
**Objective**: Provide a reproducible, step-by-step walkthrough to demonstrate the full FraudShield transaction lifecycle from ingestion to alert investigation and analyst-driven model retraining.

---

## 1. Prerequisites & Startup

FraudShield runs out of the box in the local Python environment using its SQLite fallback storage without requiring external Docker, Kafka, or PostgreSQL daemons.

### 1.1 Start FastAPI Serving Service
In terminal 1:
```powershell
.venv\Scripts\uvicorn.exe src.serving.app:app --host 127.0.0.1 --port 8000
```

### 1.2 Start Streamlit Analyst Console
In terminal 2:
```powershell
.venv\Scripts\streamlit.exe run dashboard/streamlit_app.py --server.port 8501
```

---

## 2. Step-by-Step Demonstration Script

### Step 1: Health & Readiness Verification
Verify that the serving engine, feature store, and persistence layer are active:

```powershell
curl -X GET http://127.0.0.1:8000/health
```

**Expected Output**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feast_reachable": true,
  "db_reachable": true
}
```

---

### Step 2: Submit a Normal Transaction
Submit a legitimate everyday payment (\$45.50 retail payment):

```powershell
curl -X POST http://127.0.0.1:8000/score `
  -H "Content-Type: application/json" `
  -d '{
    "txn_id": "demo_txn_normal_001",
    "step": 350,
    "type": "PAYMENT",
    "amount": 45.50,
    "nameOrig": "C1000000001",
    "oldbalanceOrg": 500.0,
    "nameDest": "M9000000001",
    "oldbalanceDest": 0.0
  }'
```

**Expected Output**:
```json
{
  "transaction_id": "demo_txn_normal_001",
  "fraud_score": 0.001126,
  "is_alert": false,
  "latency_ms": 7.82,
  "feature_fallback": true,
  "model_version": "FraudShield-XGBoost:1",
  "threshold": 0.99,
  "xgb_score": 0.001126,
  "gnn_score": null,
  "blend_score": 0.001126,
  "blend_used": false
}
```
*Note*: The fraud score is $0.0011$ (well below the $0.99$ alerting threshold). `is_alert` is `false`.

---

### Step 3: Submit a High-Risk Fraudulent Transaction
Submit an account draining transfer (\$950,000 transfer that empties the victim's account to an unseen counterparty):

```powershell
curl -X POST http://127.0.0.1:8000/score `
  -H "Content-Type: application/json" `
  -d '{
    "txn_id": "demo_txn_fraud_001",
    "step": 400,
    "type": "TRANSFER",
    "amount": 950000.0,
    "nameOrig": "C2000000002",
    "oldbalanceOrg": 950000.0,
    "nameDest": "C3000000003",
    "oldbalanceDest": 0.0
  }'
```

**Expected Output**:
```json
{
  "transaction_id": "demo_txn_fraud_001",
  "fraud_score": 0.999354,
  "is_alert": true,
  "latency_ms": 7.34,
  "feature_fallback": true,
  "model_version": "FraudShield-XGBoost:1",
  "threshold": 0.99,
  "xgb_score": 0.999354,
  "gnn_score": null,
  "blend_score": 0.999354,
  "blend_used": false
}
```
*Note*: The fraud score is $0.9994$, exceeding the $0.99$ threshold. `is_alert` is `true`. The transaction and score are persisted off-path to the database.

---

### Step 4: Display & Filter Alerts in Streamlit Dashboard
1. Open browser to: `http://localhost:8501`.
2. Navigate to **🚨 Alerts Triage** in the sidebar.
3. Observe the header KPIs:
   - Total Alerts: Incremented by 1.
   - Status: `New`.
4. In the alert table, find `demo_txn_fraud_001` with Blended Score $0.9994$.
5. Select `demo_txn_fraud_001` in the dropdown and click **🔍 Open Case Investigation**.

---

### Step 5: Investigate the Case with Tree SHAP Explanations
In the **🔬 Case Investigation** view:
1. Review Transaction Details:
   - Amount: `\$950,000.00`.
   - Origin: `C2000000002` (Balance before: \$950,000.00, balance after: \$0.00).
   - Destination: `C3000000003` (Balance before: \$0.00).
2. Inspect the **Top Contributing Features (Tree SHAP / pred_contribs)** chart:
   - `oldbalanceOrg` ($+3.3969$ log-odds shift): Indicates large account balance being drained.
   - `amount` ($+1.7178$ log-odds shift): High transaction amount.
   - `type_TRANSFER` ($+0.8321$ log-odds shift): High-risk payment mechanism.

---

### Step 6: Submit Analyst Review Feedback
1. Scroll down to **3. Analyst Decision & Feedback Submission**.
2. Select Disposition: **Confirmed Fraud**.
3. Analyst ID: `Senior Analyst Alice`.
4. Investigation Notes: `Account draining transfer to mule account; zero historical balance on destination.`
5. Click **💾 Submit Case Disposition**.
6. Observe status update:
   - Case Status updates to `Confirmed Fraud`.
   - The Audit History table records an immutable row with analyst, timestamp, previous status (`New`), and notes.

---

### Step 7: Verify Database Persistence Programmatically
Execute a quick verification script to inspect the SQLite database:

```powershell
.venv\Scripts\python.exe -c "
from src.feedback.service import get_case, get_labeled_cases

case = get_case('demo_txn_fraud_001')
print('Case Status:', case['status'])
print('Audit Trail Records:', len(case['audit_trail']))

labeled_df = get_labeled_cases()
print('Eligible Retraining Samples:')
print(labeled_df[labeled_df['transaction_id'] == 'demo_txn_fraud_001'])
"
```

**Expected Output**:
```
Case Status: Confirmed Fraud
Audit Trail Records: 1
Eligible Retraining Samples:
       transaction_id  label           labeled_at
0  demo_txn_fraud_001      1  2026-11-03 18:00:13
```

---

### Step 8: Retraining Dataset Eligibility & Zero-Leakage Gate Protection
How this feedback is consumed by model retraining:

1. **Querying Eligible Labels**:
   `get_labeled_cases()` queries the `labels` audit table and filters for `disposition IN ('Confirmed Fraud', 'False Positive')`. It converts:
   - `Confirmed Fraud` $\to$ Label `1`
   - `False Positive` $\to$ Label `0`
   - `Under Review` $\to$ Ignored (not ready for ground truth).
   If an alert was reviewed multiple times, the latest review disposition is used.

2. **Frozen Gate Set Leakage Defense**:
   When `prepare_retraining_dataset()` in `src/monitoring/retrain_trigger.py` runs, it checks whether any transaction in `get_labeled_cases()` belongs to the frozen evaluation gate set ($N=15,000$).
   - If an analyst label overlaps with the frozen gate set, an explicit `ValueError` is raised, aborting retraining.
   - Legitimate hold-out feedback is incorporated safely into candidate retraining data.

3. **Candidate Promotion Evaluation**:
   The retrained candidate model is evaluated on the frozen gate set against the champion model. It is promoted only if:
   - $\text{Candidate PR-AUC} \ge \text{Champion PR-AUC} + \text{MIN\_IMPROVEMENT\_PR\_AUC}$ ($+0.01$).
   - Recall is not degraded at the decision threshold.
   - The 95% bootstrap confidence interval difference is strictly positive.

