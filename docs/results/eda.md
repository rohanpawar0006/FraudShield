# Phase 1: Exploratory Data Analysis & Leakage Review

## 1. Executive Summary
- **Dataset:** PaySim Synthetic Financial Transactions
- **Total Transactions:** 6,362,620
- **Legitimate Transactions (0):** 6,354,407 (99.8709%)
- **Fraudulent Transactions (1):** 8,213 (0.1291%)
- **Imbalance Ratio:** ~1 fraud per 774 legitimate transactions (< 1% fraud condition verified).
- **Time Window:** 743 discrete hourly steps (~31 days of financial activity).

---

## 2. Transaction Type Breakdown & Fraud Vulnerability
In PaySim, fraud is strictly concentrated in outbound transfer and cash-out operations:

| Transaction Type | Total Count | % of Traffic | Fraud Count | Fraud Rate (%) |
|---|---|---|---|---|
| **CASH_IN** | 1,399,284 | 21.99% | 0 | 0.0000% |
| **CASH_OUT** | 2,237,500 | 35.17% | 4,116 | 0.1840% |
| **DEBIT** | 41,432 | 0.65% | 0 | 0.0000% |
| **PAYMENT** | 2,151,495 | 33.81% | 0 | 0.0000% |
| **TRANSFER** | 532,909 | 8.38% | 4,097 | 0.7688% |

> **Key Insight:** `PAYMENT`, `CASH_IN`, and `DEBIT` contain exactly 0 fraudulent events across the entire 6.36M dataset. Fraud operates as a two-stage laundering attack: first moving capital via `TRANSFER` (0.7688% fraud rate) and immediately liquidating via `CASH_OUT` (0.1840% fraud rate).

---

## 3. Amount Distribution Comparison
Fraudulent amounts are significantly larger than legitimate transactions:

| Metric | Overall Amount ($) | Legitimate (0) ($) | Fraudulent (1) ($) |
|---|---|---|---|
| **Mean** | $179,861.90 | $178,197.04 | $1,467,967.30 |
| **Std Dev** | $603,858.23 | $596,236.98 | $2,404,252.95 |
| **Min** | $0.00 | $0.01 | $0.00 |
| **Median (50%)** | $74,871.94 | $74,684.72 | $441,423.44 |
| **75th Percentile** | $208,721.48 | $208,364.76 | $1,517,771.48 |
| **Max** | $92,445,516.64 | $92,445,516.64 | $10,000,000.00 |

---

## 4. Leakage Review & Dropped Columns

### Analysis of Balance Columns
1. **`newbalanceOrig` (Post-Transaction Origin Balance):**
   - In fraudulent transactions, the origin account is completely drained in **98.05%** of cases (`newbalanceOrig == 0.0`).
   - In production, when a transaction authorization request arrives at `/score`, the transaction has not yet settled, meaning `newbalanceOrig` does not exist at inference time.
   - Including `newbalanceOrig` causes severe target leakage and produces artificially inflated offline metrics that collapse online.
2. **`newbalanceDest` (Post-Transaction Destination Balance):**
   - Similar to `newbalanceOrig`, this reflects the post-settlement balance of the recipient, which is unavailable at real-time pre-authorization scoring.
3. **`isFlaggedFraud` (Heuristic Business Rule Flag):**
   - A static simulator rule flagging `TRANSFER` transactions > $200,000. It flagged only 16 out of 8213 frauds and 0 legitimate records. Using it as a feature leaks the synthetic label rule.

### Dropped Columns Specification
| Column Name | Action | Reason |
|---|---|---|
| `newbalanceOrig` | **DROPPED** | Target leakage: Post-transaction origin balance; 98.05% zero in fraud |
| `newbalanceDest` | **DROPPED** | Target leakage: Post-transaction destination balance |
| `isFlaggedFraud` | **DROPPED** | Target leakage / Collinearity: Naive heuristic flag (> 200k transfer) |
| `nameOrig` | **EXCLUDED FROM TABULAR MATRIX** | Retained in parquet as entity ID for Feast online store and Graph edges |
| `nameDest` | **EXCLUDED FROM TABULAR MATRIX** | Retained in parquet as entity ID for Feast online store and Graph edges |

---

## 5. Time-Based Train / Val / Test Splits
Per `Rules.md`, random splitting is strictly forbidden. The splits are ordered chronologically by `step` (hourly timestamps):

| Split | Step Range (Hours) | Approximate Time Window | Total Records | % of Dataset | Fraud Count | Fraud Rate (%) |
|---|---|---|---|---|---|---|
| **Train** | `step` 1 to 323 | Days 1 to 13.5 | 4,463,587 | 70.15% | 3,643 | 0.0816% |
| **Validation** | `step` 324 to 378 | Days 13.5 to 15.8 | 980,416 | 15.41% | 564 | 0.0575% |
| **Test (Hold-Out)** | `step` 379 to 743 | Days 15.8 to 31.0 | 918,617 | 14.44% | 4,006 | 0.4361% |
| **Total** | `step` 1 to 743 | Full 31 Days | **6,362,620** | **100.00%** | **8,213** | **0.1291%** |

### Split Verification Guarantees
- `train_df['step'].max() == 323 < val_df['step'].min() == 324`: **PASSED (Zero Overlap)**
- `val_df['step'].max() == 378 < test_df['step'].min() == 379`: **PASSED (Zero Overlap)**
- Parquet Artifacts:
  - `data/processed/train.parquet`
  - `data/processed/val.parquet`
  - `data/processed/test.parquet`
