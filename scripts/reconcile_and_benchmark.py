"""Reconciliation and profiling script addressing Part A requirements (A2, A3, A4).

1. A2: Benchmarks >= 2,000 hold-out transactions through the real HTTP POST /score path (FastAPI TestClient)
2. A3: Warm service latency re-profiling across stages (Feast, DataFrame assembly, Preprocessing, XGBoost)
3. A4: Served-score sanity check comparing serving scores against offline labels & predictions.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.serving.app import app  # noqa: E402
from src.serving.schemas import TransactionPayload  # noqa: E402
from src.serving.scoring import ALL_FEATURE_COLUMNS, scoring_engine  # noqa: E402


def run_a2_http_benchmark(n_requests: int = 2000) -> Dict[str, float]:
    """Benchmark the real HTTP POST /score path using FastAPI TestClient."""
    print("\n" + "=" * 78)
    print(f" A2: HTTP-PATH LATENCY BENCHMARK ({n_requests:,} REQUESTS VIA POST /score)")
    print("=" * 78)

    test_path = "data/processed/test_features.parquet"
    df = pd.read_parquet(test_path, columns=["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest"]).iloc[:n_requests]

    latencies_http: List[float] = []

    with TestClient(app) as client:
        # Warmup
        warmup_payload = {
            "txn_id": "warmup_001",
            "step": 380,
            "type": "PAYMENT",
            "amount": 100.0,
            "nameOrig": "C12345",
            "oldbalanceOrg": 500.0,
            "nameDest": "M67890",
            "oldbalanceDest": 0.0,
        }
        for _ in range(50):
            client.post("/score", json=warmup_payload)

        # Timed benchmark
        t_start = time.perf_counter()
        for idx, row in df.iterrows():
            payload = {
                "txn_id": f"http_bench_{idx:06d}",
                "step": int(row["step"]),
                "type": str(row["type"]),
                "amount": float(row["amount"]),
                "nameOrig": str(row["nameOrig"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "nameDest": str(row["nameDest"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
            }
            t0 = time.perf_counter()
            resp = client.post("/score", json=payload)
            t_client = (time.perf_counter() - t0) * 1000.0
            latencies_http.append(t_client)
            assert resp.status_code == 200

        total_elapsed = time.perf_counter() - t_start

    lats = np.array(latencies_http)
    p50 = float(np.percentile(lats, 50))
    p90 = float(np.percentile(lats, 90))
    p95 = float(np.percentile(lats, 95))
    p99 = float(np.percentile(lats, 99))
    mean = float(np.mean(lats))
    throughput = n_requests / total_elapsed

    print(f"  HTTP Requests Processed : {n_requests:,}")
    print(f"  Total Wall Time         : {total_elapsed:.2f} s")
    print(f"  HTTP Throughput         : {throughput:,.1f} req/sec")
    print(f"  HTTP Mean Latency       : {mean:.3f} ms")
    print(f"  HTTP p50 (Median)       : {p50:.3f} ms")
    print(f"  HTTP p90                : {p90:.3f} ms")
    print(f"  HTTP p95                : {p95:.3f} ms")
    print(f"  HTTP p99                : {p99:.3f} ms")
    print("=" * 78)

    return {"http_p50": p50, "http_p95": p95, "http_mean": mean, "http_throughput": throughput}


def run_a3_profiling_breakdown(n_requests: int = 1000) -> Dict[str, Dict[str, float]]:
    """Re-profile warm scoring stages on >= 1,000 requests measuring both mean and p95."""
    print("\n" + "=" * 78)
    print(f" A3: WARM SERVICE PROFILING BREAKDOWN ({n_requests:,} REQUESTS)")
    print("=" * 78)

    scoring_engine.load()

    test_path = "data/processed/test_features.parquet"
    df = pd.read_parquet(test_path, columns=["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest"]).iloc[:n_requests]

    # Warmup
    dummy = TransactionPayload(
        txn_id="warm", step=380, type="TRANSFER", amount=1000.0,
        nameOrig="C12345", oldbalanceOrg=5000.0, nameDest="C67890", oldbalanceDest=0.0
    )
    for _ in range(200):
        scoring_engine.score(dummy)

    feast_times: List[float] = []
    df_times: List[float] = []
    preproc_times: List[float] = []
    xgb_times: List[float] = []
    total_measured_times: List[float] = []

    for _, row in df.iterrows():
        txn = TransactionPayload(
            txn_id="prof_01",
            step=int(row["step"]),
            type=str(row["type"]),
            amount=float(row["amount"]),
            nameOrig=str(row["nameOrig"]),
            oldbalanceOrg=float(row["oldbalanceOrg"]),
            nameDest=str(row["nameDest"]),
            oldbalanceDest=float(row["oldbalanceDest"]),
        )

        t_start = time.perf_counter()

        # Stage 1: Feast lookup
        t0 = time.perf_counter()
        features, _ = scoring_engine.fetch_features(txn)
        t1 = time.perf_counter()

        # Stage 2: Feature record & DataFrame assembly
        record = {
            "type": txn.type,
            "amount": float(txn.amount),
            "oldbalanceOrg": float(txn.oldbalanceOrg),
            "oldbalanceDest": float(txn.oldbalanceDest),
            "orig_txn_count_1h": int(features["orig_txn_count_1h"]),
            "orig_txn_count_24h": int(features["orig_txn_count_24h"]),
            "orig_amount_vs_mean": float(features["orig_amount_vs_mean"]),
            "orig_time_since_last_txn": float(features["orig_time_since_last_txn"]),
            "orig_has_prev_txn": int(features["orig_has_prev_txn"]),
            "dest_txn_count_24h": int(features["dest_txn_count_24h"]),
            "dest_has_prev_txn": int(features["dest_has_prev_txn"]),
        }
        df_row = pd.DataFrame([record], columns=ALL_FEATURE_COLUMNS)
        t2 = time.perf_counter()

        # Stage 3: ColumnTransformer preprocessing
        x_proc = scoring_engine.preprocessor.transform(df_row)
        t3 = time.perf_counter()

        # Stage 4: XGBoost prediction
        _ = scoring_engine.model.predict_proba(x_proc)[0, 1]
        t4 = time.perf_counter()

        t_end = time.perf_counter()

        feast_times.append((t1 - t0) * 1000.0)
        df_times.append((t2 - t1) * 1000.0)
        preproc_times.append((t3 - t2) * 1000.0)
        xgb_times.append((t4 - t3) * 1000.0)
        total_measured_times.append((t_end - t_start) * 1000.0)

    stages = {
        "Feast Lookup": np.array(feast_times),
        "DataFrame Assembly": np.array(df_times),
        "Preprocessor Transform": np.array(preproc_times),
        "XGBoost Predict": np.array(xgb_times),
        "Total (Sum of Stages)": np.array(feast_times) + np.array(df_times) + np.array(preproc_times) + np.array(xgb_times),
        "Total (Direct Measured)": np.array(total_measured_times),
    }

    results: Dict[str, Dict[str, float]] = {}
    print(f"{'Stage Name':<26} | {'Mean (ms)':<10} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'% of Total':<10}")
    print("-" * 78)
    sum_mean = float(np.mean(stages["Total (Sum of Stages)"]))
    for name, arr in stages.items():
        m = float(np.mean(arr))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        pct = (m / sum_mean * 100.0) if "Total" not in name else 100.0
        results[name] = {"mean": m, "p50": p50, "p95": p95, "pct": pct}
        print(f"{name:<26} | {m:>8.3f} ms | {p50:>8.3f} ms | {p95:>8.3f} ms | {pct:>8.1f} %")
    print("=" * 78)

    return results


def run_a4_served_score_check(n_sample: int = 25000) -> Dict[str, float]:
    """Evaluate served scores against offline test labels."""
    print("\n" + "=" * 78)
    print(f" A4: SERVED-SCORE SANITY CHECK OVER {n_sample:,} TEST TRANSACTIONS")
    print("=" * 78)

    test_path = "data/processed/test_features.parquet"
    df = pd.read_parquet(test_path).iloc[:n_sample]
    y_true = df["isFraud"].to_numpy().astype(int)
    n_pos = int(y_true.sum())
    print(f"  Evaluating {len(df):,} transactions (Fraud cases: {n_pos}, {n_pos/len(df)*100:.3f}%)")

    scoring_engine.load()

    # 1. Served scores via scoring_engine (online feature retrieval + fallback)
    served_scores: List[float] = []
    alerts: int = 0
    for _, row in df.iterrows():
        txn = TransactionPayload(
            txn_id="check_txn",
            step=int(row["step"]),
            type=str(row["type"]),
            amount=float(row["amount"]),
            nameOrig=str(row["nameOrig"]),
            oldbalanceOrg=float(row["oldbalanceOrg"]),
            nameDest=str(row["nameDest"]),
            oldbalanceDest=float(row["oldbalanceDest"]),
        )
        resp = scoring_engine.score(txn)
        served_scores.append(resp.fraud_score)
        if resp.is_alert:
            alerts += 1

    served_probs = np.array(served_scores)

    # 2. Offline predictions on exact same rows using continuous features in parquet
    X_offline = df[ALL_FEATURE_COLUMNS]
    X_offline_proc = scoring_engine.preprocessor.transform(X_offline)
    offline_probs = scoring_engine.model.predict_proba(X_offline_proc)[:, 1]

    # Metrics on served scores
    prec_s, rec_s, _ = precision_recall_curve(y_true, served_probs)
    pr_auc_served = float(auc(rec_s, prec_s))
    roc_auc_served = float(roc_auc_score(y_true, served_probs))

    # Metrics on offline continuous feature scores
    prec_o, rec_o, _ = precision_recall_curve(y_true, offline_probs)
    pr_auc_offline = float(auc(rec_o, prec_o))

    # Alert threshold t=0.99 metrics
    t = settings.DECISION_THRESHOLD
    y_pred_served = (served_probs >= t).astype(int)
    tp_s = int(np.sum((y_pred_served == 1) & (y_true == 1)))
    fp_s = int(np.sum((y_pred_served == 1) & (y_true == 0)))
    fn_s = int(np.sum((y_pred_served == 0) & (y_true == 1)))

    prec_at_t = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
    rec_at_t = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0

    print(f"  Served PR-AUC (with Feast fallback) : {pr_auc_served:.4f}")
    print(f"  Offline PR-AUC (continuous state)   : {pr_auc_offline:.4f}")
    print(f"  PR-AUC Difference (Delta)           : {pr_auc_served - pr_auc_offline:+.4f}")
    print(f"  Served ROC-AUC                      : {roc_auc_served:.4f}")
    print(f"  Total Alerts Flagged (t={t})       : {alerts}")
    print(f"  Precision at threshold t={t}        : {prec_at_t*100:.2f}% (TP={tp_s}, FP={fp_s})")
    print(f"  Recall at threshold t={t}           : {rec_at_t*100:.2f}% (TP={tp_s}, FN={fn_s})")
    print("=" * 78)

    return {
        "pr_auc_served": pr_auc_served,
        "pr_auc_offline": pr_auc_offline,
        "delta": pr_auc_served - pr_auc_offline,
        "alerts": alerts,
        "precision_at_t": prec_at_t,
        "recall_at_t": rec_at_t,
    }


if __name__ == "__main__":
    run_a2_http_benchmark(2000)
    run_a3_profiling_breakdown(1000)
    run_a4_served_score_check(25000)
