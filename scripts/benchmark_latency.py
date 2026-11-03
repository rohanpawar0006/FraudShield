#!/usr/bin/env python3
"""Execute isolated in-process and HTTP POST /score latency benchmarks.

Runs exactly two benchmarks over >= 2,000 hold-out transactions with n_jobs=1:
1. In-Process Benchmark: ScoringEngine.score_transaction()
2. HTTP POST /score Benchmark: FastAPI TestClient(app).post("/score")

Saves raw benchmark percentiles and runtime conditions to:
- docs/results/latency_in_process.json
- docs/results/latency_http.json
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.common.db import db_manager  # noqa: E402
from src.serving.app import app  # noqa: E402
from src.serving.schemas import TransactionPayload  # noqa: E402
from src.serving.scoring import ScoringEngine  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"


def run_benchmarks(sample_size: int = 2000, warmup_size: int = 100):
    print("=" * 70)
    print(f"RUNNING RIGOROUS LATENCY BENCHMARKS (N={sample_size}, Warmup={warmup_size})")
    print("=" * 70)

    # Route writes to dedicated benchmark database to preserve clean serving DB
    object.__setattr__(settings, "SQLITE_DB_PATH", settings.BENCHMARK_DB_PATH)
    bench_db = Path(settings.BENCHMARK_DB_PATH)
    if bench_db.exists():
        try:
            bench_db.unlink()
        except Exception:
            pass
    db_manager._init_sqlite_schema(bench_db)

    # 1. Load hold-out test data
    print(f"Loading hold-out test set from {TEST_DATA_PATH}...")
    df_test = pd.read_parquet(TEST_DATA_PATH).iloc[:sample_size + warmup_size].copy()

    # Convert to list of dict payloads
    payloads = []
    for idx, row in df_test.iterrows():
        p = {
            "txn_id": f"bench_{idx}",
            "step": int(row.get("step", 743)),
            "type": str(row.get("type", "TRANSFER")),
            "amount": float(row.get("amount", 1000.0)),
            "nameOrig": str(row.get("nameOrig", f"C_{idx}")),
            "oldbalanceOrg": float(row.get("oldbalanceOrg", 1000.0)),
            "newbalanceOrig": float(row.get("newbalanceOrig", 0.0)),
            "nameDest": str(row.get("nameDest", f"M_{idx}")),
            "oldbalanceDest": float(row.get("oldbalanceDest", 0.0)),
            "newbalanceDest": float(row.get("newbalanceDest", 1000.0)),
        }
        payloads.append(p)

    typed_payloads = [TransactionPayload(**p) for p in payloads]

    # =========================================================================
    # BENCHMARK 1: In-Process Scoring Function (ScoringEngine.score_transaction)
    # =========================================================================
    print("\n--- Benchmark 1: In-Process ScoringEngine (n_jobs=1) ---")
    engine = ScoringEngine()
    engine.load()

    # Warmup
    print(f"Warming up engine on {warmup_size} requests...")
    for p in typed_payloads[:warmup_size]:
        engine.score(p)

    # Timed run
    print(f"Executing {sample_size} in-process scoring calls...")
    in_process_latencies = []
    for p in typed_payloads[warmup_size:]:
        t0 = time.perf_counter()
        engine.score(p)
        t_el = (time.perf_counter() - t0) * 1000.0  # ms
        in_process_latencies.append(t_el)

    in_proc_arr = np.array(in_process_latencies)
    in_proc_summary = {
        "benchmark": "in_process_scoring_function",
        "description": "ScoringEngine.score_transaction() direct in-memory call",
        "requests_evaluated": len(in_proc_arr),
        "warmup_requests": warmup_size,
        "conditions": {
            "booster_n_jobs": 1,
            "runtime": "Python 3.13.14 on Windows host",
            "online_store": "Feast SQLite (local file)",
            "gnn_lookup": "Precomputed risk table in memory",
            "concurrency": "Sequential warm loop"
        },
        "mean_ms": round(float(np.mean(in_proc_arr)), 2),
        "std_ms": round(float(np.std(in_proc_arr)), 2),
        "min_ms": round(float(np.min(in_proc_arr)), 2),
        "p50_ms": round(float(np.percentile(in_proc_arr, 50)), 2),
        "p90_ms": round(float(np.percentile(in_proc_arr, 90)), 2),
        "p95_ms": round(float(np.percentile(in_proc_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(in_proc_arr, 99)), 2),
        "max_ms": round(float(np.max(in_proc_arr)), 2)
    }

    print("In-Process Results:")
    print(f"  Mean: {in_proc_summary['mean_ms']} ms | p50: {in_proc_summary['p50_ms']} ms | p95: {in_proc_summary['p95_ms']} ms | p99: {in_proc_summary['p99_ms']} ms")

    in_proc_file = RESULTS_DIR / "latency_in_process.json"
    with open(in_proc_file, "w", encoding="utf-8") as f:
        json.dump(in_proc_summary, f, indent=2)
    print(f"  -> Saved to {in_proc_file.relative_to(PROJECT_ROOT)}")

    # =========================================================================
    # BENCHMARK 2: HTTP POST /score via FastAPI TestClient
    # =========================================================================
    print("\n--- Benchmark 2: HTTP POST /score via TestClient (Full Stack) ---")
    client = TestClient(app)

    # Warmup
    print(f"Warming up TestClient on {warmup_size} HTTP requests...")
    for p in payloads[:warmup_size]:
        resp = client.post("/score", json=p)
        assert resp.status_code == 200

    # Timed run
    print(f"Executing {sample_size} HTTP POST /score requests...")
    http_latencies = []
    for p in payloads[warmup_size:]:
        t0 = time.perf_counter()
        resp = client.post("/score", json=p)
        t_el = (time.perf_counter() - t0) * 1000.0  # ms
        assert resp.status_code == 200
        http_latencies.append(t_el)

    http_arr = np.array(http_latencies)
    http_summary = {
        "benchmark": "http_post_score_testclient",
        "description": "Full HTTP request path (routing, Pydantic validation, serialization, scoring, async DB queue)",
        "requests_evaluated": len(http_arr),
        "warmup_requests": warmup_size,
        "conditions": {
            "booster_n_jobs": 1,
            "transport": "FastAPI TestClient (synchronous HTTP stack)",
            "runtime": "Python 3.13.14 on Windows host",
            "online_store": "Feast SQLite (local file)",
            "persistence": "Asynchronous background queue to SQLite"
        },
        "mean_ms": round(float(np.mean(http_arr)), 2),
        "std_ms": round(float(np.std(http_arr)), 2),
        "min_ms": round(float(np.min(http_arr)), 2),
        "p50_ms": round(float(np.percentile(http_arr, 50)), 2),
        "p90_ms": round(float(np.percentile(http_arr, 90)), 2),
        "p95_ms": round(float(np.percentile(http_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(http_arr, 99)), 2),
        "max_ms": round(float(np.max(http_arr)), 2)
    }

    print("HTTP POST /score Results:")
    print(f"  Mean: {http_summary['mean_ms']} ms | p50: {http_summary['p50_ms']} ms | p95: {http_summary['p95_ms']} ms | p99: {http_summary['p99_ms']} ms")

    http_file = RESULTS_DIR / "latency_http.json"
    with open(http_file, "w", encoding="utf-8") as f:
        json.dump(http_summary, f, indent=2)
    print(f"  -> Saved to {http_file.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 70)
    print("BENCHMARKS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks(sample_size=2000, warmup_size=100)
