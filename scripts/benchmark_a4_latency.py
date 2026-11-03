"""A4: Re-measure HTTP p50/p95/p99 latency over >= 2,000 requests with n_jobs=1."""

import sys
import time
from pathlib import Path
from typing import List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.common.db import db_manager  # noqa: E402
from src.serving.app import app  # noqa: E402


def run_benchmark(n_requests: int = 2000):
    print("=" * 70)
    print(f"BENCHMARKING POST /score WITH n_jobs=1 OVER {n_requests:,} REQUESTS")
    print("=" * 70)

    # Route writes to dedicated benchmark database to preserve clean serving DB
    object.__setattr__(settings, "SQLITE_DB_PATH", settings.BENCHMARK_DB_PATH)
    db_manager._init_sqlite_schema(Path(settings.BENCHMARK_DB_PATH))

    test_path = "data/processed/test_features.parquet"
    df = pd.read_parquet(
        test_path,
        columns=["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest"],
    ).iloc[:n_requests]

    latencies_http: List[float] = []

    with TestClient(app) as client:
        # Warmup
        warmup = {
            "txn_id": "warmup_01",
            "step": 380,
            "type": "PAYMENT",
            "amount": 100.0,
            "nameOrig": "C12345",
            "oldbalanceOrg": 500.0,
            "nameDest": "M67890",
            "oldbalanceDest": 0.0,
        }
        for _ in range(50):
            client.post("/score", json=warmup)

        t_start = time.perf_counter()
        for idx, row in df.iterrows():
            payload = {
                "txn_id": f"bench_{idx:06d}",
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
            lat = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200
            latencies_http.append(lat)

        total_wall = time.perf_counter() - t_start

    l_arr = np.array(latencies_http)
    p50 = float(np.percentile(l_arr, 50))
    p90 = float(np.percentile(l_arr, 90))
    p95 = float(np.percentile(l_arr, 95))
    p99 = float(np.percentile(l_arr, 99))
    mean = float(np.mean(l_arr))
    throughput = len(l_arr) / total_wall

    print(f"Total Requests: {len(l_arr):,}")
    print(f"Throughput: {throughput:.2f} req/sec")
    print(f"Mean Latency: {mean:.3f} ms")
    print(f"p50 Latency:  {p50:.3f} ms")
    print(f"p90 Latency:  {p90:.3f} ms")
    print(f"p95 Latency:  {p95:.3f} ms")
    print(f"p99 Latency:  {p99:.3f} ms")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark(2000)
