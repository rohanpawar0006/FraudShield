#!/usr/bin/env python3
"""Execute 3 trials of In-Process and TestClient HTTP benchmarks, plus real Uvicorn HTTP benchmark.

Evaluates on an idle machine with n_jobs=1:
1. In-process ScoringEngine.score() (Trials 1, 2, 3)
2. HTTP POST /score via TestClient (Trials 1, 2, 3)
3. HTTP POST /score via real Uvicorn server + separate client (TCP localhost:8000)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
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


def get_test_payloads(n: int = 2000, warmup: int = 100) -> List[Dict[str, Any]]:
    df = pd.read_parquet(
        TEST_DATA_PATH,
        columns=["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest"],
    ).iloc[: n + warmup]

    payloads = []
    for idx, row in df.iterrows():
        p = {
            "txn_id": f"bench_{idx:06d}",
            "step": int(row["step"]),
            "type": str(row["type"]),
            "amount": float(row["amount"]),
            "nameOrig": str(row["nameOrig"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "nameDest": str(row["nameDest"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
        }
        payloads.append(p)
    return payloads


def calc_stats(latencies: List[float], name: str, desc: str, conditions: Dict[str, Any], warmup: int) -> Dict[str, Any]:
    arr = np.array(latencies)
    return {
        "benchmark": name,
        "description": desc,
        "requests_evaluated": len(arr),
        "warmup_requests": warmup,
        "conditions": conditions,
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


def run_all_trials(n_requests: int = 2000, warmup: int = 100):
    print("=" * 78)
    print(f"FRAUDSHIELD RIGOROUS LATENCY BENCHMARK SUITE (N={n_requests:,}, Warmup={warmup})")
    print("=" * 78)

    # Clean & init benchmark DB
    object.__setattr__(settings, "SQLITE_DB_PATH", settings.BENCHMARK_DB_PATH)
    bench_db = Path(settings.BENCHMARK_DB_PATH)
    if bench_db.exists():
        try:
            bench_db.unlink()
        except Exception:
            pass
    db_manager._init_sqlite_schema(bench_db)

    payloads = get_test_payloads(n_requests, warmup)
    warmup_payloads = payloads[:warmup]
    timed_payloads = payloads[warmup: warmup + n_requests]
    typed_warmup = [TransactionPayload(**p) for p in warmup_payloads]
    typed_timed = [TransactionPayload(**p) for p in timed_payloads]

    # -------------------------------------------------------------------------
    # PART 1: IN-PROCESS BENCHMARK (3 TRIALS)
    # -------------------------------------------------------------------------
    print("\n>>> STARTING PART 1: IN-PROCESS BENCHMARK (ScoringEngine.score) <<<")
    engine = ScoringEngine()
    engine.load()

    in_process_results = []
    for trial in range(1, 4):
        print(f"\n[In-Process Trial {trial}/3] Warming up on {warmup} calls...")
        for p in typed_warmup:
            engine.score(p)

        print(f"[In-Process Trial {trial}/3] Executing {n_requests} timed calls...")
        lats = []
        for p in typed_timed:
            t0 = time.perf_counter()
            engine.score(p)
            lats.append((time.perf_counter() - t0) * 1000.0)

        stats = calc_stats(
            lats,
            f"in_process_scoring_trial_{trial}",
            "Direct ScoringEngine.score() invocation",
            {
                "booster_n_jobs": 1,
                "trial": trial,
                "online_store": "Feast SQLite (local file)",
                "runtime": f"Python {sys.version.split()[0]} on Windows host",
            },
            warmup,
        )
        in_process_results.append(stats)
        print(f"  Trial {trial} -> Mean: {stats['mean_ms']} ms | p50: {stats['p50_ms']} ms | p90: {stats['p90_ms']} ms | p95: {stats['p95_ms']} ms | p99: {stats['p99_ms']} ms")

    # -------------------------------------------------------------------------
    # PART 2: TESTCLIENT HTTP BENCHMARK (3 TRIALS)
    # -------------------------------------------------------------------------
    print("\n>>> STARTING PART 2: FASTAPI TESTCLIENT HTTP BENCHMARK <<<")
    testclient_results = []

    for trial in range(1, 4):
        with TestClient(app) as client:
            print(f"\n[TestClient Trial {trial}/3] Warming up on {warmup} requests...")
            for p in warmup_payloads:
                resp = client.post("/score", json=p)
                assert resp.status_code == 200

            print(f"[TestClient Trial {trial}/3] Executing {n_requests} timed requests...")
            lats = []
            for p in timed_payloads:
                t0 = time.perf_counter()
                resp = client.post("/score", json=p)
                t_el = (time.perf_counter() - t0) * 1000.0
                assert resp.status_code == 200
                lats.append(t_el)

        stats = calc_stats(
            lats,
            f"testclient_http_trial_{trial}",
            "FastAPI TestClient(app).post('/score') full stack",
            {
                "booster_n_jobs": 1,
                "trial": trial,
                "client": "Starlette TestClient",
                "persistence": "Asynchronous background SQLite queue",
            },
            warmup,
        )
        testclient_results.append(stats)
        print(f"  Trial {trial} -> Mean: {stats['mean_ms']} ms | p50: {stats['p50_ms']} ms | p90: {stats['p90_ms']} ms | p95: {stats['p95_ms']} ms | p99: {stats['p99_ms']} ms")

    # -------------------------------------------------------------------------
    # PART 3: REAL UVICORN SERVER BENCHMARK (TCP localhost:8000)
    # -------------------------------------------------------------------------
    print("\n>>> STARTING PART 3: REAL UVICORN SERVER BENCHMARK (Real TCP HTTP) <<<")
    server_port = 8008
    env = os.environ.copy()
    env["SQLITE_DB_PATH"] = settings.BENCHMARK_DB_PATH

    print(f"  -> Spawning Uvicorn process on http://127.0.0.1:{server_port}...")
    server_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.serving.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(server_port),
            "--log-level",
            "error",
            "--no-access-log",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    uvicorn_stats = None
    try:
        # Wait for server ready
        server_ready = False
        server_url = f"http://127.0.0.1:{server_port}"
        for _ in range(40):
            try:
                r = httpx.get(f"{server_url}/health", timeout=1.0)
                if r.status_code == 200 and r.json().get("model_loaded"):
                    server_ready = True
                    break
            except Exception:
                time.sleep(0.5)

        if not server_ready:
            print("  [ERROR] Uvicorn server failed to become healthy within 20 seconds.")
        else:
            print("  -> Uvicorn server is HEALTHY! Testing over real HTTP connection pool...")
            with httpx.Client(base_url=server_url, timeout=30.0) as http_client:
                print(f"  -> Warming up on {warmup} real HTTP requests...")
                for p in warmup_payloads:
                    r = http_client.post("/score", json=p)
                    assert r.status_code == 200

                print(f"  -> Executing {n_requests} timed real HTTP POST /score requests...")
                lats = []
                for p in timed_payloads:
                    t0 = time.perf_counter()
                    r = http_client.post("/score", json=p)
                    t_el = (time.perf_counter() - t0) * 1000.0
                    assert r.status_code == 200
                    lats.append(t_el)

                uvicorn_stats = calc_stats(
                    lats,
                    "real_uvicorn_http",
                    "Real uvicorn process scored via external HTTP client over TCP localhost",
                    {
                        "booster_n_jobs": 1,
                        "server": "uvicorn 0.128+",
                        "client": "httpx.Client connection pool",
                        "transport": "TCP loopback",
                    },
                    warmup,
                )
                print(f"  Real Uvicorn -> Mean: {uvicorn_stats['mean_ms']} ms | p50: {uvicorn_stats['p50_ms']} ms | p90: {uvicorn_stats['p90_ms']} ms | p95: {uvicorn_stats['p95_ms']} ms | p99: {uvicorn_stats['p99_ms']} ms")
    finally:
        print("  -> Terminating Uvicorn server process...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()

    # -------------------------------------------------------------------------
    # PART 4: PROFILING BREAKDOWN OF THE HTTP PATH
    # -------------------------------------------------------------------------
    print("\n>>> DETAILED PROFILE BREAKDOWN OF HTTP PATH (500 requests) <<<")
    t_pydantic, t_score_engine, t_response_ser = [], [], []

    engine = ScoringEngine()
    engine.load()
    for p in timed_payloads[:500]:
        # Stage 1: Pydantic parsing & validation
        t0 = time.perf_counter()
        parsed = TransactionPayload(**p)
        t_pydantic.append((time.perf_counter() - t0) * 1000.0)

        # Stage 2: ScoringEngine execution (Feast + preprocessing + XGBoost + GNN)
        t1 = time.perf_counter()
        scored = engine.score(parsed)
        t_score_engine.append((time.perf_counter() - t1) * 1000.0)

        # Stage 3: Pydantic response serialization
        t2 = time.perf_counter()
        _ = scored.model_dump()
        t_response_ser.append((time.perf_counter() - t2) * 1000.0)

    print(f"  1. Request Pydantic Validation : Mean {np.mean(t_pydantic):.2f} ms | p95 {np.percentile(t_pydantic, 95):.2f} ms")
    print(f"  2. ScoringEngine Pipeline      : Mean {np.mean(t_score_engine):.2f} ms | p95 {np.percentile(t_score_engine, 95):.2f} ms")
    print(f"  3. Response Model Serialization: Mean {np.mean(t_response_ser):.2f} ms | p95 {np.percentile(t_response_ser, 95):.2f} ms")

    # -------------------------------------------------------------------------
    # SAVE AUTHORITATIVE JSON ARTIFACTS
    # -------------------------------------------------------------------------
    # Use trial 2 (stable warm run) or median trial for official docs
    chosen_in_proc = in_process_results[1]
    chosen_http = testclient_results[1]

    # Clean benchmark key names
    chosen_in_proc["benchmark"] = "in_process_scoring_function"
    chosen_http["benchmark"] = "http_post_score_testclient"

    in_proc_file = RESULTS_DIR / "latency_in_process.json"
    with open(in_proc_file, "w", encoding="utf-8") as f:
        json.dump(chosen_in_proc, f, indent=2)

    http_file = RESULTS_DIR / "latency_http.json"
    with open(http_file, "w", encoding="utf-8") as f:
        json.dump(chosen_http, f, indent=2)

    all_trials_file = RESULTS_DIR / "latency_all_trials.json"
    with open(all_trials_file, "w", encoding="utf-8") as f:
        json.dump({
            "in_process_trials": in_process_results,
            "testclient_trials": testclient_results,
            "real_uvicorn": uvicorn_stats,
            "profiling_breakdown_ms": {
                "pydantic_validation_mean": round(float(np.mean(t_pydantic)), 2),
                "scoring_engine_mean": round(float(np.mean(t_score_engine)), 2),
                "response_serialization_mean": round(float(np.mean(t_response_ser)), 2),
            }
        }, f, indent=2)

    print("\n" + "=" * 78)
    print("ALL LATENCY TRIALS COMPLETE!")
    print(f"  Saved in-process results to {in_proc_file.relative_to(PROJECT_ROOT)}")
    print(f"  Saved HTTP results to {http_file.relative_to(PROJECT_ROOT)}")
    print(f"  Saved comprehensive trial log to {all_trials_file.relative_to(PROJECT_ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    run_all_trials(2000, 100)
