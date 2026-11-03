"""Smoke and integration benchmark script for Phase 4 streaming scoring pipeline.

Replays >= 10,000 transactions from hold-out dataset through producer and consumer,
measures end-to-end latency percentiles (p50, p95), tests DLQ handling, and logs throughput.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.common.db import db_manager  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.ingestion.consumer import ScoringConsumer  # noqa: E402
from src.ingestion.producer import run_replay_producer  # noqa: E402
from src.ingestion.streaming import MemoryStreamingChannel  # noqa: E402
from src.serving.scoring import scoring_engine  # noqa: E402

logger = get_logger(__name__)


def run_smoke_test(
    sample_count: int = 10000,
    inject_malformed: int = 25,
    source_file: str = "data/processed/test_features.parquet",
    use_benchmark_db: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end replay smoke test over >= 10,000 hold-out events.

    Args:
        sample_count: Number of valid hold-out transactions to stream.
        inject_malformed: Number of corrupted payloads to inject for DLQ validation.
        source_file: Path to hold-out test set parquet.
        use_benchmark_db: If True, isolates writes to benchmark database path.

    Returns:
        Summary metrics dictionary.
    """
    if use_benchmark_db:
        object.__setattr__(settings, "SQLITE_DB_PATH", settings.BENCHMARK_DB_PATH)
        db_manager._init_sqlite_schema(Path(settings.BENCHMARK_DB_PATH))
    print("\n" + "=" * 78)
    print(f" FRAUDSHIELD STREAMING SMOKE TEST: {sample_count:,} TRANSACTIONS")
    print("=" * 78)

    # 1. Warm up scoring engine in memory
    print("\n[1/4] Warming up model, Feast store, and preprocessors in memory...")
    scoring_engine.load()

    # 2. Set up high-throughput streaming channel
    channel = MemoryStreamingChannel()
    topic = "smoke-transactions"
    dlq_topic = "smoke-transactions-dlq"
    scores_topic = "smoke-scores"

    # 3. Stream hold-out transactions into topic
    print(f"\n[2/4] Streaming {sample_count:,} hold-out transactions from {source_file}...")
    t_prod_start = time.perf_counter()
    published = run_replay_producer(
        source_file=source_file,
        topic=topic,
        rate=0.0,  # Max speed for benchmarking
        limit=sample_count,
        channel=channel,
    )
    prod_duration = time.perf_counter() - t_prod_start
    print(f"  -> Replayed {published:,} transactions in {prod_duration:.2f}s ({published/prod_duration:,.1f} eps)")

    # 4. Inject intentional malformed payloads to exercise DLQ
    if inject_malformed > 0:
        print(f"\n[3/4] Injecting {inject_malformed} malformed events for DLQ verification...")
        for i in range(inject_malformed):
            if i % 2 == 0:
                channel.publish(topic, {"corrupted": "bad_json", "index": i})
            else:
                channel.publish(
                    topic,
                    {
                        "txn_id": f"dlq_neg_amt_{i}",
                        "type": "TRANSFER",
                        "amount": -999.0,  # Negative amount rejected by schema
                        "nameOrig": "C_TEST",
                        "nameDest": "M_TEST",
                    },
                )
        print(f"  -> Injected {inject_malformed} malformed messages.")

    # 5. Consume, validate, score, and persist
    total_expected = sample_count + inject_malformed
    print(f"\n[4/4] Consumer scoring and persisting {total_expected:,} messages...")
    consumer = ScoringConsumer(
        topic=topic,
        dlq_topic=dlq_topic,
        scores_topic=scores_topic,
        channel=channel,
    )

    t_cons_start = time.perf_counter()
    consumer.run(limit=total_expected, timeout_sec=1.0)
    cons_duration = time.perf_counter() - t_cons_start

    # 6. Analyze Latencies
    latencies = np.array(consumer.latencies) if consumer.latencies else np.array([0.0])
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    p_mean = float(np.mean(latencies))
    p_min = float(np.min(latencies))
    p_max = float(np.max(latencies))
    throughput = consumer.processed_count / max(cons_duration, 1e-4)

    # 7. Print Results Table
    print("\n" + "=" * 78)
    print(" END-TO-END STREAMING BENCHMARK RESULTS")
    print("=" * 78)
    print(f"  Total Processed Valid  : {consumer.processed_count:,}")
    print(f"  Total Routed to DLQ    : {consumer.dlq_count:,}")
    print(f"  Total Errors (Crashes) : {consumer.error_count:,}")
    print(f"  Total Time Elapsed     : {cons_duration:.2f} seconds")
    print(f"  End-to-End Throughput  : {throughput:,.1f} events/sec")
    print("-" * 78)
    print(" LATENCY DISTRIBUTION (ms per transaction):")
    print(f"  Min latency            : {p_min:.3f} ms")
    print(f"  Mean latency           : {p_mean:.3f} ms")
    print(f"  p50 latency (median)   : {p50:.3f} ms")
    print(f"  p90 latency            : {p90:.3f} ms")
    print(f"  p95 latency            : {p95:.3f} ms")
    print(f"  p99 latency            : {p99:.3f} ms")
    print(f"  Max latency            : {p_max:.3f} ms")
    print("=" * 78)

    # Acceptance verification: p50 and p95 < 100 ms
    target_threshold_ms = 100.0
    passed = (
        p50 < target_threshold_ms
        and p95 < target_threshold_ms
        and consumer.processed_count >= sample_count
        and consumer.dlq_count == inject_malformed
        and consumer.error_count == 0
    )

    if passed:
        print(f"\n>>> ACCEPTANCE CRITERIA PASSED: p50 ({p50:.2f}ms) & p95 ({p95:.2f}ms) are both < {target_threshold_ms:.0f}ms! <<<")
    else:
        print("\n>>> ACCEPTANCE CRITERIA FAILED: Latency target not satisfied or count mismatch! <<<")

    return {
        "processed_count": consumer.processed_count,
        "dlq_count": consumer.dlq_count,
        "error_count": consumer.error_count,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "throughput_eps": throughput,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FraudShield Replay Smoke Test & Latency Profiler"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of transactions to replay (default: 10000)",
    )
    parser.add_argument(
        "--malformed",
        type=int,
        default=25,
        help="Number of malformed messages to inject for DLQ test",
    )
    args = parser.parse_args()

    results = run_smoke_test(
        sample_count=args.samples,
        inject_malformed=args.malformed,
    )

    if not results["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
