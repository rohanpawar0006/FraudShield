"""Transaction replay producer streaming hold-out test events into Kafka.

Replays hold-out transactions in strictly preserved chronological order with configurable rate pacing.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.config import settings
from src.common.logging import get_logger
from src.ingestion.streaming import StreamingChannel, get_streaming_channel

logger = get_logger(__name__)


def run_replay_producer(
    source_file: str = "data/processed/test_features.parquet",
    topic: str = settings.KAFKA_TOPIC_TRANSACTIONS,
    rate: float = 100.0,
    limit: Optional[int] = None,
    bootstrap_servers: Optional[str] = None,
    channel: Optional[StreamingChannel] = None,
    use_fallback: bool = False,
    loop: bool = False,
) -> int:
    """Stream transactions from hold-out split into Kafka topic in chronological time order.

    Args:
        source_file: Path to hold-out test set parquet file.
        topic: Destination Kafka topic name.
        rate: Target replay rate in events per second (0 or None for unrestricted speed).
        limit: Maximum number of transactions to stream (None for all).
        bootstrap_servers: Kafka broker address.
        channel: Optional pre-configured streaming channel.
        use_fallback: Force use of Docker-free in-memory fallback channel.

    Returns:
        Number of events successfully published.
    """
    path = Path(source_file)
    if not path.exists():
        raise FileNotFoundError(f"Hold-out dataset not found at {path}")

    # Safety check: Verify this is hold-out test data
    if "test" not in path.name.lower():
        logger.warning(
            "CRITICAL: Replay source '%s' is not explicitly labeled as test/hold-out data!",
            path.name,
        )

    logger.info("Loading hold-out transactions from %s...", path.as_posix())
    cols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "nameDest",
        "oldbalanceDest",
        "isFraud",
        "event_timestamp",
    ]
    df = pd.read_parquet(path, columns=cols)

    # Ensure chronological order
    df = df.sort_values(by=["step", "event_timestamp"]).reset_index(drop=True)
    if limit is not None and limit > 0:
        df = df.iloc[:limit]

    total_records = len(df)
    logger.info(
        "Prepared %d transactions for replay on topic '%s' (Target rate: %s eps)",
        total_records,
        topic,
        f"{rate:.1f}" if rate > 0 else "unlimited",
    )

    streaming_chan = channel or get_streaming_channel(
        bootstrap_servers=bootstrap_servers,
        force_fallback=use_fallback,
    )

    published_count = 0
    t_start = time.perf_counter()
    interval = (1.0 / rate) if rate > 0 else 0.0

    while True:
        for idx, row in df.iterrows():
            txn_id = f"txn_holdout_{int(row['step']):04d}_{idx:07d}"
            payload = {
                "txn_id": txn_id,
                "transaction_id": txn_id,
                "step": int(row["step"]),
                "type": str(row["type"]),
                "amount": float(row["amount"]),
                "nameOrig": str(row["nameOrig"]),
                "account_id": str(row["nameOrig"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "nameDest": str(row["nameDest"]),
                "dest_id": str(row["nameDest"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "event_timestamp": str(row["event_timestamp"]),
                "ground_truth_label": int(row["isFraud"]),
            }

            success = streaming_chan.publish(topic, payload)
            if success:
                published_count += 1

            if (idx + 1) % 1000 == 0 or (idx + 1) == total_records:
                elapsed = max(time.perf_counter() - t_start, 1e-4)
                current_rate = published_count / elapsed
                logger.info(
                    "Replayed %d/%d transactions (Throughput: %.1f eps)",
                    published_count,
                    total_records,
                    current_rate,
                )

            if interval > 0:
                time.sleep(interval)

        if not loop:
            break
        logger.info("Replay loop completed. Looping from start...")

    total_elapsed = max(time.perf_counter() - t_start, 1e-4)
    logger.info(
        "Replay finished: %d events emitted in %.2fs (Average rate: %.1f eps)",
        published_count,
        total_elapsed,
        published_count / total_elapsed,
    )
    return published_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FraudShield Hold-Out Transaction Replay Producer"
    )
    parser.add_argument(
        "--source-file",
        type=str,
        default="data/processed/test_features.parquet",
        help="Path to hold-out test set Parquet file",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=settings.KAFKA_TOPIC_TRANSACTIONS,
        help="Kafka destination topic",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Replay rate in events per second (0 for max speed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of events to stream",
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=settings.KAFKA_BOOTSTRAP_SERVERS,
        help="Kafka broker address",
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Force in-memory queue fallback without requiring Docker/Kafka",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously loop replay over hold-out transactions",
    )

    args = parser.parse_args()

    try:
        run_replay_producer(
            source_file=args.source_file,
            topic=args.topic,
            rate=args.rate,
            limit=args.limit,
            bootstrap_servers=args.bootstrap_servers,
            use_fallback=args.use_fallback,
            loop=args.loop,
        )
    except Exception as e:
        logger.error("Producer encountered fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

