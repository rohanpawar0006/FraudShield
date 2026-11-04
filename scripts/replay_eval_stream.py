"""Replay evaluation stream for Stage C3 integration verification.

Selects a fixed-seed sample of 200 transactions (10 fraud + 190 legitimate) from
data/processed/test_features.parquet in chronological order, saves the manifest
to docs/deployment/evidence/c3_eval_manifest.json, and publishes to Kafka at localhost:9092.
"""
# ruff: noqa: E402

import json
import logging
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.ingestion.streaming import get_streaming_channel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("replay_eval_stream")

def create_and_run_replay(
    parquet_path: str = "data/processed/test_features.parquet",
    bootstrap_servers: str = "localhost:9092",
    topic: str = "transactions",
    manifest_out: str = "docs/deployment/evidence/c3_eval_manifest.json",
    dry_run: bool = False,
):
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Source parquet file not found at {path}")

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
    logger.info("Reading test features from %s...", path)
    df = pd.read_parquet(path, columns=cols)

    fraud_df = df[df["isFraud"] == 1].sample(n=10, random_state=42)
    legit_df = df[df["isFraud"] == 0].sample(n=190, random_state=42)

    sample_df = pd.concat([fraud_df, legit_df], axis=0)
    sample_df = sample_df.sort_values(by=["step", "event_timestamp"]).reset_index(drop=True)

    manifest_records = []
    messages = []

    for idx, row in sample_df.iterrows():
        txn_id = f"c3_eval_{idx:04d}"
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
        messages.append(payload)
        manifest_records.append({
            "idx": idx,
            "txn_id": txn_id,
            "step": int(row["step"]),
            "type": str(row["type"]),
            "amount": float(row["amount"]),
            "nameOrig": str(row["nameOrig"]),
            "nameDest": str(row["nameDest"]),
            "ground_truth_label": int(row["isFraud"]),
        })

    manifest_path = Path(manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
    logger.info("Saved manifest (%d records, %d fraud) to %s", len(manifest_records), sum(r['ground_truth_label'] for r in manifest_records), manifest_path)

    if dry_run:
        logger.info("Dry run complete. Not publishing.")
        return len(messages)

    logger.info("Connecting to Kafka broker at %s...", bootstrap_servers)
    channel = get_streaming_channel(bootstrap_servers=bootstrap_servers)

    published = 0
    t0 = time.perf_counter()
    for msg in messages:
        ok = channel.publish(topic, msg)
        if ok:
            published += 1
        else:
            logger.error("Failed to publish message: %s", msg["txn_id"])

    elapsed = max(time.perf_counter() - t0, 1e-4)
    logger.info("Published %d/%d messages to topic '%s' in %.3fs (%.1f msg/s)", published, len(messages), topic, elapsed, published / elapsed)
    return published

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    create_and_run_replay(dry_run=dry_run)
