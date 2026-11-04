"""Streaming transaction scoring consumer with Dead-Letter Queue (DLQ) routing.

Reads transaction events from Kafka, performs feature lookup and XGBoost inference,
persists records to Postgres/fallback database, and safely routes malformed events to DLQ.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.common.config import settings
from src.common.db import db_manager
from src.common.logging import get_logger
from src.ingestion.streaming import StreamingChannel, get_streaming_channel
from src.serving.metrics import metrics_tracker
from src.serving.schemas import TransactionPayload
from src.serving.scoring import scoring_engine

logger = get_logger(__name__)


class ScoringConsumer:
    """Consumes transactions from Kafka, scores them, and persists outcomes."""

    def __init__(
        self,
        topic: str = settings.KAFKA_TOPIC_TRANSACTIONS,
        dlq_topic: str = settings.KAFKA_TOPIC_DEAD_LETTER,
        scores_topic: str = settings.KAFKA_TOPIC_SCORES,
        bootstrap_servers: Optional[str] = None,
        channel: Optional[StreamingChannel] = None,
        use_fallback: bool = False,
    ) -> None:
        self.topic = topic
        self.dlq_topic = dlq_topic
        self.scores_topic = scores_topic
        self.channel = channel or get_streaming_channel(
            bootstrap_servers=bootstrap_servers,
            force_fallback=use_fallback,
        )

        self.processed_count: int = 0
        self.dlq_count: int = 0
        self.error_count: int = 0
        self.latencies: List[float] = []

    def handle_message(self, message: Dict[str, Any]) -> bool:
        """Process a single event message with resilient DLQ routing on malformed payloads.

        Returns:
            True if scored successfully, False if routed to DLQ or encountered error.
        """
        # 1. Pydantic schema validation
        try:
            payload = TransactionPayload.model_validate(message)
        except (ValidationError, Exception) as val_err:
            self._route_to_dlq(
                raw_message=message,
                reason=f"Payload validation failed: {str(val_err)}",
            )
            return False

        # 2. Score transaction
        try:
            score_resp = scoring_engine.score(payload)
            self.latencies.append(score_resp.latency_ms)
            metrics_tracker.record_request(
                latency_ms=score_resp.latency_ms,
                feature_fallback=score_resp.feature_fallback,
            )
        except Exception as score_err:
            self.error_count += 1
            metrics_tracker.record_error()
            self._route_to_dlq(
                raw_message=message,
                reason=f"Model scoring runtime failure: {str(score_err)}",
            )
            return False

        # 3. Off-path / background persistence to database
        db_manager.save_transaction_and_score(
            txn_id=payload.txn_id,
            account_id=payload.nameOrig,
            dest_id=payload.nameDest,
            amount=payload.amount,
            txn_type=payload.type,
            ts=None,
            raw_payload=message,
            xgb_score=score_resp.xgb_score,
            gnn_score=score_resp.gnn_score,
            blended_score=score_resp.fraud_score,
            model_version=score_resp.model_version,
            latency_ms=score_resp.latency_ms,
        )

        # 4. Emit score to scores topic
        score_payload = score_resp.model_dump()
        self.channel.publish(self.scores_topic, score_payload)

        self.processed_count += 1
        return True

    def _route_to_dlq(self, raw_message: Any, reason: str) -> None:
        """Route unparseable or failed messages to DLQ topic with error diagnostics."""
        self.dlq_count += 1
        dlq_payload = {
            "error_reason": reason,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "raw_payload": raw_message,
        }
        logger.warning(
            "Routing message to DLQ topic '%s': %s",
            self.dlq_topic,
            reason,
        )
        self.channel.publish(self.dlq_topic, dlq_payload)

    def run(
        self, limit: Optional[int] = None, timeout_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """Consume and process incoming messages up to optional limit."""
        logger.info(
            "Starting scoring consumer on topic '%s' (DLQ: '%s')...",
            self.topic,
            self.dlq_topic,
        )
        scoring_engine.load()

        t_start = time.perf_counter()

        for msg in self.channel.consume(self.topic, timeout_sec=timeout_sec):
            self.handle_message(msg)

            if (
                self.processed_count > 0
                and self.processed_count % 1000 == 0
            ):
                elapsed = max(time.perf_counter() - t_start, 1e-4)
                logger.info(
                    "Consumer progress: %d processed, %d DLQ, %d errors (%.1f eps)",
                    self.processed_count,
                    self.dlq_count,
                    self.error_count,
                    self.processed_count / elapsed,
                )

            if limit is not None and (self.processed_count + self.dlq_count) >= limit:
                break

        total_elapsed = max(time.perf_counter() - t_start, 1e-4)
        throughput = self.processed_count / total_elapsed

        summary = {
            "processed_count": self.processed_count,
            "dlq_count": self.dlq_count,
            "error_count": self.error_count,
            "elapsed_sec": round(total_elapsed, 3),
            "throughput_eps": round(throughput, 2),
        }
        logger.info("Consumer completed session: %s", summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FraudShield Real-Time Scoring Consumer"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=settings.KAFKA_TOPIC_TRANSACTIONS,
        help="Kafka source transactions topic",
    )
    parser.add_argument(
        "--dlq-topic",
        type=str,
        default=settings.KAFKA_TOPIC_DEAD_LETTER,
        help="Kafka Dead-Letter Queue topic",
    )
    parser.add_argument(
        "--scores-topic",
        type=str,
        default=settings.KAFKA_TOPIC_SCORES,
        help="Kafka output scores topic",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum events to consume before terminating",
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
        "--timeout-sec",
        type=float,
        default=None,
        help="Timeout in seconds for consumer idle exit (None to wait indefinitely)",
    )

    args = parser.parse_args()

    consumer = ScoringConsumer(
        topic=args.topic,
        dlq_topic=args.dlq_topic,
        scores_topic=args.scores_topic,
        bootstrap_servers=args.bootstrap_servers,
        use_fallback=args.use_fallback,
    )

    try:
        consumer.run(limit=args.limit, timeout_sec=args.timeout_sec)
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    except Exception as e:
        logger.error("Consumer fatal failure: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

