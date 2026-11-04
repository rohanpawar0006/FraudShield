"""Streaming abstraction supporting Kafka and in-memory queue fallback for Docker-free environments."""

import json
import queue
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional

from src.common.config import settings
from src.common.logging import get_logger

logger = get_logger(__name__)


class StreamingChannel(ABC):
    """Abstract interface for producing and consuming event messages."""

    @abstractmethod
    def publish(self, topic: str, message: Dict[str, Any]) -> bool:
        """Publish an event to the specified topic."""
        pass

    @abstractmethod
    def consume(
        self, topic: str, timeout_sec: float = 1.0
    ) -> Generator[Dict[str, Any], None, None]:
        """Consume events from the specified topic."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close producer/consumer resources."""
        pass


class MemoryStreamingChannel(StreamingChannel):
    """Thread-safe in-memory message channel for local testing without Docker/Kafka."""

    _instances: Dict[str, "MemoryStreamingChannel"] = {}
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.topics: Dict[str, queue.Queue] = {}
        self._running = True

    def _get_queue(self, topic: str) -> queue.Queue:
        if topic not in self.topics:
            self.topics[topic] = queue.Queue()
        return self.topics[topic]

    def publish(self, topic: str, message: Dict[str, Any]) -> bool:
        """Enqueue message to in-memory topic queue."""
        q = self._get_queue(topic)
        q.put(message)
        return True

    def consume(
        self, topic: str, timeout_sec: float = 1.0
    ) -> Generator[Dict[str, Any], None, None]:
        """Consume messages from in-memory topic queue until stopped or empty."""
        q = self._get_queue(topic)
        while self._running:
            try:
                msg = q.get(timeout=timeout_sec)
                yield msg
                q.task_done()
            except queue.Empty:
                break

    def close(self) -> None:
        self._running = False


class KafkaStreamingChannel(StreamingChannel):
    """Kafka client implementation using kafka-python."""

    def __init__(self, bootstrap_servers: str) -> None:
        from kafka import KafkaConsumer, KafkaProducer

        self.bootstrap_servers = bootstrap_servers
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
        self.consumer_instance: Optional[KafkaConsumer] = None

    def publish(self, topic: str, message: Dict[str, Any]) -> bool:
        try:
            future = self.producer.send(topic, value=message)
            future.get(timeout=10.0)
            return True
        except Exception as e:
            logger.error("Failed to publish message to Kafka topic %s: %s", topic, e)
            return False

    def consume(
        self, topic: str, timeout_sec: Optional[float] = 1.0
    ) -> Generator[Dict[str, Any], None, None]:
        from kafka import KafkaConsumer

        kwargs: Dict[str, Any] = {
            "bootstrap_servers": self.bootstrap_servers,
            "auto_offset_reset": "earliest",
            "enable_auto_commit": True,
            "group_id": settings.KAFKA_CONSUMER_GROUP,
            "value_deserializer": lambda x: json.loads(x.decode("utf-8")),
        }
        if timeout_sec is not None and timeout_sec > 0:
            kwargs["consumer_timeout_ms"] = int(timeout_sec * 1000)

        consumer = KafkaConsumer(topic, **kwargs)
        self.consumer_instance = consumer
        try:
            for message in consumer:
                yield message.value
        finally:
            consumer.close()

    def close(self) -> None:
        if self.producer:
            self.producer.flush()
            self.producer.close()
        if self.consumer_instance:
            self.consumer_instance.close()


# In-memory shared singleton for zero-Docker fallback
_shared_memory_channel = MemoryStreamingChannel()


def get_streaming_channel(
    bootstrap_servers: Optional[str] = None,
    force_fallback: bool = False,
) -> StreamingChannel:
    """Return a streaming channel, falling back gracefully to MemoryStreamingChannel if Kafka is offline."""
    servers = bootstrap_servers or settings.KAFKA_BOOTSTRAP_SERVERS

    if force_fallback or getattr(settings, "KAFKA_USE_FALLBACK", False):
        logger.info("Using MemoryStreamingChannel fallback (Docker-free mode).")
        return _shared_memory_channel

    try:
        from kafka import KafkaProducer

        # Attempt quick test connection
        prod = KafkaProducer(
            bootstrap_servers=servers,
            request_timeout_ms=1500,
            api_version_auto_timeout_ms=1500,
        )
        prod.close()
        logger.info("Connected to Kafka cluster at %s", servers)
        return KafkaStreamingChannel(bootstrap_servers=servers)
    except Exception as e:
        logger.warning(
            "Kafka broker at %s unreachable (%s). Using MemoryStreamingChannel fallback.",
            servers,
            e,
        )
        return _shared_memory_channel

