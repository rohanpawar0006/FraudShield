"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Central configuration for FraudShield services."""

    # Service & Environment
    ENV: str = os.getenv("FRAUDSHIELD_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
    )
    KAFKA_TOPIC_TRANSACTIONS: str = os.getenv(
        "KAFKA_TOPIC_TRANSACTIONS", "transactions"
    )
    KAFKA_TOPIC_SCORES: str = os.getenv("KAFKA_TOPIC_SCORES", "scores")
    KAFKA_TOPIC_DEAD_LETTER: str = os.getenv(
        "KAFKA_TOPIC_DEAD_LETTER", "transactions-dlq"
    )
    KAFKA_CONSUMER_GROUP: str = os.getenv(
        "KAFKA_CONSUMER_GROUP", "fraudshield-scoring-group"
    )

    # Redis (Feast Online Store)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

    # PostgreSQL
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "fraudshield")

    @property
    def postgres_uri(self) -> str:
        """Construct PostgreSQL connection string."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"
    )
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "fraudshield-detection"
    )

    # Scoring & Thresholds
    ALERT_THRESHOLD_HIGH: float = float(os.getenv("ALERT_THRESHOLD_HIGH", "0.7"))
    ALERT_THRESHOLD_MED: float = float(os.getenv("ALERT_THRESHOLD_MED", "0.3"))
    GNN_BLEND_WEIGHT: float = float(os.getenv("GNN_BLEND_WEIGHT", "0.3"))


settings = Settings()
