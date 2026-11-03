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
        "KAFKA_TOPIC_DEAD_LETTER", "transactions.dlq"
    )
    KAFKA_CONSUMER_GROUP: str = os.getenv(
        "KAFKA_CONSUMER_GROUP", "fraudshield-scoring-group"
    )

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

    # Data Paths
    DATA_RAW_PATH: str = os.getenv(
        "DATA_RAW_PATH", "data/raw/PS_20174392719_1491204439457_log.csv"
    )
    TRAIN_FEATURES_PATH: str = os.getenv(
        "TRAIN_FEATURES_PATH", "data/processed/train_features.parquet"
    )
    VAL_FEATURES_PATH: str = os.getenv(
        "VAL_FEATURES_PATH", "data/processed/val_features.parquet"
    )
    TEST_FEATURES_PATH: str = os.getenv(
        "TEST_FEATURES_PATH", "data/processed/test_features.parquet"
    )

    # Scoring & Thresholds
    DECISION_THRESHOLD: float = float(os.getenv("DECISION_THRESHOLD", "0.99"))
    ALERT_THRESHOLD_HIGH: float = float(os.getenv("ALERT_THRESHOLD_HIGH", "0.7"))
    ALERT_THRESHOLD_MED: float = float(os.getenv("ALERT_THRESHOLD_MED", "0.3"))
    GNN_BLEND_WEIGHT: float = float(os.getenv("GNN_BLEND_WEIGHT", "0.3"))
    BLEND_ALPHA: float = float(os.getenv("BLEND_ALPHA", "1.0"))
    BLEND_DECISION_THRESHOLD: float = float(
        os.getenv("BLEND_DECISION_THRESHOLD", "0.99")
    )
    METRICS_WINDOW_SIZE: int = int(os.getenv("METRICS_WINDOW_SIZE", "1000"))

    # Feast Feature Store
    FEAST_REPO_PATH: str = os.getenv("FEAST_REPO_PATH", "feature_repo")

    # Model Artifacts
    MODEL_NAME: str = os.getenv("MODEL_NAME", "FraudShield-XGBoost")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1")
    MODEL_LOCAL_PATH: str = os.getenv(
        "MODEL_LOCAL_PATH", "models/xgboost_model.json"
    )
    PREPROCESSOR_LOCAL_PATH: str = os.getenv(
        "PREPROCESSOR_LOCAL_PATH", "models/xgboost_preprocessor.joblib"
    )

    # Database Fallback
    USE_SQLITE_FALLBACK: bool = (
        os.getenv("USE_SQLITE_FALLBACK", "true").lower() in ("true", "1")
    )
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/fraudshield_serving.db")
    BENCHMARK_DB_PATH: str = os.getenv(
        "BENCHMARK_DB_PATH", "data/fraudshield_benchmark.db"
    )

    # Dataset Split Boundaries (time steps)
    TRAIN_MAX_STEP: int = int(os.getenv("TRAIN_MAX_STEP", "323"))
    VAL_MAX_STEP: int = int(os.getenv("VAL_MAX_STEP", "378"))
    TEST_MAX_STEP: int = int(os.getenv("TEST_MAX_STEP", "743"))

    # Graph & GNN Configuration
    GNN_MODEL_NAME: str = os.getenv("GNN_MODEL_NAME", "FraudShield-GNN")
    GRAPH_DATA_PATH: str = os.getenv("GRAPH_DATA_PATH", "models/graph_train_data.pt")
    GRAPH_NODE_MAPPING_PATH: str = os.getenv(
        "GRAPH_NODE_MAPPING_PATH", "models/graph_node_mapping.joblib"
    )
    GRAPH_SCALER_PATH: str = os.getenv(
        "GRAPH_SCALER_PATH", "models/graph_scaler.joblib"
    )
    GRAPH_RINGS_PATH: str = os.getenv(
        "GRAPH_RINGS_PATH", "models/graph_rings_summary.json"
    )
    GNN_MODEL_PATH: str = os.getenv("GNN_MODEL_PATH", "models/gnn_model.pt")
    GNN_SCORES_LOOKUP_PATH: str = os.getenv(
        "GNN_SCORES_LOOKUP_PATH", "models/gnn_scores_lookup.parquet"
    )

    # Phase 7: Drift Monitoring & Retraining Configuration
    DRIFT_REPORT_DIR: str = os.getenv("DRIFT_REPORT_DIR", "docs/results/drift")
    DRIFT_SHARE_THRESHOLD: float = float(
        os.getenv("DRIFT_SHARE_THRESHOLD", "0.3")
    )
    DRIFT_P_VALUE_THRESHOLD: float = float(
        os.getenv("DRIFT_P_VALUE_THRESHOLD", "0.05")
    )
    DRIFT_FEATURES: tuple = (
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "orig_txn_count_1h",
        "orig_txn_count_24h",
        "orig_amount_vs_mean",
        "orig_time_since_last_txn",
        "orig_has_prev_txn",
        "dest_txn_count_24h",
        "dest_has_prev_txn",
        "type",
        "fraud_score",
    )
    MIN_NEW_LABELS_FOR_RETRAIN: int = int(
        os.getenv("MIN_NEW_LABELS_FOR_RETRAIN", "10")
    )
    MIN_IMPROVEMENT_PR_AUC: float = float(
        os.getenv("MIN_IMPROVEMENT_PR_AUC", "0.001")
    )
    FROZEN_GATE_SET_PATH: str = os.getenv(
        "FROZEN_GATE_SET_PATH", "data/processed/frozen_gate_set.parquet"
    )
    RETRAIN_AUDIT_LOG_PATH: str = os.getenv(
        "RETRAIN_AUDIT_LOG_PATH", "docs/results/retrain_audit_log.json"
    )


settings = Settings()
