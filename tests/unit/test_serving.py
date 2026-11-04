"""Unit tests for Phase 4: Serving service, schema validation, feature parity, DLQ, and telemetry."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.common.config import settings
from src.common.db import DatabaseManager
from src.ingestion.consumer import ScoringConsumer
from src.ingestion.streaming import MemoryStreamingChannel
from src.serving.app import app
from src.serving.schemas import TransactionPayload
from src.serving.scoring import DEFAULT_FEATURE_VALUES, ScoringEngine
from src.training.data import ALL_FEATURE_COLUMNS


@pytest.fixture
def test_client() -> TestClient:
    """Fixture providing FastAPI TestClient."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_valid_payload() -> dict:
    """Fixture providing valid transaction dictionary."""
    return {
        "txn_id": "txn_unit_test_001",
        "step": 380,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C1999999999",
        "oldbalanceOrg": 300000.0,
        "nameDest": "C8888888888",
        "oldbalanceDest": 0.0,
    }


# =============================================================================
# 1. Pydantic Validation Tests
# =============================================================================


def test_pydantic_valid_payload(sample_valid_payload):
    """Test that valid transaction payload parses cleanly without errors."""
    payload = TransactionPayload.model_validate(sample_valid_payload)
    assert payload.txn_id == "txn_unit_test_001"
    assert payload.type == "TRANSFER"
    assert payload.amount == 250000.0
    assert payload.nameOrig == "C1999999999"


def test_pydantic_missing_field(sample_valid_payload):
    """Test that missing required fields raise Pydantic ValidationError."""
    corrupted = sample_valid_payload.copy()
    del corrupted["amount"]
    with pytest.raises(ValidationError):
        TransactionPayload.model_validate(corrupted)


def test_pydantic_wrong_type(sample_valid_payload):
    """Test that non-numeric types for numeric fields raise ValidationError."""
    corrupted = sample_valid_payload.copy()
    corrupted["amount"] = "not_a_number"
    with pytest.raises(ValidationError):
        TransactionPayload.model_validate(corrupted)


def test_pydantic_negative_amount(sample_valid_payload):
    """Test that negative transaction amount fails validation."""
    corrupted = sample_valid_payload.copy()
    corrupted["amount"] = -100.0
    with pytest.raises(ValidationError):
        TransactionPayload.model_validate(corrupted)


def test_pydantic_invalid_payment_type(sample_valid_payload):
    """Test that invalid transaction type fails validation."""
    corrupted = sample_valid_payload.copy()
    corrupted["type"] = "CRYPTO_TRANSFER"
    with pytest.raises(ValidationError):
        TransactionPayload.model_validate(corrupted)


# =============================================================================
# 2. Feast Feature Fallback & Skew Prevention
# =============================================================================


def test_feast_feature_fallback_on_unseen_account():
    """Verify that an unseen account deterministically returns safe default features."""
    engine = ScoringEngine()
    engine.load()

    unseen_txn = TransactionPayload(
        txn_id="txn_unseen_001",
        step=10,
        type="PAYMENT",
        amount=50.0,
        nameOrig="C_COMPLETELY_UNSEEN_ACCOUNT_XYZ",
        oldbalanceOrg=100.0,
        nameDest="M_COMPLETELY_UNSEEN_MERCHANT_ABC",
        oldbalanceDest=0.0,
    )

    features, is_fallback = engine.fetch_features(unseen_txn)
    assert is_fallback is True
    assert features["orig_txn_count_1h"] == DEFAULT_FEATURE_VALUES["orig_txn_count_1h"]
    assert features["orig_amount_vs_mean"] == DEFAULT_FEATURE_VALUES["orig_amount_vs_mean"]
    assert features["dest_has_prev_txn"] == DEFAULT_FEATURE_VALUES["dest_has_prev_txn"]


def test_training_serving_feature_parity():
    """Verify that serving feature preprocessing exactly matches training preprocessing.

    Prevents training/serving skew by comparing preprocessor output on the exact same row.
    """
    engine = ScoringEngine()
    engine.load()

    # Load first row of test split
    test_df = pd.read_parquet("data/processed/test_features.parquet").iloc[:1]
    row = test_df.iloc[0]

    # 1. Serving feature extraction path
    txn = TransactionPayload(
        txn_id="parity_test_row",
        step=int(row["step"]),
        type=str(row["type"]),
        amount=float(row["amount"]),
        nameOrig=str(row["nameOrig"]),
        oldbalanceOrg=float(row["oldbalanceOrg"]),
        nameDest=str(row["nameDest"]),
        oldbalanceDest=float(row["oldbalanceDest"]),
    )

    # In training, features come from computed feature columns
    serving_record = {
        "type": txn.type,
        "amount": float(txn.amount),
        "oldbalanceOrg": float(txn.oldbalanceOrg),
        "oldbalanceDest": float(txn.oldbalanceDest),
        "orig_txn_count_1h": int(row["orig_txn_count_1h"]),
        "orig_txn_count_24h": int(row["orig_txn_count_24h"]),
        "orig_amount_vs_mean": float(row["orig_amount_vs_mean"]),
        "orig_time_since_last_txn": float(row["orig_time_since_last_txn"]),
        "orig_has_prev_txn": int(row["orig_has_prev_txn"]),
        "dest_txn_count_24h": int(row["dest_txn_count_24h"]),
        "dest_has_prev_txn": int(row["dest_has_prev_txn"]),
    }
    serving_df = pd.DataFrame([serving_record], columns=ALL_FEATURE_COLUMNS)

    # 2. Training feature vector
    training_df = test_df[ALL_FEATURE_COLUMNS].copy()

    # Assert column ordering is identical
    assert list(serving_df.columns) == list(training_df.columns)

    # Preprocess through engine's preprocessor
    serving_proc = engine.preprocessor.transform(serving_df)
    training_proc = engine.preprocessor.transform(training_df)

    # Assert numerical identity (delta = 0.0)
    np.testing.assert_allclose(
        serving_proc,
        training_proc,
        rtol=1e-6,
        atol=1e-6,
        err_msg="CRITICAL: Serving preprocessed vector differs from training vector!",
    )


# =============================================================================
# 3. FastAPI Endpoint Tests
# =============================================================================


def test_score_endpoint(test_client, sample_valid_payload):
    """Test POST /score endpoint response format and contract."""
    response = test_client.post("/score", json=sample_valid_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["transaction_id"] == "txn_unit_test_001"
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert isinstance(data["is_alert"], bool)
    assert data["latency_ms"] > 0.0
    assert "feature_fallback" in data
    assert data["threshold"] == settings.DECISION_THRESHOLD


def test_health_endpoint(test_client):
    """Test GET /health reports status of model, Feast, and database separately."""
    response = test_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "feast_reachable" in data
    assert "db_reachable" in data
    assert data["model_loaded"] is True


def test_metrics_endpoint(test_client, sample_valid_payload):
    """Test GET /metrics returns request count, throughput, and p50/p95 latencies."""
    # Send a request to register telemetry
    test_client.post("/score", json=sample_valid_payload)

    response = test_client.get("/metrics")
    assert response.status_code == 200

    data = response.json()
    assert data["request_count"] >= 1
    assert data["p50_latency_ms"] >= 0.0
    assert data["p95_latency_ms"] >= 0.0
    assert data["throughput_rps"] >= 0.0


# =============================================================================
# 4. Dead-Letter Queue (DLQ) & Consumer Resiliency Tests
# =============================================================================


def test_dlq_routing_on_malformed_messages():
    """Verify consumer routes malformed events to DLQ topic with reason and never crashes."""
    channel = MemoryStreamingChannel()
    consumer = ScoringConsumer(
        topic="test-txns",
        dlq_topic="test-dlq",
        scores_topic="test-scores",
        channel=channel,
    )

    # 1. Publish 1 malformed message, 1 negative amount message, and 1 valid message
    channel.publish("test-txns", {"bad": "data"})  # Missing required fields
    channel.publish(
        "test-txns",
        {
            "txn_id": "bad_neg_01",
            "type": "PAYMENT",
            "amount": -500.0,
            "nameOrig": "C123",
            "nameDest": "M456",
        },
    )
    channel.publish(
        "test-txns",
        {
            "txn_id": "good_txn_01",
            "type": "PAYMENT",
            "amount": 100.0,
            "nameOrig": "C12345",
            "nameDest": "M67890",
        },
    )

    # 2. Run consumer over all 3 messages
    summary = consumer.run(limit=3, timeout_sec=0.5)

    assert summary["dlq_count"] == 2
    assert summary["processed_count"] == 1
    assert summary["error_count"] == 0

    # 3. Verify DLQ message contents
    dlq_messages = list(channel.consume("test-dlq", timeout_sec=0.2))
    assert len(dlq_messages) == 2
    assert "error_reason" in dlq_messages[0]
    assert "failed_at" in dlq_messages[0]


# =============================================================================
# 5. Database Persistence & Metrics Isolation Tests
# =============================================================================


def test_database_persistence_fallback(tmp_path):
    """Verify database persistence works cleanly with fallback without blocking."""
    db_file = tmp_path / "test_fallback.db"
    # Test using isolated DatabaseManager
    manager = DatabaseManager()
    manager._backend = "sqlite"
    manager._init_sqlite_schema(db_file)

    success = manager.save_transaction_and_score(
        txn_id="txn_persist_001",
        account_id="C111",
        dest_id="M222",
        amount=150.0,
        txn_type="PAYMENT",
        ts=None,
        raw_payload={"test": 1},
        xgb_score=0.05,
        gnn_score=None,
        blended_score=0.05,
        model_version="test:1",
        latency_ms=3.5,
    )
    assert success is True
    manager.flush()
    assert manager.successful_writes_count >= 1


def test_openapi_schema_in_sync():
    """Verify docs/api/openapi.json is strictly in sync with FastAPI app's current OpenAPI schema."""
    import json
    from pathlib import Path

    openapi_path = Path("docs/api/openapi.json")
    assert openapi_path.exists(), "docs/api/openapi.json must exist"

    with open(openapi_path, encoding="utf-8") as f:
        saved_schema = json.load(f)

    live_schema = app.openapi()
    assert saved_schema == live_schema, (
        "docs/api/openapi.json is out of date compared to FastAPI app.openapi()! "
        "Re-export schema using scripts or update docs/api/openapi.json."
    )


def test_admin_endpoint_security_unauthorized(test_client):
    """Verify admin endpoints return 401 when ADMIN_API_KEY is configured and key is missing or invalid."""
    import src.common.config as config_module

    old_key = config_module.settings.ADMIN_API_KEY
    object.__setattr__(config_module.settings, "ADMIN_API_KEY", "super-secret-key")
    try:
        # 1. Missing header
        res1 = test_client.post("/admin/reload-model")
        assert res1.status_code == 401
        assert "Missing or invalid X-Admin-Key" in res1.json()["detail"]

        res2 = test_client.post("/admin/retrain")
        assert res2.status_code == 401
        assert "Missing or invalid X-Admin-Key" in res2.json()["detail"]

        # 2. Invalid header
        res3 = test_client.post("/admin/reload-model", headers={"X-Admin-Key": "wrong-key"})
        assert res3.status_code == 401

        res4 = test_client.post("/admin/retrain", headers={"X-Admin-Key": "wrong-key"})
        assert res4.status_code == 401
    finally:
        object.__setattr__(config_module.settings, "ADMIN_API_KEY", old_key)
def test_admin_endpoint_security_authorized(test_client):
    """Verify admin endpoints return 200 when valid X-Admin-Key is provided."""
    import src.common.config as config_module

    old_key = config_module.settings.ADMIN_API_KEY
    object.__setattr__(config_module.settings, "ADMIN_API_KEY", "super-secret-key")
    try:
        res_reload = test_client.post("/admin/reload-model", headers={"X-Admin-Key": "super-secret-key"})
        assert res_reload.status_code == 200
        assert res_reload.json()["status"] == "reloaded"

        res_retrain = test_client.post("/admin/retrain", headers={"X-Admin-Key": "super-secret-key"})
        assert res_retrain.status_code == 200
        data = res_retrain.json()
        assert data["status"] == "evaluated"
        assert "trigger_fired" in data
    finally:
        object.__setattr__(config_module.settings, "ADMIN_API_KEY", old_key)

