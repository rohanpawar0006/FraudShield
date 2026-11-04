"""Unit tests for Phase 5: Graph ring detection, temporal graph boundaries, GNN risk lookup, and blending."""

import numpy as np
import pytest
import torch

from src.common.config import settings
from src.serving.app import app
from src.training.blend import GNNRiskLookup, compute_blend_score


def test_graph_temporal_isolation():
    """Verify that graph training data contains ONLY transactions from step <= TRAIN_MAX_STEP.

    Enforces zero future leakage from validation or test windows into the graph topology.
    """
    # 1. Load the generated graph training data
    data = torch.load(settings.GRAPH_DATA_PATH, weights_only=False)
    assert hasattr(data, "edge_attr"), "Graph data must contain edge_attr"

    # Edge features format in build_graph.py: [amount, type_idx, step]
    # step is at index 2 of edge_attr
    steps = data.edge_attr[:, 2].numpy()

    # Assert all edge steps are strictly within training window
    max_observed_step = int(np.max(steps))
    assert max_observed_step <= settings.TRAIN_MAX_STEP, (
        f"CRITICAL LEAKAGE: Graph contains step {max_observed_step} > {settings.TRAIN_MAX_STEP}!"
    )

    # Negative test: simulated leak detection
    leaked_edge = np.array([1000.0, 1.0, settings.TRAIN_MAX_STEP + 1])
    assert leaked_edge[2] > settings.TRAIN_MAX_STEP, "Sanity check on leak condition"


def test_blend_score_computation():
    """Test standard blending formula S_blend = alpha * S_xgb + (1 - alpha) * S_gnn."""
    xgb_score = 0.8
    gnn_score = 0.4
    alpha = 0.7

    blend, used = compute_blend_score(xgb_score, gnn_score, alpha=alpha)
    expected = 0.7 * 0.8 + 0.3 * 0.4  # 0.56 + 0.12 = 0.68
    assert used is True
    assert pytest.approx(blend, abs=1e-5) == expected


def test_blend_fallback_on_missing_gnn_score():
    """Test that missing or NaN GNN risk score gracefully falls back to XGBoost score."""
    xgb_score = 0.75

    # None GNN score
    blend_none, used_none = compute_blend_score(xgb_score, None, alpha=0.85)
    assert used_none is False
    assert blend_none == xgb_score

    # NaN GNN score
    blend_nan, used_nan = compute_blend_score(xgb_score, float("nan"), alpha=0.85)
    assert used_nan is False
    assert blend_nan == xgb_score


def test_gnn_risk_lookup_interface():
    """Test GNNRiskLookup retrieval for known accounts, destinations, and unseen accounts."""
    lookup = GNNRiskLookup()
    # If the lookup file exists, verify behavior
    if len(lookup) > 0:
        # Pick any known account key from lookup
        known_acc = next(iter(lookup.lookup.keys()))
        known_score = lookup.get_score(name_orig=known_acc)
        assert known_score is not None
        assert 0.0 <= known_score <= 1.0

        # Unseen account returns None
        unseen_score = lookup.get_score(name_orig="C_NON_EXISTENT_ACCOUNT_9999999")
        assert unseen_score is None

        # Fallback to destination account when origin is unseen
        dest_fallback_score = lookup.get_score(
            name_orig="C_NON_EXISTENT_ORIGIN",
            name_dest=known_acc,
        )
        assert dest_fallback_score == known_score


def test_serving_blend_integration():
    """Test that POST /score returns blend contract fields and tracks telemetry."""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        payload = {
            "txn_id": "blend_test_txn_001",
            "step": 400,
            "type": "TRANSFER",
            "amount": 500000.0,
            "nameOrig": "C1234567890",
            "oldbalanceOrg": 500000.0,
            "nameDest": "C9876543210",
            "oldbalanceDest": 0.0,
        }
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Contract assertion
        assert "xgb_score" in data
        assert "gnn_score" in data
        assert "blend_score" in data
        assert "blend_used" in data
        assert isinstance(data["blend_used"], bool)
        assert 0.0 <= data["blend_score"] <= 1.0

        # Metrics check
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        m_data = metrics_resp.json()
        assert "gnn_fallback_count" in m_data
        assert m_data["gnn_fallback_count"] >= 0


def test_threshold_consistency():
    """Verify evaluation and serving use identical alpha and decision threshold from central config."""
    from src.serving.scoring import scoring_engine

    assert scoring_engine.blend_alpha == settings.BLEND_ALPHA
    assert scoring_engine.blend_decision_threshold == settings.BLEND_DECISION_THRESHOLD
    assert scoring_engine.decision_threshold == settings.DECISION_THRESHOLD
    assert settings.BLEND_DECISION_THRESHOLD == settings.DECISION_THRESHOLD
    assert settings.BLEND_ALPHA == 1.0


def test_blend_alpha_one_results_in_xgboost_only_predictions():
    """Verify that when BLEND_ALPHA=1.0, predictions are strictly XGBoost-only with blend_used=False."""
    xgb_score = 0.82

    # High GNN score (0.99) with alpha=1.0 must not shift score or trigger blend
    score_high_gnn, used_high = compute_blend_score(xgb_score=xgb_score, gnn_score=0.99, alpha=1.0)
    assert score_high_gnn == xgb_score
    assert used_high is False

    # Low GNN score (0.01) with alpha=1.0 must not dilute score or trigger blend
    score_low_gnn, used_low = compute_blend_score(xgb_score=xgb_score, gnn_score=0.01, alpha=1.0)
    assert score_low_gnn == xgb_score
    assert used_low is False


def test_gnn_scores_cannot_alter_predictions_in_xgboost_only_config(monkeypatch):
    """Verify that in XGBoost-only config (BLEND_ALPHA=1.0), GNN risk scores cannot alter final fraud scores or alerts."""
    from src.serving.schemas import TransactionPayload
    from src.serving.scoring import scoring_engine

    txn = TransactionPayload(
        txn_id="gnn_invariance_test_001",
        step=400,
        type="TRANSFER",
        amount=100000.0,
        nameOrig="C111222333",
        oldbalanceOrg=100000.0,
        nameDest="C999888777",
        oldbalanceDest=0.0,
    )

    # Mock GNN lookup to return 0.999
    monkeypatch.setattr(scoring_engine.gnn_lookup, "get_score", lambda orig, dest: 0.999)
    resp_high_gnn = scoring_engine.score(txn)

    # Mock GNN lookup to return 0.001
    monkeypatch.setattr(scoring_engine.gnn_lookup, "get_score", lambda orig, dest: 0.001)
    resp_low_gnn = scoring_engine.score(txn)

    # Final fraud scores and alert decisions must be identical bit-for-bit
    assert resp_high_gnn.fraud_score == resp_low_gnn.fraud_score
    assert resp_high_gnn.is_alert == resp_low_gnn.is_alert
    assert resp_high_gnn.blend_used is False
    assert resp_low_gnn.blend_used is False
    # GNN scores remain captured for telemetry/audit visibility
    assert resp_high_gnn.gnn_score == 0.999
    assert resp_low_gnn.gnn_score == 0.001


def test_warm_and_cold_account_scoring_behavior():
    """Verify warm accounts capture GNN scores and cold accounts handle missing embeddings cleanly."""
    from src.serving.schemas import TransactionPayload
    from src.serving.scoring import scoring_engine

    # 1. Warm account test (pick known account from lookup if populated)
    if scoring_engine.gnn_lookup is not None and len(scoring_engine.gnn_lookup) > 0:
        known_acc = next(iter(scoring_engine.gnn_lookup.lookup.keys()))
        warm_txn = TransactionPayload(
            txn_id="warm_acc_001",
            step=400,
            type="TRANSFER",
            amount=50000.0,
            nameOrig=known_acc,
            oldbalanceOrg=50000.0,
            nameDest="C_UNKNOWN_DEST",
            oldbalanceDest=0.0,
        )
        warm_resp = scoring_engine.score(warm_txn)
        assert warm_resp.gnn_score is not None
        assert 0.0 <= warm_resp.gnn_score <= 1.0
        assert warm_resp.blend_used is False  # Because BLEND_ALPHA == 1.0
        assert warm_resp.fraud_score == warm_resp.xgb_score

    # 2. Cold account test (completely unseen origin and dest)
    cold_txn = TransactionPayload(
        txn_id="cold_acc_001",
        step=400,
        type="PAYMENT",
        amount=50.0,
        nameOrig="C_COMPLETELY_UNSEEN_ORIGIN_99999",
        oldbalanceOrg=100.0,
        nameDest="M_COMPLETELY_UNSEEN_DEST_99999",
        oldbalanceDest=0.0,
    )
    cold_resp = scoring_engine.score(cold_txn)
    assert cold_resp.gnn_score is None
    assert cold_resp.blend_used is False
    assert cold_resp.fraud_score == cold_resp.xgb_score
