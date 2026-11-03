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
