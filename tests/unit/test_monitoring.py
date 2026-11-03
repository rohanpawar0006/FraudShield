"""Unit tests for Phase 7 drift monitoring, retrain triggers, and promotion gate."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.monitoring.drift import run_drift_analysis
from src.monitoring.retrain_trigger import (
    check_retrain_trigger,
    evaluate_promotion_gate,
    prepare_retraining_dataset,
)
from src.serving.app import app


@pytest.fixture
def sample_feature_df():
    """Generates a synthetic DataFrame matching the monitored features."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "amount": np.random.lognormal(mean=7.0, sigma=1.0, size=n),
            "oldbalanceOrg": np.random.uniform(100, 10000, size=n),
            "newbalanceOrig": np.random.uniform(0, 5000, size=n),
            "oldbalanceDest": np.random.uniform(0, 10000, size=n),
            "newbalanceDest": np.random.uniform(100, 20000, size=n),
            "errorBalanceOrig": np.zeros(n),
            "errorBalanceDest": np.zeros(n),
            "orig_amount_vs_mean": np.random.normal(1.0, 0.2, size=n),
            "orig_amount_vs_max": np.random.normal(0.5, 0.1, size=n),
            "orig_velocity_24h": np.random.poisson(2, size=n),
            "type": np.random.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], size=n),
            "fraud_score": np.random.beta(0.5, 5.0, size=n),
        }
    )


def test_drift_analysis_clean_baseline(sample_feature_df, tmp_path):
    """When reference and current have identical distributions, drift is minimal/zero."""
    ref_df = sample_feature_df.copy()
    cur_df = sample_feature_df.copy()

    result = run_drift_analysis(
        reference_df=ref_df,
        current_df=cur_df,
        output_dir=str(tmp_path),
        drift_share_threshold=0.30,
    )

    assert result["drift_share"] < 0.30
    assert result["overall_drift"] is False
    assert Path(result["html_report_path"]).exists()
    assert Path(result["json_summary_path"]).exists()


def test_drift_analysis_detects_synthetic_drift(sample_feature_df, tmp_path):
    """When current data features are massively shifted, drift is detected."""
    ref_df = sample_feature_df.copy()
    cur_df = sample_feature_df.copy()

    # Shift amounts and balance distributions drastically
    cur_df["amount"] = cur_df["amount"] * 25.0
    cur_df["oldbalanceOrg"] = cur_df["oldbalanceOrg"] * 0.01
    cur_df["orig_velocity_24h"] = cur_df["orig_velocity_24h"] + 50
    cur_df["type"] = "DEBIT"  # Completely new dominant type
    cur_df["fraud_score"] = 0.99  # Massive score drift

    result = run_drift_analysis(
        reference_df=ref_df,
        current_df=cur_df,
        output_dir=str(tmp_path),
        drift_share_threshold=0.20,
    )

    assert result["drift_share"] >= 0.20
    assert result["overall_drift"] is True
    assert result["prediction_drift"] is True


def test_retrain_trigger_conditions():
    """Verify trigger conditions: drift share threshold, labeled fraud counts, combined."""
    # 1. Neither condition met
    res1 = check_retrain_trigger(
        drift_share=0.10,
        drift_threshold=0.30,
        new_labels_count=5,
        min_new_labels=10,
    )
    assert not res1["trigger_fired"]
    assert res1["reason"] == "no_trigger_condition_met"
    assert not res1["drift_condition_met"]
    assert not res1["label_condition_met"]

    # 2. Only drift condition met
    res2 = check_retrain_trigger(
        drift_share=0.35,
        drift_threshold=0.30,
        new_labels_count=5,
        min_new_labels=10,
    )
    assert res2["trigger_fired"]
    assert res2["reason"] == "data_drift_exceeded_threshold"
    assert res2["drift_condition_met"]
    assert not res2["label_condition_met"]

    # 3. Only label threshold met
    res3 = check_retrain_trigger(
        drift_share=0.10,
        drift_threshold=0.30,
        new_labels_count=15,
        min_new_labels=10,
    )
    assert res3["trigger_fired"]
    assert res3["reason"] == "analyst_labels_threshold_met"
    assert not res3["drift_condition_met"]
    assert res3["label_condition_met"]

    # 4. Both met
    res4 = check_retrain_trigger(
        drift_share=0.40,
        drift_threshold=0.30,
        new_labels_count=20,
        min_new_labels=10,
    )
    assert res4["trigger_fired"]
    assert res4["reason"] == "both_drift_and_labels"
    assert res4["drift_condition_met"] and res4["label_condition_met"]


def test_retraining_dataset_zero_leakage_enforcement(tmp_path):
    """Enforces zero leakage: if any frozen gate set ID appears in retraining data, raises ValueError."""
    gate_df = pd.DataFrame({"txn_id": ["TX_GATE_1", "TX_GATE_2", "TX_GATE_3"]})

    # Clean labeled cases (disjoint IDs)
    clean_labels = pd.DataFrame(
        {
            "transaction_id": ["TX_NEW_1", "TX_NEW_2"],
            "analyst_label": [1, 0],
        }
    )

    base_train = pd.DataFrame(
        {
            "txn_id": ["TX_TRAIN_1", "TX_TRAIN_2"],
            "isFraud": [0, 1],
            "step": [1, 2],
        }
    )
    base_train_path = tmp_path / "base_train.parquet"
    base_train.to_parquet(base_train_path)

    # Clean labels pass without error
    with patch("src.monitoring.retrain_trigger.get_labeled_cases", return_value=clean_labels):
        df_train, gate = prepare_retraining_dataset(
            base_train_path=str(base_train_path),
            frozen_gate_df=gate_df,
        )
        assert len(df_train) == 2

    # Contaminated labels containing gate set ID
    leaking_labels = pd.DataFrame(
        {
            "transaction_id": ["TX_GATE_2", "TX_NEW_3"],
            "analyst_label": [1, 0],
        }
    )

    with patch("src.monitoring.retrain_trigger.get_labeled_cases", return_value=leaking_labels):
        with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
            prepare_retraining_dataset(
                base_train_path=str(base_train_path),
                frozen_gate_df=gate_df,
            )


def test_promotion_gate_decision_promotion_and_rejection(tmp_path):
    """Tests honest promotion gate decisions for both promotion and rejection scenarios."""
    from src.common.config import settings

    gate_df = pd.read_parquet(settings.FROZEN_GATE_SET_PATH).head(10).copy()
    gate_df["isFraud"] = [0, 0, 0, 0, 0, 1, 1, 1, 0, 1]
    n = len(gate_df)

    # Mock preprocessors
    mock_prep = MagicMock()
    mock_prep.transform.return_value = np.zeros((n, 2))
    mock_prep.get_feature_names_out.return_value = ["f1", "f2"]

    # Mock models (ensure predict_proba attribute does not exist on booster mocks)
    mock_champ_model = MagicMock(spec=["predict"])
    mock_cand_model = MagicMock(spec=["predict"])

    # Case 1: Candidate is WORSE than champion -> REJECTED
    # Champion predicts well (frauds score 0.99, non-frauds score 0.01)
    mock_champ_model.predict.return_value = np.where(gate_df["isFraud"] == 1, 0.99, 0.01)
    # Candidate predicts poorly (frauds score 0.20, non-frauds score 0.60)
    mock_cand_model.predict.return_value = np.where(gate_df["isFraud"] == 1, 0.20, 0.60)

    decision_rejected = evaluate_promotion_gate(
        champion_preprocessor=mock_prep,
        champion_model=mock_champ_model,
        candidate_preprocessor=mock_prep,
        candidate_model=mock_cand_model,
        gate_df=gate_df,
        candidate_version="2",
        trigger_reason="test_drift",
        min_improvement=0.001,
        save_audit_log=False,
    )

    assert decision_rejected["promoted"] is False
    assert decision_rejected["decision"] == "REJECTED"
    assert decision_rejected["candidate_pr_auc"] < decision_rejected["champion_pr_auc"]

    # Case 2: Candidate is BETTER than champion -> PROMOTED
    # Champion predicts mediocre
    mock_champ_model.predict.return_value = np.full(n, 0.5)
    # Candidate predicts perfectly
    mock_cand_model.predict.return_value = np.where(gate_df["isFraud"] == 1, 0.99, 0.01)

    with (
        patch("src.monitoring.retrain_trigger.MlflowClient") as mock_mlflow_client,
        patch("src.monitoring.retrain_trigger.joblib.dump"),
    ):
        decision_promoted = evaluate_promotion_gate(
            champion_preprocessor=mock_prep,
            champion_model=mock_champ_model,
            candidate_preprocessor=mock_prep,
            candidate_model=mock_cand_model,
            gate_df=gate_df,
            candidate_version="3",
            trigger_reason="test_labels",
            min_improvement=0.001,
            save_audit_log=False,
        )

        assert decision_promoted["promoted"] is True
        assert decision_promoted["decision"] == "PROMOTED"
        assert decision_promoted["candidate_pr_auc"] > decision_promoted["champion_pr_auc"]
        mock_mlflow_client.return_value.set_registered_model_alias.assert_called_with(
            "FraudShield-XGBoost", "champion", "3"
        )


def test_admin_reload_model_endpoint():
    """Verify that POST /admin/reload-model reloads model cleanly without restarting."""
    client = TestClient(app)
    response = client.post("/admin/reload-model")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reloaded"
    assert "model_version" in data
