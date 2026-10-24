"""Unit tests for Phase 3: Model training, leakage prevention, MLflow logging, and reproducibility."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from src.training.data import (
    ALL_FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    create_preprocessor,
    get_feature_names_out,
    load_feature_split,
    validate_split_integrity,
)
from src.training.evaluate import (
    evaluate_predictions,
    find_optimal_thresholds,
    sweep_thresholds,
)


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Fixture providing valid sample transaction data for unit testing."""
    np.random.seed(42)
    n = 100
    types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    return pd.DataFrame(
        {
            "step": np.random.randint(1, 10, n),
            "type": np.random.choice(types, n),
            "amount": np.random.exponential(1000.0, n),
            "nameOrig": [f"C{i:07d}" for i in range(n)],
            "oldbalanceOrg": np.random.exponential(5000.0, n),
            "newbalanceOrig": np.random.exponential(5000.0, n),  # leakage
            "nameDest": [f"M{i:07d}" for i in range(n)],
            "oldbalanceDest": np.random.exponential(10000.0, n),
            "newbalanceDest": np.random.exponential(10000.0, n),  # leakage
            "isFraud": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "isFlaggedFraud": np.zeros(n, dtype=int),  # leakage
            "orig_txn_count_1h": np.random.randint(0, 5, n),
            "orig_txn_count_24h": np.random.randint(0, 10, n),
            "orig_amount_vs_mean": np.random.uniform(0.5, 2.0, n),
            "orig_time_since_last_txn": np.random.uniform(-1.0, 10.0, n),
            "orig_has_prev_txn": np.random.randint(0, 2, n),
            "dest_txn_count_24h": np.random.randint(0, 20, n),
            "dest_has_prev_txn": np.random.randint(0, 2, n),
            "event_timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        }
    )


def test_leakage_free_feature_selection():
    """Verify that feature extraction strictly excludes leakage and forbidden columns."""
    X_train, y_train = load_feature_split("train")

    # Assert target is detached
    assert isinstance(y_train, pd.Series)
    assert y_train.name == "isFraud"
    assert "isFraud" not in X_train.columns

    # Assert forbidden columns are absent
    for col in FORBIDDEN_COLUMNS:
        assert (
            col not in X_train.columns
        ), f"Leakage/Forbidden column '{col}' leaked into feature matrix!"

    # Assert only legitimate feature columns are present
    assert set(X_train.columns) == set(ALL_FEATURE_COLUMNS)


def test_leakage_detection_raises_error(sample_feature_df):
    """Verify that validate_split_integrity detects missing features or nulls."""
    corrupted_df = sample_feature_df.drop(columns=["amount"])
    with pytest.raises(ValueError, match="Required feature 'amount' missing"):
        validate_split_integrity(corrupted_df, "test_split")

    null_df = sample_feature_df.copy()
    null_df.loc[0, "orig_amount_vs_mean"] = np.nan
    with pytest.raises(ValueError, match="Null values detected"):
        validate_split_integrity(null_df, "test_split")


def test_train_val_schema_consistency():
    """Verify schema and feature definitions match identically across train, val, and test splits."""
    X_train, _ = load_feature_split("train")
    X_val, _ = load_feature_split("val")
    X_test, _ = load_feature_split("test")

    assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns)
    assert (
        X_train.dtypes.to_dict()
        == X_val.dtypes.to_dict()
        == X_test.dtypes.to_dict()
    )

    # Preprocessor output names
    preprocessor = create_preprocessor(scale_numerical=False)
    preprocessor.fit(X_train.iloc[:50])
    feature_names = get_feature_names_out(preprocessor)
    assert len(feature_names) == 15
    assert "type_TRANSFER" in feature_names
    assert "type_CASH_OUT" in feature_names


def test_model_training_and_inference(sample_feature_df):
    """Test model training, probability output ranges, and threshold-based decisions."""
    X = sample_feature_df[ALL_FEATURE_COLUMNS]
    y = sample_feature_df["isFraud"]

    preprocessor = create_preprocessor(scale_numerical=True)
    X_proc = preprocessor.fit_transform(X)

    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_proc, y)

    probs = clf.predict_proba(X_proc)[:, 1]
    assert len(probs) == len(sample_feature_df)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Evaluation and thresholding
    metrics = evaluate_predictions(y.to_numpy(), probs, threshold=0.5)
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert metrics.tp + metrics.fp + metrics.fn + metrics.tn == len(
        sample_feature_df
    )


def test_threshold_sweep_and_optimization():
    """Test that sweep_thresholds and find_optimal_thresholds correctly optimize."""
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.4, 0.9, 0.2, 0.1, 0.85])

    sweep_df = sweep_thresholds(y_true, y_prob, thresholds=[0.2, 0.5, 0.8])
    assert len(sweep_df) == 3
    assert set(sweep_df.columns).issuperset(
        {"threshold", "precision", "recall", "f1", "business_cost"}
    )

    opt = find_optimal_thresholds(sweep_df)
    assert "max_f1" in opt
    assert "min_cost" in opt
    assert opt["max_f1"]["f1"] > 0.0


def test_mlflow_logging_and_model_loading(tmp_path):
    """Test MLflow run tracking, parameter/metric logging, artifact persistence, and model reloading."""
    import mlflow
    import mlflow.xgboost

    tracking_uri = f"sqlite:///{tmp_path}/test_mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-experiment")

    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 5.0]])
    y = np.array([0, 0, 1, 1])

    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)

    with mlflow.start_run(run_name="unit-test-run") as run:
        run_id = run.info.run_id
        mlflow.log_param("test_param", 42)
        mlflow.log_metric("test_pr_auc", 0.95)
        mlflow.xgboost.log_model(model, artifact_path="model")

    # Verify run recorded
    client = mlflow.tracking.MlflowClient()
    fetched_run = client.get_run(run_id)
    assert fetched_run.data.params["test_param"] == "42"
    assert fetched_run.data.metrics["test_pr_auc"] == 0.95

    # Verify model reload and exact prediction equivalence
    loaded_model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    orig_preds = model.predict_proba(X)
    loaded_preds = loaded_model.predict_proba(X)
    np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)


def test_reproducibility_with_fixed_seeds():
    """Verify that training with fixed seeds generates perfectly reproducible predictions."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = (np.random.rand(200) < 0.2).astype(int)

    clf1 = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf1.fit(X, y)
    preds1 = clf1.predict_proba(X)[:, 1]

    clf2 = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf2.fit(X, y)
    preds2 = clf2.predict_proba(X)[:, 1]

    np.testing.assert_array_equal(
        preds1, preds2, err_msg="Models with identical seeds produced differing predictions!"
    )
