"""Inference and feature retrieval engine for FraudShield.

Loads models and preprocessors once at startup and retrieves point-in-time features from Feast
with deterministic cold-start defaults ensuring zero training/serving skew.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from feast import FeatureStore

from src.common.config import settings
from src.common.logging import get_logger
from src.serving.schemas import ScoreResponse, TransactionPayload
from src.training.blend import GNNRiskLookup, compute_blend_score
from src.training.data import ALL_FEATURE_COLUMNS

logger = get_logger(__name__)

# Predefined cold-start defaults for unseen accounts (documented in docs/features.md)
DEFAULT_FEATURE_VALUES: Dict[str, Any] = {
    "orig_txn_count_1h": 0,
    "orig_txn_count_24h": 0,
    "orig_amount_vs_mean": 1.0,
    "orig_time_since_last_txn": -1.0,
    "orig_has_prev_txn": 0,
    "dest_txn_count_24h": 0,
    "dest_has_prev_txn": 0,
}

ORIGIN_FEATURES: list[str] = [
    "origin_account_features:orig_txn_count_1h",
    "origin_account_features:orig_txn_count_24h",
    "origin_account_features:orig_amount_vs_mean",
    "origin_account_features:orig_time_since_last_txn",
    "origin_account_features:orig_has_prev_txn",
]

DEST_FEATURES: list[str] = [
    "destination_features:dest_txn_count_24h",
    "destination_features:dest_has_prev_txn",
]

COMBINED_FEATURES: list[str] = ORIGIN_FEATURES + DEST_FEATURES


class ScoringEngine:
    """Core scoring engine maintaining loaded models, preprocessors, and Feast store."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.feast_store: Optional[FeatureStore] = None
        self.gnn_lookup: Optional[GNNRiskLookup] = None
        self.model_version: str = settings.MODEL_VERSION
        self.model_name: str = settings.MODEL_NAME
        self.decision_threshold: float = settings.DECISION_THRESHOLD
        self.blend_alpha: float = settings.BLEND_ALPHA
        self.blend_decision_threshold: float = settings.BLEND_DECISION_THRESHOLD
        self._is_initialized: bool = False

    def load(self) -> None:
        """Load XGBoost model, fitted preprocessor, Feast store, and GNN risk lookup."""
        if self._is_initialized:
            return

        logger.info("Initializing FraudShield ScoringEngine...")

        # 1. Load Preprocessor
        preproc_path = Path(settings.PREPROCESSOR_LOCAL_PATH)
        if not preproc_path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found at {preproc_path}"
            )
        self.preprocessor = joblib.load(preproc_path)
        logger.info(
            "Loaded fitted preprocessor from %s", preproc_path.as_posix()
        )

        # 2. Load Model (MLflow with fallback to local JSON artifact)
        self._load_model()

        # 3. Load GNN Risk Lookup
        try:
            self.gnn_lookup = GNNRiskLookup()
            logger.info("Loaded GNN risk lookup for fast serving.")
        except Exception as gnn_err:
            logger.warning("Could not load GNN risk lookup: %s. Defaulting to fallback.", gnn_err)
            self.gnn_lookup = None

        # 4. Connect to Feast
        feast_repo = Path(settings.FEAST_REPO_PATH)
        if feast_repo.exists():
            try:
                self.feast_store = FeatureStore(repo_path=str(feast_repo))
                logger.info(
                    "Connected to Feast FeatureStore at %s",
                    feast_repo.as_posix(),
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize Feast FeatureStore: %s. Using default features.",
                    e,
                )
                self.feast_store = None
        else:
            logger.warning(
                "Feast repo path %s does not exist. Using fallback defaults.",
                feast_repo,
            )
            self.feast_store = None

        self._is_initialized = True
        logger.info("ScoringEngine initialization complete.")

    def reload(self) -> None:
        """Force reload the model and preprocessor from latest local or MLflow champion artifacts."""
        self._is_initialized = False
        self.load()
        logger.info("ScoringEngine successfully reloaded champion model.")

    def _load_model(self) -> None:
        """Attempt loading from MLflow registry; fall back to local JSON artifact if offline."""
        # Try loading from MLflow registry first
        try:
            import mlflow
            import mlflow.xgboost

            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            model_uri = f"models:/{self.model_name}/{self.model_version}"
            logger.info("Attempting to load model from MLflow: %s", model_uri)
            self.model = mlflow.xgboost.load_model(model_uri)
            if hasattr(self.model, "set_params"):
                self.model.set_params(n_jobs=1)
            logger.info("Successfully loaded model from MLflow (%s) with n_jobs=1", model_uri)
            return
        except Exception as mlflow_err:
            logger.info(
                "Could not load from MLflow registry (%s). Falling back to local artifact.",
                mlflow_err,
            )

        # Fallback to local model JSON
        local_path = Path(settings.MODEL_LOCAL_PATH)
        if not local_path.exists():
            raise FileNotFoundError(
                f"Local model artifact not found at {local_path}"
            )
        clf = xgb.XGBClassifier()
        clf.load_model(str(local_path))
        clf.set_params(n_jobs=1)
        self.model = clf
        logger.info("Successfully loaded model from %s with n_jobs=1", local_path.as_posix())

    @property
    def is_model_loaded(self) -> bool:
        """Check if model and preprocessor are loaded in memory."""
        return self.model is not None and self.preprocessor is not None

    @property
    def is_feast_reachable(self) -> bool:
        """Check if Feast feature store is loaded and operational."""
        return self.feast_store is not None

    def fetch_features(
        self, txn: TransactionPayload
    ) -> Tuple[Dict[str, Any], bool]:
        """Fetch online features from Feast, falling back to defaults for cold-start accounts.

        Returns:
            Tuple of (features_dict, is_fallback)
        """
        feature_fallback = False
        features: Dict[str, Any] = {}

        if self.feast_store is not None:
            try:
                # Combined single Feast online lookup for origin and destination entities
                online_res = self.feast_store.get_online_features(
                    features=COMBINED_FEATURES,
                    entity_rows=[{"nameOrig": txn.nameOrig, "nameDest": txn.nameDest}],
                ).to_dict()

                for f_name in [
                    "orig_txn_count_1h",
                    "orig_txn_count_24h",
                    "orig_amount_vs_mean",
                    "orig_time_since_last_txn",
                    "orig_has_prev_txn",
                    "dest_txn_count_24h",
                    "dest_has_prev_txn",
                ]:
                    val = online_res.get(f_name, [None])[0]
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        features[f_name] = DEFAULT_FEATURE_VALUES[f_name]
                        feature_fallback = True
                    else:
                        features[f_name] = val

            except Exception as feast_err:
                logger.warning(
                    "Feast online feature lookup failed: %s. Using default values.",
                    feast_err,
                )
                feature_fallback = True
                features.update(DEFAULT_FEATURE_VALUES)
        else:
            feature_fallback = True
            features.update(DEFAULT_FEATURE_VALUES)

        return features, feature_fallback

    def score(self, txn: TransactionPayload) -> ScoreResponse:
        """Score an incoming transaction using Feast online features and XGBoost.

        Measures end-to-end latency and returns standard ScoreResponse contract.
        """
        if not self._is_initialized or self.model is None or self.preprocessor is None:
            self.load()

        t0 = time.perf_counter()

        # 1. Fetch online features with cold-start fallback
        history_features, feature_fallback = self.fetch_features(txn)

        # 2. Assemble complete feature dictionary matching training columns exactly
        record = {
            "type": txn.type,
            "amount": float(txn.amount),
            "oldbalanceOrg": float(txn.oldbalanceOrg),
            "oldbalanceDest": float(txn.oldbalanceDest),
            "orig_txn_count_1h": int(history_features["orig_txn_count_1h"]),
            "orig_txn_count_24h": int(history_features["orig_txn_count_24h"]),
            "orig_amount_vs_mean": float(history_features["orig_amount_vs_mean"]),
            "orig_time_since_last_txn": float(
                history_features["orig_time_since_last_txn"]
            ),
            "orig_has_prev_txn": int(history_features["orig_has_prev_txn"]),
            "dest_txn_count_24h": int(history_features["dest_txn_count_24h"]),
            "dest_has_prev_txn": int(history_features["dest_has_prev_txn"]),
        }

        # 3. Construct single-row DataFrame using canonical training column order
        df = pd.DataFrame([record], columns=ALL_FEATURE_COLUMNS)

        # 4. Transform using fitted preprocessor
        x_proc = self.preprocessor.transform(df)

        # 5. XGBoost inference
        xgb_proba = float(self.model.predict_proba(x_proc)[0, 1])

        # 6. GNN Risk Score lookup and blending
        gnn_score: Optional[float] = None
        if self.gnn_lookup is not None:
            gnn_score = self.gnn_lookup.get_score(txn.nameOrig, txn.nameDest)

        blend_score, blend_used = compute_blend_score(
            xgb_score=xgb_proba,
            gnn_score=gnn_score,
            alpha=self.blend_alpha,
        )

        # 7. Alert decision based on blended or XGBoost threshold
        if blend_used:
            effective_threshold = self.blend_decision_threshold
            effective_score = blend_score
        else:
            effective_threshold = self.decision_threshold
            effective_score = xgb_proba

        is_alert = bool(effective_score >= effective_threshold)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ScoreResponse(
            transaction_id=txn.txn_id,
            fraud_score=round(effective_score, 6),
            is_alert=is_alert,
            latency_ms=round(latency_ms, 3),
            feature_fallback=feature_fallback,
            model_version=f"{self.model_name}:{self.model_version}",
            threshold=effective_threshold,
            xgb_score=round(xgb_proba, 6),
            gnn_score=round(gnn_score, 6) if gnn_score is not None else None,
            blend_score=round(blend_score, 6),
            blend_used=blend_used,
        )


# Global scoring engine instance
scoring_engine = ScoringEngine()

