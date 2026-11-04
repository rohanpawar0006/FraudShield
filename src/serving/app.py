"""FastAPI application for real-time financial fraud risk scoring."""

import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import src.common.config as config_module
from src.common.db import db_manager
from src.common.logging import get_logger
from src.serving.metrics import metrics_tracker
from src.serving.schemas import (
    HealthResponse,
    MetricsResponse,
    ScoreResponse,
    TransactionPayload,
)
from src.serving.scoring import scoring_engine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: load ML models and initialize stores once on startup."""
    logger.info("Starting FraudShield Serving Service...")
    scoring_engine.load()
    db_connected = db_manager.is_reachable()
    logger.info(
        "Service started. Model loaded: %s, Feast: %s, DB: %s (%s)",
        scoring_engine.is_model_loaded,
        scoring_engine.is_feast_reachable,
        db_connected,
        db_manager.backend,
    )
    yield
    logger.info("Shutting down FraudShield Serving Service...")


app = FastAPI(
    title="FraudShield Real-Time Risk Scoring API",
    version="1.0.0",
    description="High-throughput streaming fraud detection scoring service backed by Feast and XGBoost.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle and track validation errors without crashing."""
    metrics_tracker.record_error()
    logger.warning("Request validation failed: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.post(
    "/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a financial transaction for fraud risk",
)
async def score_transaction(
    payload: TransactionPayload,
    background_tasks: BackgroundTasks,
) -> ScoreResponse:
    """Score an incoming transaction against registered XGBoost model and Feast features.

    Persists the transaction and resulting score to PostgreSQL off the critical request path.
    """
    try:
        response = scoring_engine.score(payload)

        # Off-request-path persistence so database I/O does not penalize scoring latency
        background_tasks.add_task(
            db_manager.save_transaction_and_score,
            txn_id=payload.txn_id,
            account_id=payload.nameOrig,
            dest_id=payload.nameDest,
            amount=payload.amount,
            txn_type=payload.type,
            ts=None,
            raw_payload=payload.model_dump(),
            xgb_score=response.xgb_score,
            gnn_score=response.gnn_score,
            blended_score=response.blend_score,
            model_version=response.model_version,
            latency_ms=response.latency_ms,
        )

        metrics_tracker.record_request(
            latency_ms=response.latency_ms,
            feature_fallback=response.feature_fallback,
            gnn_fallback=not response.blend_used,
        )
        return response

    except Exception as exc:
        metrics_tracker.record_error()
        logger.error(
            "Internal scoring error for txn %s: %s",
            payload.txn_id,
            exc,
            exc_info=True,
            extra={"txn_id": payload.txn_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring error: {str(exc)}",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check reporting status of individual subsystems",
)
async def health_check() -> HealthResponse:
    """Report individual health status of model, Feast store, and database."""
    model_ok = scoring_engine.is_model_loaded
    feast_ok = scoring_engine.is_feast_reachable
    db_ok = db_manager.is_reachable()

    if model_ok and feast_ok and db_ok:
        svc_status = "healthy"
    elif model_ok:
        svc_status = "degraded"
    else:
        svc_status = "unhealthy"

    return HealthResponse(
        status=svc_status,
        model_loaded=model_ok,
        feast_reachable=feast_ok,
        db_reachable=db_ok,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Operational metrics including rolling p50/p95 latency and throughput",
)
async def get_metrics() -> MetricsResponse:
    """Return live operational telemetry and latency percentiles."""
    return metrics_tracker.get_metrics()


def verify_admin_key(
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> None:
    """Verify administrative API key using constant-time comparison.

    In production (or whenever ADMIN_API_KEY is configured), enforces X-Admin-Key validation.
    In development mode without ADMIN_API_KEY, allows operation with an audit log warning.
    """
    admin_key = config_module.settings.ADMIN_API_KEY
    is_prod = config_module.settings.ENV.lower() in ("production", "prod")

    if is_prod and not admin_key:
        logger.error("ADMIN_API_KEY must be configured in production!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server security misconfiguration: ADMIN_API_KEY not set",
        )

    if admin_key:
        if not x_admin_key or not secrets.compare_digest(x_admin_key, admin_key):
            logger.warning("Unauthorized administrative access attempt.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Missing or invalid X-Admin-Key header",
            )
    elif is_prod:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Admin access disabled without API key",
        )
    else:
        logger.warning(
            "Administrative operation invoked without ADMIN_API_KEY configured (dev-only)."
        )


@app.post(
    "/admin/reload-model",
    status_code=status.HTTP_200_OK,
    summary="Reload model and preprocessor from latest champion (admin-only)",
    dependencies=[Depends(verify_admin_key)],
)
async def reload_model() -> Dict[str, Any]:
    """Reload the registered champion model and preprocessor without service restart."""
    try:
        scoring_engine.reload()
        return {
            "status": "reloaded",
            "model_version": scoring_engine.model_version,
            "model_loaded": scoring_engine.is_model_loaded,
            "reloaded_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("Failed to reload model: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {e}",
        )


@app.post(
    "/admin/retrain",
    status_code=status.HTTP_200_OK,
    summary="Trigger model retraining condition check (admin-only)",
    dependencies=[Depends(verify_admin_key)],
)
async def trigger_retrain() -> Dict[str, Any]:
    """Manually trigger retraining evaluation conditions (drift & analyst labels)."""
    try:
        from src.monitoring.retrain_trigger import check_retrain_trigger

        status_result = check_retrain_trigger()
        return {
            "status": "evaluated",
            "trigger_fired": status_result.get("trigger_fired", False),
            "reason": status_result.get("reason", "unknown"),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("Failed to evaluate retrain trigger: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrain trigger evaluation failed: {e}",
        )

