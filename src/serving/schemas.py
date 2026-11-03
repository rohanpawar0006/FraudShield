"""Pydantic schemas and contract definitions for FraudShield serving API."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.training.data import PAYMENT_TYPES


class TransactionPayload(BaseModel):
    """Input payload for scoring a financial transaction.

    Validates schema types, non-negative amounts, and recognized payment types.
    """

    model_config = ConfigDict(populate_by_name=True)

    txn_id: str = Field(
        ...,
        alias="transaction_id",
        min_length=1,
        description="Unique transaction identifier",
    )
    step: int = Field(
        default=1,
        ge=1,
        description="Chronological simulation step/hour",
    )
    type: str = Field(
        ...,
        description="Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)",
    )
    amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount in currency units (must be non-negative)",
    )
    nameOrig: str = Field(
        ...,
        alias="account_id",
        min_length=1,
        description="Originating customer account identifier",
    )
    oldbalanceOrg: float = Field(
        default=0.0,
        ge=0.0,
        description="Origin account balance prior to transaction execution",
    )
    nameDest: str = Field(
        ...,
        alias="dest_id",
        min_length=1,
        description="Destination payee/merchant account identifier",
    )
    oldbalanceDest: float = Field(
        default=0.0,
        ge=0.0,
        description="Destination account balance prior to transaction arrival",
    )

    @field_validator("type")
    @classmethod
    def validate_payment_type(cls, v: str) -> str:
        upper_v = v.upper()
        if upper_v not in PAYMENT_TYPES:
            raise ValueError(
                f"Invalid transaction type '{v}'. Allowed types: {PAYMENT_TYPES}"
            )
        return upper_v

    @field_validator("amount")
    @classmethod
    def validate_non_negative_amount(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Transaction amount must be non-negative.")
        return float(v)


class ScoreResponse(BaseModel):
    """Output risk prediction contract for a scored transaction."""

    transaction_id: str = Field(..., description="Unique transaction ID")
    fraud_score: float = Field(
        ..., description="Predicted fraud probability in [0.0, 1.0]"
    )
    is_alert: bool = Field(
        ...,
        description="True if fraud_score meets or exceeds configured decision threshold",
    )
    latency_ms: float = Field(
        ..., description="End-to-end scoring latency in milliseconds"
    )
    feature_fallback: bool = Field(
        ...,
        description="True if Feast lookup missed and safe default features were applied",
    )
    model_version: str = Field(
        ..., description="Model registry version or identifier"
    )
    threshold: float = Field(
        ..., description="Decision threshold applied for alerting"
    )
    xgb_score: float = Field(
        ..., description="Raw tabular XGBoost risk probability"
    )
    gnn_score: Optional[float] = Field(
        default=None, description="Precomputed or online GNN risk score"
    )
    blend_score: float = Field(
        ..., description="Blended fraud score combining XGBoost and GNN"
    )
    blend_used: bool = Field(
        default=False,
        description="True if GNN risk score was found and blended into decision",
    )


class HealthResponse(BaseModel):
    """Health status contract detailing individual subsystem availability."""

    status: str = Field(..., description="Overall service status (healthy, degraded)")
    model_loaded: bool = Field(
        ..., description="Whether XGBoost model is loaded in memory"
    )
    feast_reachable: bool = Field(
        ..., description="Whether Feast feature store is accessible"
    )
    db_reachable: bool = Field(
        ..., description="Whether persistence database backend is accessible"
    )


class MetricsResponse(BaseModel):
    """Operational telemetry contract reporting request volume and latency percentiles."""

    request_count: int = Field(
        ..., description="Total scoring requests processed"
    )
    error_count: int = Field(..., description="Total unhandled or client errors")
    feature_fallback_count: int = Field(
        ..., description="Count of transactions requiring Feast feature defaults"
    )
    gnn_fallback_count: int = Field(
        default=0,
        description="Count of transactions requiring GNN fallback to XGBoost-only",
    )
    throughput_rps: float = Field(
        ..., description="Current request processing throughput (requests/sec)"
    )
    p50_latency_ms: float = Field(
        ..., description="50th percentile (median) latency over rolling window"
    )
    p95_latency_ms: float = Field(
        ..., description="95th percentile latency over rolling window"
    )
    window_size: int = Field(
        ..., description="Number of samples in the rolling latency window"
    )

