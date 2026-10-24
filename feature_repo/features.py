"""Feast Feature Store Definitions for FraudShield.

Defines Entities, Sources, and Feature Views for offline Parquet and online serving.
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

# Locate offline Parquet batch source
REPO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_DIR.parent
PARQUET_SOURCE_PATH = str(
    (PROJECT_ROOT / "data" / "processed" / "train_features.parquet").resolve()
)

# -------------------------------------------------------------------------
# 1. Entities
# -------------------------------------------------------------------------
origin_account = Entity(
    name="nameOrig",
    join_keys=["nameOrig"],
    value_type=ValueType.STRING,
    description="Unique identifier for the origin (payer) account",
)

destination_account = Entity(
    name="nameDest",
    join_keys=["nameDest"],
    value_type=ValueType.STRING,
    description="Unique identifier for the destination (payee/merchant) account",
)

# -------------------------------------------------------------------------
# 2. Batch Data Sources
# -------------------------------------------------------------------------
origin_feature_source = FileSource(
    name="origin_account_features_source",
    path=PARQUET_SOURCE_PATH,
    timestamp_field="event_timestamp",
)

dest_feature_source = FileSource(
    name="dest_account_features_source",
    path=PARQUET_SOURCE_PATH,
    timestamp_field="event_timestamp",
)

# -------------------------------------------------------------------------
# 3. Feature Views
# -------------------------------------------------------------------------
origin_account_features_view = FeatureView(
    name="origin_account_features",
    entities=[origin_account],
    ttl=timedelta(days=30),
    schema=[
        Field(name="orig_txn_count_1h", dtype=Int64),
        Field(name="orig_txn_count_24h", dtype=Int64),
        Field(name="orig_amount_vs_mean", dtype=Float64),
        Field(name="orig_time_since_last_txn", dtype=Float64),
        Field(name="orig_has_prev_txn", dtype=Int64),
    ],
    online=True,
    source=origin_feature_source,
)

destination_features_view = FeatureView(
    name="destination_features",
    entities=[destination_account],
    ttl=timedelta(days=30),
    schema=[
        Field(name="dest_txn_count_24h", dtype=Int64),
        Field(name="dest_has_prev_txn", dtype=Int64),
    ],
    online=True,
    source=dest_feature_source,
)
