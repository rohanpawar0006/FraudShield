"""Feast train-serve parity test suite.

Verifies zero skew between offline historical features (Parquet batch source)
and online serving retrieval (Feast online store get_online_features).
"""

import shutil
from datetime import timedelta

import pandas as pd
import pytest
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

from src.features.feature_pipeline import compute_features


@pytest.fixture(scope="module")
def parity_environment(tmp_path_factory):
    """Set up an isolated Feast feature store environment for parity verification."""
    temp_dir = tmp_path_factory.mktemp("feast_parity_repo")
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create feature_store.yaml
    yaml_config = f"""project: parity_test_project
registry: {str((data_dir / "registry.db").as_posix())}
provider: local
online_store:
  type: sqlite
  path: {str((data_dir / "online_store.db").as_posix())}
offline_store:
  type: file
"""
    (temp_dir / "feature_store.yaml").write_text(yaml_config, encoding="utf-8")

    # 2. Build deterministic sample transactions across 5 hours
    sample_raw = pd.DataFrame(
        {
            "step": [1, 1, 2, 3, 4, 5],
            "nameOrig": ["C_ACC1", "C_ACC2", "C_ACC1", "C_ACC3", "C_ACC1", "C_ACC2"],
            "nameDest": [
                "M_MERCH1",
                "M_MERCH2",
                "M_MERCH1",
                "M_MERCH1",
                "M_MERCH2",
                "M_MERCH1",
            ],
            "amount": [150.0, 50.0, 300.0, 75.0, 450.0, 120.0],
            "isFraud": [0, 0, 0, 0, 0, 0],
        }
    )

    # Compute point-in-time features offline
    offline_features_df = compute_features(sample_raw)
    parquet_path = data_dir / "offline_features.parquet"
    offline_features_df.to_parquet(parquet_path, index=False, engine="pyarrow")

    # 3. Define Feast objects
    account_entity = Entity(
        name="nameOrig",
        join_keys=["nameOrig"],
        value_type=ValueType.STRING,
        description="Origin account ID",
    )

    file_source = FileSource(
        name="parity_offline_source",
        path=str(parquet_path.resolve()),
        timestamp_field="event_timestamp",
    )

    account_fv = FeatureView(
        name="account_features",
        entities=[account_entity],
        ttl=timedelta(days=30),
        schema=[
            Field(name="orig_txn_count_1h", dtype=Int64),
            Field(name="orig_txn_count_24h", dtype=Int64),
            Field(name="orig_amount_vs_mean", dtype=Float64),
            Field(name="orig_time_since_last_txn", dtype=Float64),
            Field(name="orig_has_prev_txn", dtype=Int64),
        ],
        online=True,
        source=file_source,
    )

    # 4. Initialize FeatureStore and apply definitions
    fs = FeatureStore(repo_path=str(temp_dir))
    fs.apply([account_entity, account_fv])

    # 5. Materialize to SQLite online store up to step 5 (cutoff timestamp)
    end_date = offline_features_df["event_timestamp"].max().to_pydatetime()
    start_date = offline_features_df[
        "event_timestamp"
    ].min().to_pydatetime() - timedelta(hours=1)
    fs.materialize(start_date=start_date, end_date=end_date + timedelta(seconds=1))

    yield fs, offline_features_df

    # Teardown
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_feast_online_offline_parity(parity_environment) -> None:
    """Verify that values retrieved from Feast online store match offline features exactly."""
    fs, offline_df = parity_environment

    # Target entities to verify
    test_entities = ["C_ACC1", "C_ACC2", "C_ACC3"]

    # Retrieve from online store
    online_response = fs.get_online_features(
        features=[
            "account_features:orig_txn_count_1h",
            "account_features:orig_txn_count_24h",
            "account_features:orig_amount_vs_mean",
            "account_features:orig_time_since_last_txn",
            "account_features:orig_has_prev_txn",
        ],
        entity_rows=[{"nameOrig": acc} for acc in test_entities],
    ).to_dict()

    online_df = pd.DataFrame(online_response)

    # For each entity, compare against the latest offline record
    for acc in test_entities:
        offline_record = offline_df[offline_df["nameOrig"] == acc].iloc[-1]
        online_record = online_df[online_df["nameOrig"] == acc].iloc[0]

        # Exact parity assertions
        assert online_record["orig_txn_count_1h"] == offline_record["orig_txn_count_1h"]
        assert (
            online_record["orig_txn_count_24h"] == offline_record["orig_txn_count_24h"]
        )
        assert online_record["orig_has_prev_txn"] == offline_record["orig_has_prev_txn"]
        assert online_record["orig_amount_vs_mean"] == pytest.approx(
            offline_record["orig_amount_vs_mean"], rel=1e-4
        )
        assert online_record["orig_time_since_last_txn"] == pytest.approx(
            offline_record["orig_time_since_last_txn"], rel=1e-4
        )


def test_feast_unseen_entity_handling(parity_environment) -> None:
    """Verify that querying Feast online store for an unseen entity returns None safely."""
    fs, _ = parity_environment

    unseen_response = fs.get_online_features(
        features=[
            "account_features:orig_txn_count_1h",
            "account_features:orig_amount_vs_mean",
        ],
        entity_rows=[{"nameOrig": "C_NONEXISTENT_ACCOUNT"}],
    ).to_dict()

    # Feast must return None for unmaterialized/unseen entities without throwing
    assert unseen_response["orig_txn_count_1h"][0] is None
    assert unseen_response["orig_amount_vs_mean"][0] is None
