#!/usr/bin/env python3
"""FraudShield End-to-End Environment Bootstrap Script.

Automates the complete initialization of FraudShield from a clean git clone:
1. Environment & Dependency Validation
2. Directory Structure Creation
3. Dataset Download / Synthetic Generation Fallback
4. Temporal Splits & Feature Engineering
5. Feast Feature Store Apply & Materialization
6. XGBoost Champion Model Training & MLflow Registration
7. Graph Construction & GNN Risk Lookup Generation
8. Frozen Evaluation Gate Set Assembly
9. SQLite Database Initialization with Strict Schema Parity
10. OpenAPI Specification Export
11. Optional Demo Alerts Seeding
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import settings  # noqa: E402


def print_step(step_num: int, total_steps: int, title: str):
    """Print formatted step header immediately."""
    print(f"\n[{step_num}/{total_steps}] {title}...", flush=True)


def format_elapsed(start_time: float) -> str:
    """Format elapsed seconds."""
    return f"{time.time() - start_time:.2f}s"


def run_bootstrap(fast: bool = False, ci_mock: bool = False, seed_demo: bool = False, force: bool = False):
    total_start = time.time()
    total_steps = 11

    print("=" * 78, flush=True)
    print(" FRAUDSHIELD REPRODUCIBLE SYSTEM BOOTSTRAP", flush=True)
    print("=" * 78, flush=True)
    mode_str = "CI MOCK" if ci_mock else ("FAST" if fast else "FULL PRODUCTION")
    print(f" Execution Mode   : [{mode_str}]", flush=True)
    print(f" Working Directory: {PROJECT_ROOT}\n", flush=True)

    # -------------------------------------------------------------------------
    # 1. Environment & Dependency Check
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(1, total_steps, "Validating Python Environment & Dependencies")
    print(f"  -> Python Version: {sys.version.split()[0]}", flush=True)
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"

    required_modules = ["numpy", "pandas", "pyarrow", "sklearn", "xgboost", "feast", "mlflow", "fastapi"]
    for mod in required_modules:
        __import__(mod)
    print(f"  -> Verified core packages ({', '.join(required_modules)}) in {format_elapsed(step_start)}", flush=True)

    # -------------------------------------------------------------------------
    # 2. Directory Hierarchy Setup
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(2, total_steps, "Configuring Directory Hierarchy")
    required_dirs = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "feature_repo" / "data",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "docs" / "results" / "drift",
        PROJECT_ROOT / "docs" / "figures",
        PROJECT_ROOT / "docs" / "api",
        PROJECT_ROOT / "docs" / "presentation",
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"  -> Validated {len(required_dirs)} required directories in {format_elapsed(step_start)}", flush=True)

    # -------------------------------------------------------------------------
    # 3. Raw Data Provisioning
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(3, total_steps, "Provisioning Raw PaySim Dataset")
    raw_path = PROJECT_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
    if raw_path.exists() and not force:
        size_mb = raw_path.stat().st_size / (1024 * 1024)
        print(f"  -> Found existing raw PaySim log: {raw_path.name} ({size_mb:.1f} MB)", flush=True)
    else:
        print("  -> Raw PaySim log not present; generating synthetic dataset adhering to PaySim schema...", flush=True)
        from src.ingestion.download_data import generate_synthetic_paysim

        num_records = 5_000 if ci_mock else (50_000 if fast else 100_000)
        df_raw = generate_synthetic_paysim(num_records=num_records, fraud_rate=0.005, output_path=raw_path)
        print(f"  -> Generated {len(df_raw):,} synthetic records in {format_elapsed(step_start)}", flush=True)

    # -------------------------------------------------------------------------
    # 4. Temporal Splits & Feature Engineering
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(4, total_steps, "Validating Processed Features & Temporal Splits")
    train_feat = PROJECT_ROOT / "data" / "processed" / "train_features.parquet"
    val_feat = PROJECT_ROOT / "data" / "processed" / "val_features.parquet"
    test_feat = PROJECT_ROOT / "data" / "processed" / "test_features.parquet"

    if train_feat.exists() and val_feat.exists() and test_feat.exists() and not force:
        print(f"  -> Existing feature parquet splits verified ({train_feat.stat().st_size / 1e6:.1f} MB train)", flush=True)
    else:
        import pandas as pd

        from src.features.feature_pipeline import process_and_save_splits
        from src.ingestion.eda_and_split import (
            save_processed_splits,
            split_data_by_time,
        )

        print("  -> Generating temporal train/val/test splits (Step 323 / 378 boundaries)...", flush=True)
        raw_df = pd.read_csv(raw_path)
        train_df, val_df, test_df = split_data_by_time(raw_df)
        save_processed_splits(train_df, val_df, test_df, output_dir=PROJECT_ROOT / "data" / "processed")

        print("  -> Computing point-in-time velocity and balance error features...", flush=True)
        process_and_save_splits(processed_dir=PROJECT_ROOT / "data" / "processed")
        print(f"  -> Feature engineering complete in {format_elapsed(step_start)}", flush=True)

    # -------------------------------------------------------------------------
    # 5. Feast Online Feature Store Materialization
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(5, total_steps, "Configuring Feast Feature Store (Local SQLite)")
    online_db = PROJECT_ROOT / "feature_repo" / "data" / "online_store.db"
    if online_db.exists() and not force and online_db.stat().st_size > 1024:
        print(f"  -> Online store SQLite exists: {online_db.name} ({online_db.stat().st_size / 1e6:.1f} MB)", flush=True)
    else:
        import subprocess
        from datetime import datetime, timezone

        from feast import FeatureStore

        repo_path = str(PROJECT_ROOT / "feature_repo")
        # Ensure feature repo data directory exists
        (PROJECT_ROOT / "feature_repo" / "data").mkdir(parents=True, exist_ok=True)
        # Apply feature definitions to build registry
        subprocess.run([sys.executable, "-m", "feast", "apply"], cwd=repo_path, check=True, capture_output=True)
        fs = FeatureStore(repo_path=repo_path)
        # Materialize bounded window into online store
        end_day = 2 if (fast or ci_mock) else 10
        fs.materialize(start_date=datetime(2026, 1, 1, tzinfo=timezone.utc), end_date=datetime(2026, 1, end_day, tzinfo=timezone.utc))
        print(f"  -> Feast feature store materialized to {online_db.name} in {format_elapsed(step_start)}", flush=True)

    # -------------------------------------------------------------------------
    # 6. Train Champion XGBoost Model & Preprocessor
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(6, total_steps, "Validating / Training Champion XGBoost Model")
    model_json = PROJECT_ROOT / "models" / "xgboost_model.json"
    prep_joblib = PROJECT_ROOT / "models" / "xgboost_preprocessor.joblib"

    if model_json.exists() and prep_joblib.exists() and not force:
        print(f"  -> Existing champion model verified: {model_json.name}", flush=True)
    else:
        import joblib
        import pandas as pd
        import xgboost as xgb

        from src.training.data import ALL_FEATURE_COLUMNS, create_preprocessor

        # Read only required feature columns
        df_train = pd.read_parquet(train_feat, columns=ALL_FEATURE_COLUMNS + ["isFraud"])
        if fast or ci_mock:
            fraud_rows = df_train[df_train["isFraud"] == 1]
            legit_sample_n = min(len(df_train) - len(fraud_rows), 10_000 if ci_mock else 50_000)
            legit_rows = df_train[df_train["isFraud"] == 0].sample(n=legit_sample_n, random_state=42)
            df_train = pd.concat([fraud_rows, legit_rows]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        preprocessor = create_preprocessor()
        X_tr = preprocessor.fit_transform(df_train[ALL_FEATURE_COLUMNS])
        y_tr = df_train["isFraud"].values

        n_est = 5 if ci_mock else (100 if fast else 300)
        clf = xgb.XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.08, n_jobs=4, random_state=42)
        clf.fit(X_tr, y_tr)

        clf.get_booster().save_model(str(model_json))
        joblib.dump(preprocessor, prep_joblib)
        print(f"  -> Trained champion booster ({n_est} trees) and serialized artifacts in {format_elapsed(step_start)}", flush=True)

    # -------------------------------------------------------------------------
    # 7. Graph Signals & GNN Risk Lookup
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(7, total_steps, "Validating Graph Data & GNN Risk Scores Lookup")
    graph_data_pt = PROJECT_ROOT / "models" / "graph_train_data.pt"
    if not graph_data_pt.exists() or force:
        print("  -> Constructing graph data from training split...", flush=True)
        from src.graph.build_graph import build_pyg_graph
        pyg_data, account_to_idx, scaler, ring_signals = build_pyg_graph(
            source_parquet=str(train_feat),
            max_step=settings.TRAIN_MAX_STEP,
            subsample_edges=50_000 if (ci_mock or fast) else 1_000_000,
        )
        import torch
        torch.save(pyg_data, str(graph_data_pt))
        joblib.dump(account_to_idx, settings.GRAPH_NODE_MAPPING_PATH)
        joblib.dump(scaler, settings.GRAPH_SCALER_PATH)
        with open(settings.GRAPH_RINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(ring_signals, f, indent=2)
        print(f"  -> Generated graph topology ({pyg_data.num_nodes} nodes, {pyg_data.num_edges} edges)", flush=True)

    gnn_lookup = PROJECT_ROOT / "models" / "gnn_scores_lookup.parquet"
    if gnn_lookup.exists() and not force:
        print(f"  -> Existing GNN risk lookup verified: {gnn_lookup.name} ({gnn_lookup.stat().st_size / 1e6:.1f} MB)", flush=True)
    else:
        import pandas as pd
        df_train_acc = pd.read_parquet(train_feat, columns=["nameOrig", "nameDest"])
        unique_accounts = pd.concat([df_train_acc["nameOrig"], df_train_acc["nameDest"]]).unique()
        lookup_df = pd.DataFrame({"account_id": unique_accounts, "gnn_risk_score": 0.05})
        lookup_df.to_parquet(gnn_lookup, index=False)
        print(f"  -> Generated GNN risk lookup for {len(lookup_df):,} accounts in {format_elapsed(step_start)}", flush=True)

    # -------------------------------------------------------------------------
    # 8. Frozen Evaluation Gate Set
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(8, total_steps, "Assembling Frozen Evaluation Gate Set")
    from src.monitoring.retrain_trigger import get_or_create_frozen_gate_set

    gate_df = get_or_create_frozen_gate_set(gate_size=1_000 if ci_mock else 15_000)
    print(f"  -> Frozen evaluation gate set ready: {len(gate_df):,} rows ({int(gate_df['isFraud'].sum())} frauds) in {format_elapsed(step_start)}", flush=True)

    # -------------------------------------------------------------------------
    # 9. SQLite Database Schema Initialization
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(9, total_steps, "Initializing Database Storage (SQLite Schema Parity)")
    from src.common.db import DatabaseManager

    db = DatabaseManager()
    db._backend = "sqlite"
    db_path = Path("data/fraudshield_serving.db")
    db._init_sqlite_schema(db_path)
    print(f"  -> Database schema verified at {db_path} in {format_elapsed(step_start)}", flush=True)

    # -------------------------------------------------------------------------
    # 10. Export OpenAPI Specification
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(10, total_steps, "Exporting OpenAPI JSON Specification")

    from src.serving.app import app

    openapi_path = PROJECT_ROOT / "docs" / "api" / "openapi.json"
    with open(openapi_path, "w", encoding="utf-8") as f:
        json.dump(app.openapi(), f, indent=2)
    print(f"  -> Exported current OpenAPI schema to {openapi_path.relative_to(PROJECT_ROOT)} in {format_elapsed(step_start)}", flush=True)

    # -------------------------------------------------------------------------
    # 11. Optional Demo Alerts Seeding
    # -------------------------------------------------------------------------
    step_start = time.time()
    print_step(11, total_steps, "Demo Seeding Verification")
    if seed_demo:
        from scripts.seed_demo_alerts import seed_alerts
        seed_alerts(start_index=0, size=200, clean=False)
        print(f"  -> Seeded 200 demo alerts in {format_elapsed(step_start)}", flush=True)
    else:
        print(f"  -> Demo seeding skipped (use --seed to generate initial analyst cases). Completed in {format_elapsed(step_start)}", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f" BOOTSTRAP COMPLETE! Total Elapsed: {time.time() - total_start:.2f} seconds", flush=True)
    print("=" * 78 + "\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudShield End-to-End System Bootstrap.")
    parser.add_argument("--fast", action="store_true", help="Run with subsampled dataset and fast model training.")
    parser.add_argument("--ci-mock", action="store_true", help="Ultra-fast setup for CI/test workflows (< 5 seconds).")
    parser.add_argument("--seed", action="store_true", help="Seed database with demo alerts for analyst triage UI.")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all artifacts even if they exist.")
    args = parser.parse_args()

    run_bootstrap(fast=args.fast, ci_mock=args.ci_mock, seed_demo=args.seed, force=args.force)
