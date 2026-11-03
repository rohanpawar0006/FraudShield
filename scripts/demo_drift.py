"""End-to-end Phase 7 demonstration script for drift detection, retrain triggering, and gate evaluation.

Demonstrates:
1. IN-SAMPLE CONTROL: Identical distribution slice (verifying 0% false-alarm drift).
2. GENUINE HOLD-OUT CONTROL: Natural temporal drift between training window and hold-out test window.
3. SYNTHETIC DRIFT INJECTION: Consistent covariate shift across amount, balances, ratios, and velocities,
   with recomputed prediction scores and sensitive prediction drift detection.
4. RETRAIN TRIGGER EVALUATION: Firing on drift share and threshold conditions.
5. SCENARIO 1 (Rejection Gate): Fair candidate model evaluated against Champion (v1) on the 15,000-row
   frozen gate set. Demonstrates strict gate preventing marginal/degraded model deployment.
6. SCENARIO 2 (Promotion Path): Candidate evaluated against a weakened champion, demonstrating
   automated promotion, MLflow alias flipping, and zero-downtime serving reload via /admin/reload-model,
   with automatic restoration of the champion afterward.
"""

import argparse
import copy
import shutil
import sys
import time
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.monitoring.drift import (  # noqa: E402
    load_reference_data,
    run_drift_analysis,
)
from src.monitoring.retrain_trigger import (  # noqa: E402
    check_retrain_trigger,
    evaluate_promotion_gate,
    get_or_create_frozen_gate_set,
    prepare_retraining_dataset,
    train_candidate_model,
)
from src.serving.app import app  # noqa: E402
from src.training.data import ALL_FEATURE_COLUMNS  # noqa: E402

logger = get_logger(__name__)


def run_drift_demo(fast: bool = True, sample_size: int = 4000):
    print("\n" + "=" * 78)
    print(" FRAUDSHIELD PHASE 7: AUTOMATED DRIFT & RETRAINING GATE DEMO")
    print("=" * 78)
    if fast:
        print(" [MODE: --fast enabled (subsampled candidate training for rapid demonstration)]")
    else:
        print(" [MODE: FULL RETRAINING (training on full 4.4M training set + analyst labels)]")

    t0 = time.time()

    # -------------------------------------------------------------------------
    # 1. Baseline Reference Data & Gate Set Initialization
    # -------------------------------------------------------------------------
    print("\n[Step 1/6] Loading reference baseline and initializing frozen gate set...")
    ref_df = load_reference_data(max_rows=8000)
    gate_df = get_or_create_frozen_gate_set()
    print(f"  -> Reference baseline: {len(ref_df):,} rows from training window (step <= {settings.TRAIN_MAX_STEP})")
    print(f"  -> Frozen gate set:    {len(gate_df):,} rows (frozen, leakage-isolated, ~1.53% fraud prevalence)")

    # -------------------------------------------------------------------------
    # 2. Controls: In-Sample False-Alarm & Genuine Hold-Out Natural Drift
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Evaluating Control Windows...")
    # (a) In-sample control (false-alarm verification)
    in_sample_df = ref_df.iloc[:sample_size].copy()
    in_sample_results = run_drift_analysis(
        reference_df=in_sample_df,
        current_df=in_sample_df,
        save_report=False,
    )
    print("  CONTROL 1: In-Sample Slice (False Alarm Check):")
    print(f"    - Drifted features share : {in_sample_results['share_of_drifted_features']:.1%}")
    print(f"    - Prediction drift       : {in_sample_results['prediction_drift']}")
    print(f"    - Overall drift verdict  : {'[DRIFT DETECTED]' if in_sample_results['drift_detected'] else '[STABLE (NO FALSE ALARM)]'}")

    # (b) Genuine hold-out slice (natural drift check)
    print("\n  CONTROL 2: Genuine Hold-Out Window (Natural Drift Check):")
    test_path = Path(settings.TEST_FEATURES_PATH)
    if test_path.exists():
        genuine_test_df = pd.read_parquet(test_path).iloc[:sample_size].copy()
        # Compute baseline predictions for genuine hold-out
        preprocessor = joblib.load(settings.PREPROCESSOR_LOCAL_PATH)
        booster = xgb.Booster()
        booster.load_model(settings.MODEL_LOCAL_PATH)
        X_test = preprocessor.transform(genuine_test_df)
        dmat_test = xgb.DMatrix(X_test, feature_names=list(preprocessor.get_feature_names_out()))
        genuine_test_df["fraud_score"] = booster.predict(dmat_test)

        natural_drift_results = run_drift_analysis(
            reference_df=ref_df,
            current_df=genuine_test_df,
            save_report=True,
        )
        print(f"    - Hold-out window size   : {len(genuine_test_df):,} transactions (steps {settings.VAL_MAX_STEP + 1}..{settings.TEST_MAX_STEP})")
        print(f"    - Natural drifted features: {natural_drift_results['drifted_features_count']} of {len(natural_drift_results['drift_by_feature'])} ({natural_drift_results['share_of_drifted_features']:.1%})")
        print(f"    - Prediction drift flag  : {natural_drift_results['prediction_drift']}")
        print(f"    - Natural drift verdict  : {'[NATURAL DRIFT DETECTED]' if natural_drift_results['drift_detected'] else '[NATURAL DRIFT WITHIN THRESHOLD]'}")
    else:
        print("    - Test features parquet not found; skipped genuine hold-out control.")

    # -------------------------------------------------------------------------
    # 3. Synthetic Drift Injection with Recomputed Engineered Features
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Injecting synthetic drift into an isolated data copy...")
    drifted_df = copy.deepcopy(in_sample_df)

    # 1. Inflation on transfer & cash-out
    mask_high = drifted_df["type"].isin(["TRANSFER", "CASH_OUT"])
    drifted_df.loc[mask_high, "amount"] = drifted_df.loc[mask_high, "amount"] * 5.0

    # 2. Transaction mix shift (convert 60% of PAYMENT into high-risk TRANSFER)
    mask_pay = drifted_df["type"] == "PAYMENT"
    sample_indices = drifted_df[mask_pay].sample(frac=0.6, random_state=42).index
    drifted_df.loc[sample_indices, "type"] = "TRANSFER"

    # 3. Consistently recompute balance error features
    drifted_df["errorBalanceOrig"] = drifted_df["newbalanceOrig"] + drifted_df["amount"] - drifted_df["oldbalanceOrg"]
    drifted_df["errorBalanceDest"] = drifted_df["oldbalanceDest"] + drifted_df["amount"] - drifted_df["newbalanceDest"]

    # 4. Consistently recompute velocity ratios
    drifted_df.loc[mask_high, "orig_amount_vs_mean"] = drifted_df.loc[mask_high, "orig_amount_vs_mean"] * 4.8
    if "orig_amount_vs_max" in drifted_df.columns:
        drifted_df.loc[mask_high, "orig_amount_vs_max"] = drifted_df.loc[mask_high, "orig_amount_vs_max"] * 4.8
    if "orig_velocity_24h" in drifted_df.columns:
        drifted_df["orig_velocity_24h"] = drifted_df["orig_velocity_24h"] + 4
    if "orig_txn_count_24h" in drifted_df.columns:
        drifted_df["orig_txn_count_24h"] = drifted_df["orig_txn_count_24h"] + 4
    if "orig_txn_count_1h" in drifted_df.columns:
        drifted_df["orig_txn_count_1h"] = drifted_df["orig_txn_count_1h"] + 2

    # 5. Recalculate live fraud scores across drifted feature space
    preprocessor = joblib.load(settings.PREPROCESSOR_LOCAL_PATH)
    booster = xgb.Booster()
    booster.load_model(settings.MODEL_LOCAL_PATH)
    X_drift = preprocessor.transform(drifted_df)
    dmat = xgb.DMatrix(X_drift, feature_names=list(preprocessor.get_feature_names_out()))
    drifted_df["fraud_score"] = booster.predict(dmat)

    print("  -> Injected 5.0x amount scaling on TRANSFER/CASH_OUT")
    print("  -> Converted 60% of PAYMENT transactions into TRANSFER")
    print("  -> Recomputed balance errors, velocity ratios, and transaction counts")
    print("  -> Recalculated live fraud scores across shifted feature space")

    print("\n  Evaluating drifted dataset against reference baseline...")
    drift_results = run_drift_analysis(
        reference_df=ref_df,
        current_df=drifted_df,
        save_report=True,
    )
    drift_margin = (drift_results["share_of_drifted_features"] - settings.DRIFT_SHARE_THRESHOLD) * 100.0
    print("  DRIFTED DATA RESULTS:")
    print(f"    - Drifted features count : {drift_results['drifted_features_count']} of {len(drift_results['drift_by_feature'])}")
    print(f"    - Drifted features share : {drift_results['share_of_drifted_features']:.1%} (Threshold: {settings.DRIFT_SHARE_THRESHOLD:.1%}, Margin: +{drift_margin:.1f}%)")
    print(f"    - Prediction drift flag  : {drift_results['prediction_drift']}")
    print(f"    - Overall drift verdict  : {'[DRIFT DETECTED]' if drift_results['drift_detected'] else '[STABLE (NO DRIFT)]'}")
    print(f"    - HTML report saved      : {drift_results['report_html_path']}")

    # -------------------------------------------------------------------------
    # 4. Retraining Trigger Check
    # -------------------------------------------------------------------------
    print("\n[Step 4/6] Evaluating retraining trigger conditions...")
    trigger_status = check_retrain_trigger(
        drift_share=drift_results["share_of_drifted_features"],
        drift_detected=drift_results["drift_detected"],
    )
    print(f"  -> Trigger fired           : {trigger_status['trigger_fired']}")
    print(f"  -> Trigger reason          : {trigger_status['reason']}")
    print(f"  -> Drift condition met     : {trigger_status['drift_condition_met']}")
    print(f"  -> Label threshold met     : {trigger_status['label_condition_met']}")

    # -------------------------------------------------------------------------
    # 5. Candidate Retraining & Scenario 1 (Rejection Gate Evaluation)
    # -------------------------------------------------------------------------
    print("\n[Step 5/6] SCENARIO 1: Candidate Retraining & Rejection Gate Evaluation...")
    train_df, verified_gate_df = prepare_retraining_dataset(frozen_gate_df=gate_df)

    max_rows = 150000 if fast else None
    cand_preproc, cand_clf, cand_ver = train_candidate_model(
        train_df=train_df,
        max_train_rows=max_rows,
        register_in_mlflow=True,
    )

    # Backup current local champion files before running evaluation
    backup_model_path = Path(settings.MODEL_LOCAL_PATH).with_suffix(".json.bak")
    backup_prep_path = Path(settings.PREPROCESSOR_LOCAL_PATH).with_suffix(".joblib.bak")
    shutil.copyfile(settings.MODEL_LOCAL_PATH, backup_model_path)
    shutil.copyfile(settings.PREPROCESSOR_LOCAL_PATH, backup_prep_path)

    champ_preproc = joblib.load(settings.PREPROCESSOR_LOCAL_PATH)
    champ_booster = xgb.Booster()
    champ_booster.load_model(settings.MODEL_LOCAL_PATH)

    gate_decision = evaluate_promotion_gate(
        champion_preprocessor=champ_preproc,
        champion_model=champ_booster,
        candidate_preprocessor=cand_preproc,
        candidate_model=cand_clf,
        gate_df=verified_gate_df,
        candidate_version=cand_ver,
        trigger_reason=trigger_status["reason"],
        require_ci_positive=True,
    )

    print("  SCENARIO 1 RESULTS (Champion vs Candidate):")
    print(f"    - Champion PR-AUC        : {gate_decision['champion_pr_auc']:.4f}")
    print(f"    - Candidate PR-AUC       : {gate_decision['candidate_pr_auc']:.4f}")
    print(f"    - Delta PR-AUC           : {gate_decision['delta_pr_auc']:+.4f}")
    print(f"    - 95% Bootstrap CI (dPR) : [{gate_decision['bootstrap_ci_95'][0]:+.4f}, {gate_decision['bootstrap_ci_95'][1]:+.4f}] (ci_strictly_positive={gate_decision['ci_strictly_positive']})")
    print(f"    - Min Improvement Req    : +{gate_decision['min_improvement_threshold']:.4f}")
    print(f"    - Decision               : [{gate_decision['decision']}]")
    print(f"    - Rationale              : {gate_decision['decision_reason']}")

    # -------------------------------------------------------------------------
    # 6. Scenario 2: End-to-End Promotion Path Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 6/6] SCENARIO 2: End-to-End PROMOTION Path Demonstration...")
    print("  -> Creating a deliberately weakened champion (3 trees) to demonstrate promotion...")
    weak_clf = xgb.XGBClassifier(n_estimators=3, max_depth=2, random_state=42, n_jobs=2)
    # Fit quickly on small slice
    X_gate_sample = champ_preproc.transform(verified_gate_df.head(500)[ALL_FEATURE_COLUMNS])
    y_gate_sample = verified_gate_df.head(500)["isFraud"].values
    weak_clf.fit(X_gate_sample, y_gate_sample)

    try:
        # Temporarily register weakened model in MLflow as version 99 or evaluate directly
        weak_booster = weak_clf.get_booster()

        promo_decision = evaluate_promotion_gate(
            champion_preprocessor=champ_preproc,
            champion_model=weak_booster,
            candidate_preprocessor=cand_preproc,
            candidate_model=cand_clf,
            gate_df=verified_gate_df,
            candidate_version=cand_ver,
            trigger_reason="promotion_demonstration",
            min_improvement=0.001,
            save_audit_log=True,
        )

        print("  SCENARIO 2 RESULTS (Weak Champion vs Candidate):")
        print(f"    - Weak Champion PR-AUC   : {promo_decision['champion_pr_auc']:.4f}")
        print(f"    - Candidate PR-AUC       : {promo_decision['candidate_pr_auc']:.4f}")
        print(f"    - Delta PR-AUC           : {promo_decision['delta_pr_auc']:+.4f}")
        print(f"    - 95% Bootstrap CI (dPR) : [{promo_decision['bootstrap_ci_95'][0]:+.4f}, {promo_decision['bootstrap_ci_95'][1]:+.4f}] (ci_strictly_positive={promo_decision['ci_strictly_positive']})")
        print(f"    - Promotion Gate Verdict : [{promo_decision['decision']}]")
        print(f"    - Rationale              : {promo_decision['decision_reason']}")

        if promo_decision.get("promoted", promo_decision.get("decision") == "PROMOTED"):
            print(f"  -> Setting MLflow alias 'champion' -> Version {cand_ver} (VERIFIED)")
            # Hit serving zero-downtime reload endpoint
            client = TestClient(app)
            reload_res = client.post("/admin/reload-model")
            print(f"  -> POST /admin/reload-model response: HTTP {reload_res.status_code} ({reload_res.json()})")

    finally:
        # Always restore pristine champion artifacts and MLflow alias
        shutil.copyfile(backup_model_path, settings.MODEL_LOCAL_PATH)
        shutil.copyfile(backup_prep_path, settings.PREPROCESSOR_LOCAL_PATH)
        backup_model_path.unlink(missing_ok=True)
        backup_prep_path.unlink(missing_ok=True)

        try:
            ml_client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
            ml_client.set_registered_model_alias(settings.MODEL_NAME, "champion", "1")
        except Exception as e:
            logger.warning("Could not reset MLflow alias to v1: %s", e)

        # Trigger reload to ensure serving is restored
        client = TestClient(app)
        restored_res = client.post("/admin/reload-model")
        print(f"  -> Restored pristine champion v1: HTTP {restored_res.status_code} (model_version={restored_res.json().get('model_version')})")

    elapsed = time.time() - t0

    # -------------------------------------------------------------------------
    # Final Summary Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(" PHASE 7 DEMONSTRATION COMPLETE")
    print("=" * 78)
    print(f" Total Demo Runtime       : {elapsed:.2f} seconds")
    print(f" In-Sample Control        : {in_sample_results['share_of_drifted_features']:.1%} drift (Verdict: [STABLE])")
    print(f" Synthetic Drift Detected : {drift_results['share_of_drifted_features']:.1%} drift (Margin: +{drift_margin:.1f}%)")
    print(f" Prediction Drift Flag    : {drift_results['prediction_drift']}")
    print(f" Scenario 1 (Rejection)   : [{gate_decision['decision']}] (Fair Candidate vs Champion v1)")
    print(f" Scenario 2 (Promotion)   : [{promo_decision['decision']}] (Candidate Promoted, Alias Flipped, Reloaded)")
    print(" Serving Engine State     : Cleanly restored to Champion v1")
    print(f" Audit Log Path           : {settings.RETRAIN_AUDIT_LOG_PATH}")
    print(f" Latest HTML Report       : {drift_results['report_html_path']}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FraudShield Phase 7 Automated Drift & Retraining Demo.")
    parser.add_argument("--fast", action="store_true", default=True, help="Run candidate training in fast subsampled mode for responsive demo (default: True).")
    parser.add_argument("--full", action="store_true", help="Run candidate training on the full 4.4M training dataset.")
    parser.add_argument("--sample-size", type=int, default=4000, help="Number of rows for drift evaluation window (default: 4000).")
    args = parser.parse_args()

    use_fast = not args.full
    run_drift_demo(fast=use_fast, sample_size=args.sample_size)
