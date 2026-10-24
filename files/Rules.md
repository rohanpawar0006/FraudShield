# Rules.md — Development Rules (AI agents & developers)

Follow these unless the user explicitly instructs otherwise.

## General Coding
- Python 3.13 compatible [verify PyG/Feast wheel support; fall back to 3.11 in Docker if needed — document in Memory.md].
- Type hints everywhere; `ruff` + `black` clean; functions < ~50 lines.
- Names: `snake_case` functions/files, `PascalCase` classes, `UPPER_CASE` constants.
- Docstrings on every public function; module-level README in each `src/` folder if non-obvious.
- Config via `src/common/config.py` (env vars); no magic numbers in logic.
- Logging via `logging` (structured JSON in services); no `print` in `src/`.
- Tests: pytest; every feature function and API endpoint has at least one test.

## Technology
**MUST USE:** Kafka, XGBoost, PyTorch Geometric, Feast, MLflow, FastAPI, Kubernetes (Minikube), Evidently AI, Pydantic, pytest.
**PREFER:** pandas/pyarrow, scikit-learn metrics, Streamlit, Postgres, Docker.
**DO NOT USE:** random train/test splits; accuracy as the headline metric; notebooks as production code; global mutable state in services; unpinned dependencies.
Never add a dependency without stating why it is necessary and updating `requirements`.

## ML-Specific Rules
1. **Time-based splits only** (train < val < test by timestamp).
2. **Leakage check** before training: document dropped columns (e.g., post-transaction balance fields in PaySim).
3. Headline metrics: **PR-AUC**, recall@precision, confusion matrix at chosen threshold. Never report accuracy alone.
4. Handle imbalance with `scale_pos_weight` first; SMOTE only on training folds, never on val/test.
5. Every training run logged in MLflow: params, metrics, dataset hash, git commit, model artifact.
6. Feature definitions written **once** (`src/features`) and reused by Feast offline and online paths; parity test required.
7. Seeds fixed and logged.
8. A new model is promoted only if it beats the current one on the held-out set.
9. Graph construction must use training-window data only (no future edges).

## Architecture
Follow `Architecture.md`. Keep services separated (ingestion / serving / training / monitoring / dashboard). No duplicate logic; reuse `common/`. No unnecessary abstractions.

## AI Coding Rules
1. Read existing code and docs before changing anything. 2. Check `Phases.md` and `Memory.md` for the current phase. 3. Stay in the current phase. 4. Don't rewrite working code. 5. Don't change API contracts without updating `Architecture.md`. 6. Don't silently change requirements. 7. Don't remove functionality unless told. 8. Test before declaring done. 9. **Never claim something was run/passed if it was not.** 10. Make small, focused commits with clear messages (`feat:`, `fix:`, `docs:`, `test:`).

## Error Handling
Detect via validation + try/except at boundaries; log with context (txn_id, request_id); return structured errors; retry transient Kafka/DB errors with backoff (max 3); route poison messages to dead-letter topic; never swallow exceptions silently.

## Security
Never hardcode secrets or commit `.env`/keys; validate all input; no disabling checks to "make it work"; use K8s Secrets/ConfigMaps for config.

## Change Management
Before a major change state: what changes, why, files affected, risks, docs to update.
