# Architecture — Week 3 (Baseline ML System)

## Goal
Given a feature table (`data/processed/features.<csv|parquet>`), build a baseline ML system that is:
- reproducible (run_id + metadata + environment snapshot)
- evaluable (baseline + holdout metrics)
- shippable (saved model + schema + batch predict CLI)

## Tool stack (minimal, high ROI)
Core:
- **pandas** + **numpy** (tabular)
- **scikit-learn** (splits, pipelines, metrics)
- **joblib** (save/load model pipeline)
- **typer** (CLI)
- **pytest** + **ruff** (quality)

Optional (stretch):
- **pyarrow** for Parquet
- **plotly** (+ kaleido) for richer plots
- **shap** for interpretability
- **streamlit** for a tiny demo UI

## Minimum requirements (everyone)

### 1) Training command
`ml-baseline train` must:
- read `data/processed/features.<csv|parquet>`
- define X/y from a target column
- perform a **holdout split** (random stratified for binary classification)
- run at least one **baseline**:
  - classification: DummyClassifier (majority)
  - regression: DummyRegressor (mean)
- train a **scikit-learn Pipeline** (preprocess + model)
- save artifacts under `models/runs/<run_id>/`:
  - `model/model.joblib`
  - `schema/input_schema.json`
  - `metrics/holdout_metrics.json`
  - `tables/holdout_predictions.<csv|parquet>`
  - `tables/holdout_input.<csv|parquet>` (features-only, for predict skew checks)
  - `env/pip_freeze.txt` (best-effort)
  - `run_meta.json` (dataset hash + config + metrics)
- update `models/registry/latest.txt`

### 2) Prediction command
`ml-baseline predict` must:
- load a trained run folder
- validate/align input columns using `schema/input_schema.json`
- write predictions to a file (CSV or Parquet)

### 3) Reporting
- Fill out `reports/model_card.md`
- Fill out `reports/eval_summary.md` with:
  - baseline vs model metrics
  - key caveats and failure modes

## Stretch goals (strong students)
- Add split strategies: time-based and group-based
- Add threshold selection (`max_f1` or `min_precision`) and record it in run_meta
- Add a simple bootstrap CI for one metric
- Add error slices (metrics by segment column)
- Add a small interpretability artifact (permutation importance)

## Non-goals
- Deep learning
- Distributed training
- Heavy experiment tracking systems
