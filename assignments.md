# Assignments — Week 3 (5 days)

Goal: finish minimum training + prediction flow by Day 4.

## Day 1 — Data contract + CLI skeleton
Minimum outcomes:
- [ ] `ml-baseline --help` works
- [ ] `make-sample-data` writes `data/processed/features.csv`
- [ ] Data contract written down in `reports/model_card.md` (draft)

Stretch:
- [ ] Support Parquet if `pyarrow` installed

## Day 2 — Train + baseline
Minimum outcomes:
- [ ] `train` loads feature table and runs a holdout split
- [ ] Baseline (DummyClassifier/Regressor) metrics saved
- [ ] Trained Pipeline saved to `models/runs/<run_id>/model/`

Stretch:
- [ ] Add cross-validation on training split

## Day 3 — Evaluate + artifacts
Minimum outcomes:
- [ ] Holdout metrics computed and saved
- [ ] Holdout predictions table saved
- [ ] Input schema contract saved

Stretch:
- [ ] Add threshold selection for classification

## Day 4 — Predict CLI + skew check
Minimum outcomes:
- [ ] `predict` loads saved model + schema and scores a file
- [ ] Run the predict CLI on `holdout_input` and ensure it works

Stretch:
- [ ] Add helpful errors for missing columns / wrong dtypes

## Day 5 — Write-up + submit
Minimum outcomes:
- [ ] Update `reports/model_card.md`
- [ ] Update `reports/eval_summary.md`
- [ ] `pytest` and `ruff` clean
- [ ] Push to GitHub (public)
