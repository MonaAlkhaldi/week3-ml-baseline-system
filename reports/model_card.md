# Model Card — Week 3 Baseline

## Problem
- Target: is_high_value (binary classification: 1 = high-value user, 0 = not high-value)
- Unit of analysis: Individual user (one row per user)
- Decision enabled: Identify high-value users for prioritization, targeting, or intervention

## Data
- Feature table: data/processed/features.csv
- Dataset hash (sha256): <COPY FROM run_meta.json → dataset_hash>

## Splits
- Holdout strategy: Random stratified split
- Test size: 0.2
- Random seed: 42

## Metrics (holdout)
- Baseline (DummyClassifier, most_frequent):
  - Accuracy: <COPY baseline_accuracy FROM holdout_metrics.json>
- Model (Logistic Regression):
  - Accuracy: <COPY model_accuracy FROM holdout_metrics.json>
  - ROC AUC: <COPY roc_auc IF PRESENT, otherwise note "undefined due to single-class split">

## Limitations
- Dataset is synthetic and small, limiting generalization to real-world data
- Class imbalance may affect metric stability, especially ROC AUC
- Features are limited and do not capture temporal or behavioral dynamics

## Monitoring sketch
- Track prediction class distribution over time
- Monitor accuracy and ROC AUC on fresh labeled data
- Alert if input schema or feature distributions drift

## Reproducibility
- Run id: <RUN_ID>
- Git commit: <COPY git_commit FROM run_meta.json>
- Environment snapshot: models/runs/<RUN_ID>/env/pip_freeze.txt
