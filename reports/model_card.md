# Model Card — Week 3 Baseline

## Problem
- Predict: is_high_value for one row per user
- Decision enabled: identify high-value users for retention offers or targeted campaigns
- Constraints: CPU-only; offline-first; batch inference

## Data (contract)
- Feature table: data/processed/features.csv
- Unit of analysis: one row per user
- Target column: is_high_value, positive class: 1 (high-value user)
- Optional IDs (passthrough): user_id

## Splits (draft for now)
- Holdout strategy: random stratified split (default)
- Leakage risks: using post-outcome information such as future orders or revenue-derived features beyond the prediction point

## Metrics (draft for now)
- Primary: ROC AUC (captures ranking quality and is robust to class imbalance)
- Baseline: dummy classifier (most_frequent) must be reported

## Shipping
- Artifacts: trained model, input schema, evaluation metrics, holdout input and predictions tables, run metadata
- Known limitations: baseline model is simple and may underfit; performance may be unstable due to small dataset size
- Monitoring sketch: track prediction distribution, class balance drift, ROC AUC over time, and missing or unexpected feature values
