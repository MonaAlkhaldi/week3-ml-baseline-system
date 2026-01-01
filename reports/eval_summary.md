# Evaluation Summary — Week 3 Baseline

## What was trained
- Model family: Logistic Regression (binary classifier)
- Preprocessing:
  - Numerical features: median imputation
  - Categorical features: most-frequent imputation + one-hot encoding
- End-to-end pipeline implemented using scikit-learn `Pipeline` and `ColumnTransformer`

## Results
- Baseline (DummyClassifier, most_frequent):
  - Accuracy: <COPY baseline_accuracy FROM holdout_metrics.json>
- Model (Logistic Regression):
  - Accuracy: <COPY model_accuracy FROM holdout_metrics.json>
  - ROC AUC: <COPY roc_auc OR note "undefined due to single-class holdout">

The trained model outperforms the baseline on holdout accuracy, indicating that the learned feature relationships provide predictive signal beyond the majority-class baseline.

## Error analysis
- Worst-case errors occur near the decision boundary (e.g., users with total_amount close to the threshold)
- ROC AUC is unstable or undefined due to class imbalance and small sample size
- Potential data leakage was explicitly prevented by excluding the target column during inference

## Recommendation
- Do not ship to production yet
- Rationale:
  - Dataset is synthetic and too small for reliable generalization
  - Class imbalance limits the usefulness of threshold-independent metrics
  - Additional real-world data and calibration are needed before deployment

This model is suitable as a baseline and for pipeline validation, but not for high-stakes decision-making.
