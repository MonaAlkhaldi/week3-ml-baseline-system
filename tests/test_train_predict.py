from pathlib import Path

from ml_baseline.config import Paths, PredictConfig, TrainConfig
from ml_baseline.sample_data import make_sample_feature_table
from ml_baseline.train import run_train
from ml_baseline.predict import run_predict, resolve_run_dir


def test_train_and_predict(tmp_path: Path) -> None:
    # Create a temp repo-like root
    root = tmp_path
    (root / "data" / "processed").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    (root / "outputs").mkdir(parents=True)

    features_path = make_sample_feature_table(root=root, n_users=30)
    cfg = TrainConfig(features_path=features_path, target="is_high_value")
    run_dir = run_train(cfg, root=root)

    assert (run_dir / "model" / "model.joblib").exists()
    assert (run_dir / "schema" / "input_schema.json").exists()
    assert (run_dir / "metrics" / "holdout_metrics.json").exists()

    # Predict using holdout_input
    # Determine ext used by sample generator
    holdout_input = next((run_dir / "tables").glob("holdout_input.*"))
    out_path = root / "outputs" / "preds.csv"
    pcfg = PredictConfig(run_dir=run_dir, input_path=holdout_input, output_path=out_path)
    run_predict(pcfg)

    assert out_path.exists()
