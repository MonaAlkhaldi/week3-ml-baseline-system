from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

from .config import PredictConfig
from .io import read_tabular, write_tabular
from .schema import InputSchema, validate_and_align

log = logging.getLogger(__name__)


def resolve_run_dir(run: str, *, models_dir: Path) -> Path:
    if run == "latest":
        p = models_dir / "registry" / "latest.txt"
        if not p.exists():
            raise FileNotFoundError("No latest.txt found. Train a model first.")
        run_id = p.read_text(encoding="utf-8").strip()
        return models_dir / "runs" / run_id
    return Path(run).resolve()


def run_predict(cfg: PredictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    schema_path = cfg.run_dir / "schema" / "input_schema.json"
    model_path = cfg.run_dir / "model" / "model.joblib"

    schema = InputSchema.load(schema_path)
    model = joblib.load(model_path)

    df_in = read_tabular(cfg.input_path)
    X, ids = validate_and_align(df_in, schema)

    # Try classification proba first
    out: pd.DataFrame
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        threshold = cfg.threshold if cfg.threshold is not None else 0.5
        out = pd.DataFrame({"score": score, "prediction": (score >= threshold).astype(int)})
    else:
        pred = model.predict(X)
        out = pd.DataFrame({"prediction": pred})

    if len(ids.columns) > 0:
        out = pd.concat([ids.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    write_tabular(out, cfg.output_path)
    log.info("Wrote predictions: %s (%s rows)", cfg.output_path, len(out))
