from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from .config import PredictConfig
from .io import read_tabular, write_tabular
from .schema import InputSchema, validate_and_align

log = logging.getLogger(__name__)


def resolve_run_dir(run: str, *, models_dir: Path) -> Path:
    """Resolve a run identifier into a run directory.

    - run="latest" -> models/registry/latest.txt -> models/runs/<run_id>
    - otherwise treat `run` as a path.
    """
    if run == "latest":
        p = models_dir / "registry" / "latest.txt"
        if not p.exists():
            raise FileNotFoundError("No latest.txt found. Train a model first.")
        run_id = p.read_text(encoding="utf-8").strip()
        if not run_id:
            raise ValueError("latest.txt is empty. Train a model first.")
        return models_dir / "runs" / run_id

    return Path(run).expanduser().resolve()


def _load_saved_threshold(run_dir: Path) -> float | None:
    """Stretch: use saved threshold from holdout metrics if present."""
    p = run_dir / "metrics" / "holdout_metrics.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        t = d.get("threshold", None)
        return float(t) if t is not None else None
    except Exception:
        return None


def run_predict(cfg: PredictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 1) load schema + model from run folder
    schema_path = cfg.run_dir / "schema" / "input_schema.json"
    model_path = cfg.run_dir / "model" / "model.joblib"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    schema = InputSchema.load(schema_path)
    model = joblib.load(model_path)

    # 2) read input
    df_in = read_tabular(cfg.input_path)

    # 3) validate + align
    X, ids = validate_and_align(df_in, schema)

    # 4) predict
    out: pd.DataFrame
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        # threshold behavior:
        # - if user passes --threshold, use it
        # - else try saved threshold from run artifacts
        # - else default 0.5
        if cfg.threshold is not None:
            t = float(cfg.threshold)
        else:
            t = _load_saved_threshold(cfg.run_dir)
            if t is None:
                t = 0.5

        out = pd.DataFrame({"score": score, "prediction": (score >= t).astype(int)})

    else:
        pred = model.predict(X)
        out = pd.DataFrame({"prediction": pred})

    # attach optional ids if present
    if len(ids.columns) > 0:
        out = pd.concat([ids.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    # 5) write output
    write_tabular(out, cfg.output_path)
    log.info("Wrote predictions: %s (%s rows)", cfg.output_path, len(out))
