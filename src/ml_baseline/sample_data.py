from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths
from .io import best_effort_ext, write_tabular

def make_sample_feature_table(
    *,
    root: Path | None = None,
    n_users: int = 50,
    seed: int = 42,
) -> Path:
    # Resolve project paths
    paths = Paths.from_repo_root() if root is None else Paths(root=root)
    paths.data_processed_dir.mkdir(parents=True, exist_ok=True)

    # Random generator (reproducible)
    rng = np.random.default_rng(seed)

    # Create base feature table
    df = pd.DataFrame(
        {
            "user_id": [f"u{i:03d}" for i in range(1, n_users + 1)],
            "country": rng.choice(["US", "CA", "GB"], size=n_users),
            "n_orders": rng.integers(1, 10, size=n_users),
        }
    )

    # Derived features
    df["avg_amount"] = rng.normal(10, 3, size=n_users).clip(min=1).round(2)
    df["total_amount"] = (df["n_orders"] * df["avg_amount"]).round(2)
    df["is_high_value"] = (df["total_amount"] >= 80).astype(int)

    # Write outputs
    csv_path = paths.data_processed_dir / "features.csv"
    write_tabular(df, csv_path)


    return csv_path
