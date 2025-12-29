from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def runs_dir(self) -> Path:
        return self.models_dir / "runs"

    @property
    def registry_dir(self) -> Path:
        return self.models_dir / "registry"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @staticmethod
    def from_repo_root() -> "Paths":
        return Paths(root=Path(__file__).resolve().parents[2])


Task = Literal["classification", "regression"]
SplitStrategy = Literal["random", "time", "group"]
ThresholdStrategy = Literal["fixed", "max_f1"]


@dataclass(frozen=True)
class TrainConfig:
    features_path: Path
    target: str
    task: Task = "classification"
    split_strategy: SplitStrategy = "random"

    # columns
    id_cols: tuple[str, ...] = ("user_id",)
    time_col: str | None = None
    group_col: str | None = None

    # splitting
    test_size: float = 0.2
    seed: int = 42

    # classification thresholding
    threshold_strategy: ThresholdStrategy = "fixed"
    threshold_value: float = 0.5


@dataclass(frozen=True)
class PredictConfig:
    run_dir: Path
    input_path: Path
    output_path: Path
    threshold: float | None = None
