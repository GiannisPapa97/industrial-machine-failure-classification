"""The training pipeline: load, split, and fit one model on the training rows."""

import dataclasses
import logging
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from machine_failure.config import TrainingConfig
from machine_failure.models import FinalModel, get_model, save_final_model
from machine_failure.pipelines.splits import load_split, split_fingerprint

logger = logging.getLogger(__name__)

PROJECT = "industrial-machine-failure-classification"


@dataclass(frozen=True)
class TrainingResult:
    """What a training run produced: the fitted model and what it was fitted on."""

    final: FinalModel
    config: TrainingConfig
    n_rows: int
    n_failures: int

    def describe(self) -> str:
        """One line saying what was trained, on how many rows, with which threshold."""
        return (
            f"Trained {self.final.name} on the training set ({self.n_rows:,} rows, "
            f"{self.n_failures} failures); threshold {self.final.threshold}."
        )


class TrainingPipeline:
    """Train the model a ``TrainingConfig`` describes on the training rows.

    The held-out test rows are split off and never used here; they are for the
    final evaluation only. ``run`` loads, splits and fits; ``fit`` does only the
    fitting, for data you already have; ``save`` writes the result to the
    configured path.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def load_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """The training rows and their labels, as the config splits them."""
        X_train, _, y_train, _ = load_split(self.config.data)
        return X_train, y_train

    def build_model(self) -> Pipeline:
        """A new, unfitted pipeline for the configured model, tuned or default."""
        model = get_model(self.config.model)
        if self.config.tuned:
            return model.build_tuned(self.config.feature_set)
        return model.build(self.config.feature_set)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        split: dict[str, str] | None = None,
    ) -> TrainingResult:
        """Fit the model on ``X_train`` (training rows only), with the threshold.

        ``split`` identifies the held-out rows (``split_fingerprint``); ``run``
        passes it, and it is saved in the model's metadata.
        """
        logger.info(
            "fitting %s on %s training rows", self.config.model, f"{len(X_train):,}"
        )
        fitted = self.build_model().fit(X_train, y_train)
        n_rows, n_failures = len(X_train), int(y_train.sum())
        final = FinalModel(
            self.config.model,
            fitted,
            self.config.decision_threshold,
            self.metadata(n_rows, n_failures, split),
        )
        return TrainingResult(final, self.config, n_rows, n_failures)

    def run(self) -> TrainingResult:
        """Load the data, split it and fit the model on the training rows."""
        X_train, X_test, y_train, y_test = load_split(self.config.data)
        split = split_fingerprint(X_train, X_test, y_train, y_test)
        return self.fit(X_train, y_train, split=split)

    def metadata(
        self, n_rows: int, n_failures: int, split: dict[str, str] | None
    ) -> dict[str, Any]:
        """How the model is made, to save with it."""
        config = dataclasses.asdict(self.config)
        record: dict[str, Any] = {
            "config": _plain(config),
            "threshold_source": (
                "set in the config"
                if self.config.threshold is not None
                else f"the tuned {self.config.model} model's own"
            ),
            "training_rows": n_rows,
            "training_failures": n_failures,
            "versions": library_versions(),
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if split is not None:
            record["split"] = split
        return record

    def save(self, result: TrainingResult, *, overwrite: bool = False) -> Path:
        """Save the trained model to the configured path.

        An existing file is never replaced unless ``overwrite`` is True.
        """
        return save_final_model(
            result.final, self.config.output_path, overwrite=overwrite
        )


def library_versions() -> dict[str, str]:
    """The versions that matter to reload and reproduce a model."""
    names = ["scikit-learn", "xgboost-cpu", "numpy", "pandas", PROJECT]
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _plain(value: Any) -> Any:
    """``value`` with paths as strings, so the metadata is plain data."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value
