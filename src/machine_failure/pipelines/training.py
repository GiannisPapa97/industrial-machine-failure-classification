"""The training pipeline: load, split, and fit one model on the training rows."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from machine_failure.config import TrainingConfig
from machine_failure.models import FinalModel, get_model, save_final_model
from machine_failure.pipelines.splits import load_split

logger = logging.getLogger(__name__)


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

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> TrainingResult:
        """Fit the model on ``X_train`` (training rows only), with the threshold."""
        logger.info(
            "fitting %s on %s training rows", self.config.model, f"{len(X_train):,}"
        )
        fitted = self.build_model().fit(X_train, y_train)
        final = FinalModel(self.config.model, fitted, self.config.decision_threshold)
        return TrainingResult(final, self.config, len(X_train), int(y_train.sum()))

    def run(self) -> TrainingResult:
        """Load the data, split it and fit the model on the training rows."""
        return self.fit(*self.load_training_data())

    def save(self, result: TrainingResult, *, overwrite: bool = False) -> Path:
        """Save the trained model to the configured path.

        An existing file is never replaced unless ``overwrite`` is True.
        """
        return save_final_model(
            result.final, self.config.output_path, overwrite=overwrite
        )
