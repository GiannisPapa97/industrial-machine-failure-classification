"""The comparison pipeline: cross-validate several models on the training rows."""

from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from machine_failure.config import ComparisonConfig
from machine_failure.evaluation import compare_models, describe_cross_validation
from machine_failure.models import get_model
from machine_failure.pipelines.splits import load_split


@dataclass(frozen=True)
class ComparisonResult:
    """The cross-validated scores of the compared models.

    ``table`` has one row per model and a (metric, mean/std) column for every
    metric, in the order of the config's models.
    """

    table: pd.DataFrame
    config: ComparisonConfig
    n_rows: int
    n_failures: int

    def describe(self) -> str:
        """One line saying how the models were cross-validated."""
        return describe_cross_validation(
            self.n_rows, self.n_failures, self.config.n_repeats, self.config.n_splits
        )


class ComparisonPipeline:
    """Compare the models a ``ComparisonConfig`` names, on the same folds.

    Only the training rows are used; the held-out test rows are never loaded into
    the comparison. ``run`` loads, splits and compares; ``compare`` takes training
    rows you already have.
    """

    def __init__(self, config: ComparisonConfig | None = None) -> None:
        self.config = config or ComparisonConfig()

    def load_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """The training rows and their labels, as the config splits them."""
        X_train, _, y_train, _ = load_split(self.config.data)
        return X_train, y_train

    def build_models(self) -> dict[str, Pipeline]:
        """New, unfitted pipelines with default settings, in the config's order."""
        return {
            name: get_model(name).build(self.config.feature_set)
            for name in self.config.models
        }

    def compare(self, X_train: pd.DataFrame, y_train: pd.Series) -> ComparisonResult:
        """Cross-validate every model on ``X_train`` (training rows only)."""
        table = compare_models(
            self.build_models(),
            X_train,
            y_train,
            n_splits=self.config.n_splits,
            n_repeats=self.config.n_repeats,
            random_state=self.config.random_state,
        )
        return ComparisonResult(table, self.config, len(X_train), int(y_train.sum()))

    def run(self) -> ComparisonResult:
        """Load the data, split it and compare the models on the training rows."""
        return self.compare(*self.load_training_data())
