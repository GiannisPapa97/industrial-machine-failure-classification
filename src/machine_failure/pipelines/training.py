"""The training pipeline: load, split, and fit one model on the training rows."""

from dataclasses import dataclass

from machine_failure.config import TrainingConfig
from machine_failure.models import FinalModel


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
