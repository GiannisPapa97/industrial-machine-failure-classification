"""The inference pipeline: score new rows with the saved final model."""

import logging
from pathlib import Path

import pandas as pd

from machine_failure.config import InferenceConfig
from machine_failure.data import FEATURE_COLUMNS, validate_features
from machine_failure.models import FinalModel, load_final_model

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Score rows of measurements with the model saved by the training pipeline.

    The rows are checked first (see ``validate_features``), so a missing column or
    an unknown machine type fails with a clear message instead of deep inside the
    model. Only the feature columns are used; other columns, such as an identifier
    or the true label, are ignored. The model is loaded on first use.

    Saved models are pickles: load only files you created yourself.
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._final: FinalModel | None = None

    @property
    def final(self) -> FinalModel:
        """The saved model and its threshold, loaded once."""
        if self._final is None:
            logger.debug("loading the model from %s", self.config.model_path)
            self._final = load_final_model(self.config.model_path)
        return self._final

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """The failure probability and alert of each row, with the index of ``X``.

        ``flagged`` is 1 when the probability is at least the model's threshold.
        """
        validate_features(X)
        features = X[FEATURE_COLUMNS]
        return pd.DataFrame(
            {
                "probability": self.final.probability(features),
                "flagged": self.final.predict(features),
            }
        )

    def predict_csv(self, path: Path) -> pd.DataFrame:
        """``predict`` for the rows of the CSV file at ``path``."""
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        return self.predict(pd.read_csv(path))
