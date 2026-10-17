"""The final model: the chosen tuned model, trained on the whole training set."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from machine_failure.models.base import get_model
from machine_failure.models.candidates import RandomForestModel

# The chosen model, and its decision threshold (``Model.threshold``): the one with
# the best F1 on its out-of-fold predictions (see the README). The final model is
# refit on all the training rows, so that threshold approximates the best one.
FINAL_MODEL = RandomForestModel.name
FINAL_THRESHOLD = get_model(FINAL_MODEL).own_threshold()
MODEL_PATH = Path("models/final_model.joblib")

SAVED_KEYS = {"name", "model", "threshold"}


@dataclass(frozen=True)
class FinalModel:
    """A fitted model together with the threshold that turns probabilities to alerts."""

    name: str
    model: Pipeline
    threshold: float

    def probability(self, X: pd.DataFrame) -> pd.Series:
        """The predicted probability of failure of each row."""
        return pd.Series(
            self.model.predict_proba(X)[:, 1], index=X.index, name="probability"
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """1 for rows flagged as failures (probability at least the threshold)."""
        flagged = self.probability(X) >= self.threshold
        return flagged.astype(int).rename("flagged")


def save_final_model(
    final: FinalModel, path: Path = MODEL_PATH, *, overwrite: bool = False
) -> Path:
    """Save the fitted model and its threshold to ``path``.

    An existing file is never replaced unless ``overwrite`` is True. Saved models
    are not committed to git (the ``models/`` folder is ignored).
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"name": final.name, "model": final.model, "threshold": final.threshold}, path
    )
    return path


def load_final_model(path: Path = MODEL_PATH) -> FinalModel:
    """Load a model saved by ``save_final_model``.

    Only load files you created yourself: joblib files are pickles, which can run
    arbitrary code when loaded.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `uv run mf-train` first")
    saved = joblib.load(path)
    if not isinstance(saved, dict) or set(saved) != SAVED_KEYS:
        raise ValueError(f"{path} is not a saved final model")
    return FinalModel(saved["name"], saved["model"], saved["threshold"])
