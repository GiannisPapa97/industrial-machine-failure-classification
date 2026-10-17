"""The final model: the chosen tuned model, trained on the whole training set."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from machine_failure.cli import UserError
from machine_failure.models.base import get_model
from machine_failure.models.candidates import RandomForestModel

# The chosen model, and its decision threshold (``Model.threshold``): the one with
# the best F1 on its out-of-fold predictions (see the README). The final model is
# refit on all the training rows, so that threshold approximates the best one.
FINAL_MODEL = RandomForestModel.name
FINAL_THRESHOLD = get_model(FINAL_MODEL).own_threshold()
MODEL_PATH = Path("models/final_model.joblib")

# Version 2 added ``metadata``. Version 1 files (no "format" key) still load.
FORMAT = 2
SAVED_KEYS = {"format", "name", "model", "threshold", "metadata"}
VERSION_1_KEYS = {"name", "model", "threshold"}

logger = logging.getLogger(__name__)


class ModelFileError(UserError):
    """A file is not a model saved by ``save_final_model``, or cannot be read."""


@dataclass(frozen=True)
class FinalModel:
    """A fitted model together with the threshold that turns probabilities to alerts.

    ``metadata`` records how the model was made: the training config, where the
    threshold came from, the train/test split, the library versions and when it
    was trained (see ``TrainingPipeline``). It is saved with the model.
    """

    name: str
    model: Pipeline
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

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
    """Save the fitted model, its threshold and its metadata to ``path``.

    An existing file is never replaced unless ``overwrite`` is True. Saved models
    are not committed to git (the ``models/`` folder is ignored).
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "format": FORMAT,
        "name": final.name,
        "model": final.model,
        "threshold": final.threshold,
        "metadata": final.metadata,
    }
    joblib.dump(saved, path)
    return path


def load_final_model(path: Path = MODEL_PATH) -> FinalModel:
    """Load a model saved by ``save_final_model``.

    Only load files you created yourself: joblib files are pickles, which can run
    arbitrary code when loaded.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `uv run mf-train` first")
    try:
        saved = joblib.load(path)
    except Exception as error:  # any unreadable file: not a model, not a bug here
        raise ModelFileError(f"{path} cannot be read as a saved model") from error
    if isinstance(saved, dict) and set(saved) == VERSION_1_KEYS:
        logger.warning(
            "%s was saved without metadata; retrain it with `uv run mf-train "
            "--overwrite` to record how it was made",
            path,
        )
        return FinalModel(saved["name"], saved["model"], saved["threshold"])
    if not isinstance(saved, dict) or set(saved) != SAVED_KEYS:
        raise ModelFileError(f"{path} is not a saved final model")
    if saved["format"] != FORMAT:
        raise ModelFileError(
            f"{path} has format {saved['format']!r}; this version reads {FORMAT}"
        )
    final = FinalModel(
        saved["name"], saved["model"], saved["threshold"], saved["metadata"]
    )
    check_versions(final, path)
    return final


def check_versions(final: FinalModel, path: Path) -> None:
    """Warn when the model was saved with another scikit-learn version.

    Pickled scikit-learn models are only guaranteed to load and behave the same
    with the version that saved them.
    """
    saved = final.metadata.get("versions", {}).get("scikit-learn")
    if saved is not None and saved != sklearn.__version__:
        logger.warning(
            "%s was saved with scikit-learn %s; this is %s, so its predictions "
            "may differ",
            path,
            saved,
            sklearn.__version__,
        )
