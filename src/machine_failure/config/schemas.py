"""Typed, validated settings for the workflows.

Each config is a frozen dataclass that checks itself on construction, so a
mistake (an unknown model, a test size of 1.5) fails early with a clear message
instead of deep inside a workflow. The defaults are the values the workflows used
before configs existed.
"""

from dataclasses import dataclass, field
from pathlib import Path

from machine_failure.config.errors import ConfigError
from machine_failure.data import (
    DEFAULT_RAW_DIR,
    RANDOM_STATE,
    RAW_FILENAME,
)
from machine_failure.evaluation import N_SPLITS
from machine_failure.features import FEATURE_SETS, FeatureSet
from machine_failure.models import (
    FINAL_MODEL,
    FINAL_THRESHOLD,
    MODEL_PATH,
    all_models,
    get_model,
)

N_BOOTSTRAP = 1000


def check_model_name(name: str) -> None:
    """Raise ``ConfigError`` unless ``name`` is a registered model."""
    if name not in all_models():
        raise ConfigError(f"unknown model {name!r}; known: {list(all_models())}")


def check_feature_set(feature_set: str) -> None:
    """Raise ``ConfigError`` unless ``feature_set`` is one of the feature sets."""
    if feature_set not in FEATURE_SETS:
        raise ConfigError(
            f"unknown feature set {feature_set!r}; known: {list(FEATURE_SETS)}"
        )


@dataclass(frozen=True)
class DataConfig:
    """Where the dataset is.

    The split into training and held-out test rows is not a setting: it is fixed
    for the project (``TEST_SIZE`` and ``RANDOM_STATE`` in ``machine_failure.data``).
    Every choice so far (model, settings, features, threshold) was made on the
    same training rows, so a different split would put rows that shaped those
    choices into the test set.
    """

    path: Path = DEFAULT_RAW_DIR / RAW_FILENAME


@dataclass(frozen=True)
class TrainingConfig:
    """Which model to train on the training rows, and where to save it.

    ``tuned`` trains the model with its tuned settings (only models that have
    them); ``threshold`` turns its probabilities into failure alerts.
    """

    model: str = FINAL_MODEL
    feature_set: FeatureSet = "all"
    tuned: bool = True
    threshold: float = FINAL_THRESHOLD
    output_path: Path = MODEL_PATH
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        check_model_name(self.model)
        check_feature_set(self.feature_set)
        if self.tuned and not get_model(self.model).is_tuned:
            raise ConfigError(f"{self.model} has no tuned settings; set tuned = false")
        if not 0 < self.threshold < 1:
            raise ConfigError(
                f"threshold must be between 0 and 1, not {self.threshold}"
            )


def all_model_names() -> tuple[str, ...]:
    """The names of every registered model, in registry order."""
    return tuple(all_models())


@dataclass(frozen=True)
class ComparisonConfig:
    """Which models to compare, and how to cross-validate them.

    All registered models are compared by default. ``n_repeats`` repeats the
    cross-validation with different shuffles for steadier estimates.
    """

    models: tuple[str, ...] = field(default_factory=all_model_names)
    feature_set: FeatureSet = "all"
    n_splits: int = N_SPLITS
    n_repeats: int = 1
    random_state: int = RANDOM_STATE
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        if not self.models:
            raise ConfigError("models must name at least one model")
        if len(set(self.models)) != len(self.models):
            raise ConfigError(f"models lists a model twice: {list(self.models)}")
        for name in self.models:
            check_model_name(name)
        check_feature_set(self.feature_set)
        if self.n_splits < 2:
            raise ConfigError(f"n_splits must be at least 2, not {self.n_splits}")
        if self.n_repeats < 1:
            raise ConfigError(f"n_repeats must be at least 1, not {self.n_repeats}")


@dataclass(frozen=True)
class EvaluationConfig:
    """How the final model is scored once on the held-out test set.

    The model is loaded from ``model_path`` and judged at the threshold saved with
    it. ``n_bootstrap`` resamples of the test rows give the intervals;
    ``reference`` adds the cross-validation estimate at the same threshold, to
    compare with the test result (it takes about a minute).
    """

    model_path: Path = MODEL_PATH
    n_bootstrap: int = N_BOOTSTRAP
    reference: bool = True
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        if self.n_bootstrap < 1:
            raise ConfigError(f"n_bootstrap must be at least 1, not {self.n_bootstrap}")


@dataclass(frozen=True)
class InferenceConfig:
    """Which saved model scores new rows.

    Saved models are pickles: only point this at a file you created yourself.
    """

    model_path: Path = MODEL_PATH
