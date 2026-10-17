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
    TEST_SIZE,
)
from machine_failure.features import FEATURE_SETS, FeatureSet
from machine_failure.models import (
    FINAL_MODEL,
    FINAL_THRESHOLD,
    MODEL_PATH,
    all_models,
    get_model,
)


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
    """Where the dataset is and how it is split into training and test rows."""

    path: Path = DEFAULT_RAW_DIR / RAW_FILENAME
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ConfigError(
                f"test_size must be between 0 and 1, not {self.test_size}"
            )


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
