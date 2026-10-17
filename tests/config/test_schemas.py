from pathlib import Path

import pytest

from machine_failure.config import (
    ComparisonConfig,
    ConfigError,
    DataConfig,
    EvaluationConfig,
    InferenceConfig,
    TrainingConfig,
)
from machine_failure.data import DEFAULT_RAW_DIR, RAW_FILENAME
from machine_failure.evaluation import DEFAULT_TARGETS, N_SPLITS
from machine_failure.models import FINAL_MODEL, FINAL_THRESHOLD, MODEL_PATH, all_models


def test_data_config_defaults_are_the_values_used_so_far() -> None:
    config = DataConfig()

    assert config.path == DEFAULT_RAW_DIR / RAW_FILENAME
    assert config.test_size == 0.2
    assert config.random_state == 42


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_data_config_rejects_a_test_size_outside_zero_and_one(test_size: float) -> None:
    with pytest.raises(ConfigError, match="test_size"):
        DataConfig(test_size=test_size)


def test_training_config_defaults_are_the_final_model() -> None:
    config = TrainingConfig()

    assert config.model == FINAL_MODEL
    assert config.feature_set == "all"
    assert config.tuned
    assert config.threshold == FINAL_THRESHOLD
    assert config.output_path == MODEL_PATH
    assert config.data == DataConfig()


def test_configs_are_frozen() -> None:
    with pytest.raises(AttributeError):
        TrainingConfig().model = "xgboost"  # type: ignore[misc]


def test_training_config_rejects_an_unknown_model() -> None:
    with pytest.raises(ConfigError, match="unknown model 'nope'"):
        TrainingConfig(model="nope")


def test_training_config_rejects_an_unknown_feature_set() -> None:
    with pytest.raises(ConfigError, match="unknown feature set"):
        TrainingConfig(feature_set="everything")  # type: ignore[arg-type]


def test_training_config_rejects_tuned_settings_for_an_untuned_model() -> None:
    with pytest.raises(ConfigError, match="knn has no tuned settings"):
        TrainingConfig(model="knn")

    assert TrainingConfig(model="knn", tuned=False).model == "knn"


@pytest.mark.parametrize("threshold", [0.0, 1.0, 2.0])
def test_training_config_rejects_a_threshold_outside_zero_and_one(
    threshold: float,
) -> None:
    with pytest.raises(ConfigError, match="threshold"):
        TrainingConfig(threshold=threshold)


def test_paths_can_be_set() -> None:
    config = TrainingConfig(output_path=Path("out.joblib"))

    assert config.output_path == Path("out.joblib")


def test_comparison_config_defaults_compare_every_model_like_mf_compare() -> None:
    config = ComparisonConfig()

    assert config.models == tuple(all_models())
    assert config.feature_set == "all"
    assert config.n_splits == N_SPLITS
    assert config.n_repeats == 1
    assert config.random_state == 42
    assert config.data == DataConfig()


def test_comparison_config_accepts_a_subset_of_models() -> None:
    assert ComparisonConfig(models=("dummy", "xgboost")).models == ("dummy", "xgboost")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"models": ()}, "at least one model"),
        ({"models": ("dummy", "dummy")}, "twice"),
        ({"models": ("dummy", "nope")}, "unknown model 'nope'"),
        ({"feature_set": "everything"}, "unknown feature set"),
        ({"n_splits": 1}, "n_splits"),
        ({"n_repeats": 0}, "n_repeats"),
    ],
)
def test_comparison_config_rejects_bad_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        ComparisonConfig(**kwargs)  # type: ignore[arg-type]


def test_evaluation_config_defaults_are_the_final_threshold_and_targets() -> None:
    config = EvaluationConfig()

    assert config.threshold == FINAL_THRESHOLD
    assert config.recall_targets == tuple(DEFAULT_TARGETS)
    assert config.data == DataConfig()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 0.0}, "threshold"),
        ({"threshold": 1.0}, "threshold"),
        ({"recall_targets": ()}, "at least one target"),
        ({"recall_targets": (0.8, 0.0)}, "recall targets"),
        ({"recall_targets": (1.2,)}, "recall targets"),
    ],
)
def test_evaluation_config_rejects_bad_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        EvaluationConfig(**kwargs)  # type: ignore[arg-type]


def test_a_recall_target_of_one_is_allowed() -> None:
    assert EvaluationConfig(recall_targets=(1.0,)).recall_targets == (1.0,)


def test_inference_config_defaults_to_the_default_model_path() -> None:
    assert InferenceConfig().model_path == MODEL_PATH
