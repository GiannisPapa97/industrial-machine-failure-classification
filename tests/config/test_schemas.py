import dataclasses
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
from machine_failure.evaluation import N_SPLITS
from machine_failure.models import FINAL_MODEL, FINAL_THRESHOLD, MODEL_PATH, all_models


def test_data_config_defaults_are_the_values_used_so_far() -> None:
    config = DataConfig()

    assert config.path == DEFAULT_RAW_DIR / RAW_FILENAME


def test_the_split_is_not_a_setting() -> None:
    # every decision was made on the same training rows, so the split is fixed
    assert {f.name for f in dataclasses.fields(DataConfig)} == {"path"}


def test_training_config_defaults_are_the_final_model() -> None:
    config = TrainingConfig()

    assert config.model == FINAL_MODEL
    assert config.feature_set == "all"
    assert config.tuned
    assert config.threshold is None  # the model's own
    assert config.decision_threshold == FINAL_THRESHOLD == 0.475
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

    assert TrainingConfig(model="knn", tuned=False, threshold=0.3).model == "knn"


def test_each_tuned_model_brings_its_own_threshold() -> None:
    assert TrainingConfig(model="xgboost").decision_threshold == 0.654
    assert TrainingConfig(model="extra_trees").decision_threshold == 0.416


def test_a_threshold_that_is_set_wins() -> None:
    config = TrainingConfig(model="xgboost", threshold=0.3)

    assert config.decision_threshold == 0.3


@pytest.mark.parametrize("model", ["knn", "random_forest"])
def test_an_untuned_model_needs_a_threshold(model: str) -> None:
    # a tuned model's threshold does not fit the same model with default settings
    with pytest.raises(ConfigError, match="set threshold"):
        TrainingConfig(model=model, tuned=False)


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


def test_evaluation_config_defaults_score_the_default_model_like_mf_evaluate() -> None:
    config = EvaluationConfig()

    assert config.model_path == MODEL_PATH
    assert config.n_bootstrap == 1000
    assert config.reference is True
    assert config.data == DataConfig()


@pytest.mark.parametrize("n_bootstrap", [0, -5])
def test_evaluation_config_rejects_too_few_resamples(n_bootstrap: int) -> None:
    with pytest.raises(ConfigError, match="n_bootstrap"):
        EvaluationConfig(n_bootstrap=n_bootstrap)


def test_inference_config_defaults_to_the_default_model_path() -> None:
    assert InferenceConfig().model_path == MODEL_PATH
