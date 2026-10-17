from pathlib import Path

import pytest

from machine_failure.config import ConfigError, DataConfig, TrainingConfig
from machine_failure.data import DEFAULT_RAW_DIR, RAW_FILENAME
from machine_failure.models import FINAL_MODEL, FINAL_THRESHOLD, MODEL_PATH


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
