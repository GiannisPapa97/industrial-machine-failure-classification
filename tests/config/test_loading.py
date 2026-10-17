from pathlib import Path

import pytest

from machine_failure.config import (
    ComparisonConfig,
    ConfigError,
    DataConfig,
    EvaluationConfig,
    TrainingConfig,
    load_config,
)
from machine_failure.config.loading import convert


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_an_empty_file_gives_the_defaults(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, ""), TrainingConfig) == TrainingConfig()
    assert load_config(write(tmp_path, ""), ComparisonConfig) == ComparisonConfig()


def test_keys_override_the_defaults_including_nested_data(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
models = ["dummy", "xgboost"]
feature_set = "raw"
n_repeats = 3

[data]
path = "somewhere/data.csv"
test_size = 0.25
""",
    )

    config = load_config(path, ComparisonConfig)

    assert config.models == ("dummy", "xgboost")
    assert config.feature_set == "raw"
    assert config.n_repeats == 3
    assert config.n_splits == ComparisonConfig().n_splits
    assert config.data == DataConfig(path=Path("somewhere/data.csv"), test_size=0.25)


def test_paths_bools_and_floats_are_converted(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        'model = "knn"\ntuned = false\nthreshold = 0.3\noutput_path = "m.joblib"\n',
    )

    config = load_config(path, TrainingConfig)

    assert config.tuned is False
    assert config.threshold == 0.3
    assert config.output_path == Path("m.joblib")


def test_an_integer_is_accepted_where_a_float_is_expected() -> None:
    value = convert(1, float, "x")

    assert value == 1.0 and isinstance(value, float)


def test_float_lists_are_converted() -> None:
    assert convert([0.7, 1], tuple[float, ...], "x") == (0.7, 1.0)


def test_evaluation_settings_are_read(tmp_path: Path) -> None:
    config = load_config(
        write(tmp_path, "n_bootstrap = 50\nreference = false\n"), EvaluationConfig
    )

    assert (config.n_bootstrap, config.reference) == (50, False)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("modle = 'knn'", "unknown key.*modle"),
        ("[data]\nsplit = 1", "unknown key.*data.*split"),
        ("model = 3", "model must be str"),
        ("tuned = 'yes'", "tuned must be bool"),
        ("threshold = true", "threshold must be float"),
        ("feature_set = 1", "feature_set must be str"),
        ("data = 3", "data must be a table"),
        ("[data]\ntest_size = 'big'", r"data\.test_size must be float"),
        ("[data]\nrandom_state = true", r"data\.random_state must be int"),
        ("model = 'nope'", "unknown model"),
    ],
)
def test_bad_files_name_the_problem(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, text), TrainingConfig)


def test_a_list_must_be_a_list_of_the_right_items(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="models must be a list"):
        load_config(write(tmp_path, "models = 'dummy'"), ComparisonConfig)
    with pytest.raises(ConfigError, match=r"models\[1\] must be str"):
        load_config(write(tmp_path, "models = ['dummy', 2]"), ComparisonConfig)


def test_invalid_toml_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(write(tmp_path, "model = "), TrainingConfig)


def test_the_error_names_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, "oops = 1")

    with pytest.raises(ConfigError, match="config.toml"):
        load_config(path, TrainingConfig)


def test_a_missing_file_is_a_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml", TrainingConfig)
