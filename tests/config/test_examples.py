from pathlib import Path

import pytest

from machine_failure.config import (
    ComparisonConfig,
    EvaluationConfig,
    InterpretationConfig,
    ThresholdsConfig,
    TrainingConfig,
    TuningConfig,
    load_config,
)

CONFIG_DIR = Path(__file__).parents[2] / "configs"
# The file name starts with the kind of config it holds.
KINDS: dict[str, type] = {
    "compare": ComparisonConfig,
    "train": TrainingConfig,
    "evaluate": EvaluationConfig,
    "tune": TuningConfig,
    "thresholds": ThresholdsConfig,
    "interpret": InterpretationConfig,
}
EXAMPLES = sorted(CONFIG_DIR.glob("*.toml"))


def test_the_example_configs_exist() -> None:
    assert [path.name for path in EXAMPLES] == [
        "compare_default.toml",
        "compare_trees.toml",
        "evaluate_final.toml",
        "interpret_final.toml",
        "thresholds_default.toml",
        "train_final.toml",
        "tune_default.toml",
    ]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_every_example_loads_and_validates(path: Path) -> None:
    kind = KINDS[path.name.split("_")[0]]

    assert isinstance(load_config(path, kind), kind)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("compare_default.toml", ComparisonConfig),
        ("train_final.toml", TrainingConfig),
        ("evaluate_final.toml", EvaluationConfig),
        ("tune_default.toml", TuningConfig),
        ("thresholds_default.toml", ThresholdsConfig),
        ("interpret_final.toml", InterpretationConfig),
    ],
)
def test_the_default_examples_spell_out_the_defaults(name: str, kind: type) -> None:
    assert load_config(CONFIG_DIR / name, kind) == kind()
