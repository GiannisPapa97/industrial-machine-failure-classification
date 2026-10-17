import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from machine_failure.config import EvaluationConfig, TrainingConfig, load_config
from machine_failure.data import split_train_test
from machine_failure.experiments import app
from machine_failure.pipelines import TrainingPipeline, evaluation, splits
from machine_failure.pipelines.splits import split_fingerprint

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture
def saved_model(make_xy: MakeXY, tmp_path: Path) -> Path:
    """A quick k-nearest-neighbours model trained on a synthetic training split."""
    X, y = make_xy(signal=6.0)
    config = TrainingConfig(
        model="knn", tuned=False, threshold=0.5, output_path=tmp_path / "final.joblib"
    )
    pipeline = TrainingPipeline(config)
    parts = split_train_test(X, y)
    X_train, _, y_train, _ = parts
    return pipeline.save(
        pipeline.fit(X_train, y_train, split=split_fingerprint(*parts))
    )


@pytest.fixture
def small_run(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """A small synthetic dataset instead of the downloaded one."""
    monkeypatch.setattr(splits, "load_xy", lambda path: make_xy(signal=6.0))


def test_main_refuses_to_use_the_test_set_without_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    saved_model: Path,
) -> None:
    def must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("the data must not be read without --confirm")

    monkeypatch.setattr(splits, "load_xy", must_not_run)
    monkeypatch.setattr(evaluation, "evaluate_final", must_not_run)
    monkeypatch.setattr(
        sys, "argv", ["mf", "evaluate", "--model-path", str(saved_model)]
    )

    with pytest.raises(SystemExit) as exit_info:
        app.main()

    assert exit_info.value.code == 1
    assert "--confirm" in capsys.readouterr().err


def test_main_evaluates_on_the_test_set_when_confirmed(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    saved_model: Path,
) -> None:
    argv = ["mf", "evaluate", "--confirm", "--model-path", str(saved_model)]
    monkeypatch.setattr(sys, "argv", argv + ["--n-bootstrap", "50", "--skip-reference"])

    app.main()

    out = capsys.readouterr().out
    assert "on the test set (120 rows, 12 failures)" in out  # 20% of 600 rows
    assert "95% interval" in out and "cross-validation" not in out


def test_main_adds_the_cross_validation_reference_by_default(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    saved_model: Path,
) -> None:
    argv = ["mf", "evaluate", "--confirm", "--model-path", str(saved_model)]
    monkeypatch.setattr(sys, "argv", argv + ["--n-bootstrap", "50"])

    app.main()

    assert "cross-validation" in capsys.readouterr().out


def test_main_reads_the_settings_from_a_config_file(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    saved_model: Path,
    tmp_path: Path,
) -> None:
    config = tmp_path / "evaluate.toml"
    config.write_text(
        f'model_path = "{saved_model}"\nn_bootstrap = 30\nreference = false\n'
        '[data]\npath = "other.csv"\n'
    )
    argv = ["mf", "evaluate", "--confirm", "--config", str(config)]
    monkeypatch.setattr(sys, "argv", argv)

    app.main()

    out = capsys.readouterr().out
    assert "on the test set (120 rows, 12 failures)" in out  # 20% of 600 rows
    assert "cross-validation" not in out


def test_command_line_options_override_the_config_file(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    saved_model: Path,
    tmp_path: Path,
) -> None:
    config = tmp_path / "evaluate.toml"
    config.write_text(f'model_path = "{tmp_path / "nope.joblib"}"\nreference = true\n')
    argv = ["mf", "evaluate", "--confirm", "--config", str(config)]
    argv += [
        "--model-path",
        str(saved_model),
        "--skip-reference",
        "--n-bootstrap",
        "20",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    app.main()

    assert "cross-validation" not in capsys.readouterr().out


def test_the_example_config_is_the_default_evaluation() -> None:
    example = Path(__file__).parents[2] / "configs" / "evaluate_final.toml"

    assert load_config(example, EvaluationConfig) == EvaluationConfig()


def test_main_points_to_the_command_that_creates_a_missing_model(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    argv = [
        "mf",
        "evaluate",
        "--confirm",
        "--model-path",
        str(tmp_path / "nope.joblib"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        app.main()
    assert "mf-train" in capsys.readouterr().err
