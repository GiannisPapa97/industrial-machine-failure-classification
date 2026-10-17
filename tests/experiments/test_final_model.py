import runpy
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from machine_failure.experiments import final_model
from machine_failure.models import FINAL_THRESHOLD, load_final_model
from machine_failure.pipelines import splits

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture
def small_run(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """A small synthetic dataset instead of the downloaded one."""
    monkeypatch.setattr(splits, "load_xy", lambda path: make_xy(n=400))


def test_main_trains_on_the_training_set_and_saves(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "models" / "final.joblib"
    monkeypatch.setattr(sys, "argv", ["final_model", "--output", str(path)])

    final_model.main()

    out = capsys.readouterr().out
    assert "320 rows, 32 failures" in out  # 80% of 400 rows: the test set is not used
    assert f"threshold {FINAL_THRESHOLD}" in out
    assert load_final_model(path).threshold == FINAL_THRESHOLD


def test_main_accepts_a_threshold_and_refuses_to_overwrite(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "final.joblib"
    argv = ["final_model", "--output", str(path), "--threshold", "0.3"]
    monkeypatch.setattr(sys, "argv", argv)

    final_model.main()
    assert load_final_model(path).threshold == 0.3

    with pytest.raises(SystemExit):
        final_model.main()
    assert "already exists" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*argv, "--overwrite"])
    final_model.main()


def test_main_reads_the_settings_from_a_config_file(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "final.joblib"
    config = tmp_path / "train.toml"
    config.write_text(
        f'model = "xgboost"\nthreshold = 0.3\noutput_path = "{path}"\n'
        '[data]\npath = "other.csv"\n'
    )
    monkeypatch.setattr(sys, "argv", ["final_model", "--config", str(config)])

    final_model.main()

    assert "xgboost on the training set (320 rows" in capsys.readouterr().out
    saved = load_final_model(path)
    assert (saved.name, saved.threshold) == ("xgboost", 0.3)


def test_command_line_options_override_the_config_file(
    small_run: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "train.toml"
    config.write_text('model = "xgboost"\nthreshold = 0.3\n')
    path = tmp_path / "final.joblib"
    argv = ["final_model", "--config", str(config), "--model", "random_forest"]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(path)])

    final_model.main()

    saved = load_final_model(path)
    assert (saved.name, saved.threshold) == ("random_forest", 0.3)


def test_the_example_config_trains_the_same_model_as_the_defaults(
    small_run: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    example = Path(__file__).parents[2] / "configs" / "train_final.toml"
    path = tmp_path / "final.joblib"
    argv = ["final_model", "--config", str(example), "--output", str(path)]
    monkeypatch.setattr(sys, "argv", argv)

    final_model.main()

    saved = load_final_model(path)
    assert (saved.name, saved.threshold) == ("random_forest", FINAL_THRESHOLD)


def test_main_rejects_a_model_without_tuned_settings(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    argv = ["final_model", "--model", "knn", "--output", str(tmp_path / "m.joblib")]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        final_model.main()
    assert (
        capsys.readouterr().err
        == "error: knn has no tuned settings; set tuned = false\n"
    )
    assert not (tmp_path / "m.joblib").exists()


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_runs_main_when_executed_as_script(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "final.joblib"
    monkeypatch.setattr(sys, "argv", ["final_model", "--output", str(path)])

    runpy.run_module("machine_failure.experiments.final_model", run_name="__main__")

    assert "Saved to" in capsys.readouterr().out
    assert path.is_file()
