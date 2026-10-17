import runpy
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from machine_failure import data, tuned_models
from machine_failure.experiments import thresholds
from machine_failure.models import with_preprocessing

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


def cheap_models() -> dict[str, Pipeline]:
    forest = RandomForestClassifier(n_estimators=10, random_state=0)
    return {
        "forest": with_preprocessing(forest),
        "logistic": with_preprocessing(LogisticRegression()),
    }


@pytest.fixture
def small_run(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """A small synthetic dataset and two quick models instead of the tuned four."""
    monkeypatch.setattr(thresholds, "load_xy", lambda: make_xy(n=400))
    monkeypatch.setattr(thresholds, "build_tuned_models", cheap_models)
    # Running the module as a script re-imports it, so patch the sources too.
    monkeypatch.setattr(data, "load_xy", lambda: make_xy(n=400))
    monkeypatch.setattr(tuned_models, "build_tuned_models", cheap_models)


def test_main_prints_the_comparison_for_the_training_set(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["thresholds"])

    thresholds.main()

    captured = capsys.readouterr()
    assert "320 rows, 32 failures" in captured.out  # 80% of 400 rows, 10% failures
    assert "At each recall target" in captured.out
    assert "At the threshold with the best F1" in captured.out
    for target in ("80%", "85%", "90%"):
        assert target in captured.out
    assert "forest" in captured.out and "logistic" in captured.out
    assert "forest: out-of-fold predictions" in captured.err
    assert "Plot saved" not in captured.out


def test_main_accepts_targets_and_saves_a_plot(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "pr.png"
    argv = ["thresholds", "--targets", "0.7", "--plot", str(path)]
    monkeypatch.setattr(sys, "argv", argv)

    thresholds.main()

    out = capsys.readouterr().out
    assert "70%" in out and "85%" not in out
    assert f"Plot saved to {path}" in out
    assert path.is_file()


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_runs_main_when_executed_as_script(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["thresholds"])

    runpy.run_module("machine_failure.experiments.thresholds", run_name="__main__")

    assert "At each recall target" in capsys.readouterr().out
