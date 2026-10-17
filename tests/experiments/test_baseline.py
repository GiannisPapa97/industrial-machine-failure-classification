import runpy
import sys
from collections.abc import Callable

import pandas as pd
import pytest

from machine_failure import data
from machine_failure.experiments import baseline

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture
def fake_load_xy(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the dataset with a small synthetic one, for the command-line tests."""
    monkeypatch.setattr(data, "load_xy", lambda: make_xy())
    monkeypatch.setattr(baseline, "load_xy", lambda: make_xy())


def test_main_prints_the_comparison_for_the_training_set(
    fake_load_xy: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["baseline"])

    baseline.main()

    out = capsys.readouterr().out
    assert "5-fold stratified cross-validation on the training set" in out
    assert "480 rows, 48 failures" in out  # 80% of 600 rows, 10% failures
    assert "dummy" in out and "logistic_regression" in out


def test_main_accepts_repeats(
    fake_load_xy: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["baseline", "--repeats", "2"])

    baseline.main()

    assert "2 repeat(s)" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_runs_main_when_executed_as_script(
    fake_load_xy: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["baseline"])

    runpy.run_module("machine_failure.experiments.baseline", run_name="__main__")

    assert "logistic_regression" in capsys.readouterr().out
