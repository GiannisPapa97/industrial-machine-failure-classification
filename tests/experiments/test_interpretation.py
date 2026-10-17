import runpy
import sys
from collections.abc import Callable

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from machine_failure import data, models
from machine_failure.data import FAILURE_MODE_COLUMNS
from machine_failure.experiments import interpretation
from machine_failure.models import with_preprocessing

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


def make_raw(make_xy: MakeXY, n: int = 600) -> pd.DataFrame:
    """The raw table: the features, the target and the failure-mode flags."""
    X, y = make_xy(n, signal=12.0)
    raw = X.copy()
    raw["UID"] = range(n)
    raw["Product ID"] = "x"
    raw["Machine failure"] = y
    for column in FAILURE_MODE_COLUMNS:
        raw[column] = 0
    failures = raw.index[raw["Machine failure"] == 1]
    for position, row in enumerate(failures):
        mode = position % (len(FAILURE_MODE_COLUMNS) + 1)
        if mode < len(FAILURE_MODE_COLUMNS):  # the last one has no flag at all
            raw.loc[row, FAILURE_MODE_COLUMNS[mode]] = 1
    return raw


def cheap_forest() -> Pipeline:
    forest = RandomForestClassifier(n_estimators=20, random_state=0)
    return with_preprocessing(forest)


@pytest.fixture
def small_run(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """A small synthetic table and a quick forest instead of the tuned model."""
    quick = {"random_forest": cheap_forest()}
    monkeypatch.setattr(interpretation, "load_raw", lambda: make_raw(make_xy))
    monkeypatch.setattr(interpretation, "build_tuned_models", lambda: quick)
    # Running the module as a script re-imports it, so patch the sources too.
    monkeypatch.setattr(data, "load_raw", lambda *args, **kwargs: make_raw(make_xy))
    monkeypatch.setattr(models, "build_tuned_models", lambda: quick)


def test_main_reports_importance_misses_and_medians(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["interpretation", "--n-repeats", "2"])

    interpretation.main()

    out = capsys.readouterr().out
    assert "480 rows, 48 failures" in out  # 80% of 600 rows: the test set is not used
    assert "Drop in PR-AUC" in out and "Torque" in out
    assert "Failures caught and missed at the threshold" in out
    assert "TWF" in out and "none" in out
    assert "normal" in out and "missed failure" in out


def test_main_accepts_a_model_and_threshold(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["interpretation", "--model", "random_forest", "--threshold", "0.3"]
    monkeypatch.setattr(sys, "argv", argv + ["--n-repeats", "2"])

    interpretation.main()

    assert "threshold 0.3." in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_runs_main_when_executed_as_script(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["interpretation", "--n-repeats", "2"])

    runpy.run_module("machine_failure.experiments.interpretation", run_name="__main__")

    assert "Drop in PR-AUC" in capsys.readouterr().out
