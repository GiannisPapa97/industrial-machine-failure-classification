import sys
from collections.abc import Callable

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from machine_failure.data import FAILURE_MODE_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from machine_failure.experiments import app, interpretation
from machine_failure.models import with_preprocessing
from machine_failure.pipelines import splits

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
    raw = make_raw(make_xy)
    X, y = raw[FEATURE_COLUMNS].copy(), raw[TARGET_COLUMN].copy()
    monkeypatch.setattr(splits, "load_xy", lambda path: (X, y))
    monkeypatch.setattr(interpretation, "load_raw", lambda path: raw)
    monkeypatch.setattr(interpretation, "build_tuned_models", lambda: quick)


def test_main_reports_importance_misses_and_medians(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["mf", "interpret", "--n-repeats", "2"])

    app.main()

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
    argv = ["mf", "interpret", "--model", "random_forest", "--threshold", "0.3"]
    monkeypatch.setattr(sys, "argv", argv + ["--n-repeats", "2"])

    app.main()

    assert "threshold 0.3." in capsys.readouterr().out


@pytest.mark.usefixtures("small_run")
def test_main_uses_the_model_threshold_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app.main(["interpret", "--n-repeats", "1"])

    assert "threshold 0.475" in capsys.readouterr().out


def test_main_rejects_a_model_that_is_not_tuned(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        app.main(["interpret", "--model", "svm"])

    assert exit_info.value.code == 2  # argparse: not one of the choices
    assert "invalid choice: 'svm'" in capsys.readouterr().err
