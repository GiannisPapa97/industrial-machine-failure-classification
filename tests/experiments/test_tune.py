import runpy
import sys
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from machine_failure import data, models
from machine_failure.evaluation import METRICS
from machine_failure.experiments import tune
from machine_failure.experiments.tune import (
    TuningResult,
    comparison_table,
    format_results,
    tune_models,
)

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]

SMALL_SPACE: dict[str, list[Any]] = {
    "model__n_estimators": [10, 20],
    "model__max_depth": [3, None],
}


def cheap_candidates() -> dict[str, Pipeline]:
    """A quick random forest standing in for the real, slower candidates."""
    forest = RandomForestClassifier(
        n_estimators=10, class_weight="balanced_subsample", random_state=0
    )
    return {"random_forest": models.with_preprocessing(forest)}


@pytest.fixture
def small_run(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """One cheap model, a tiny search space and a small synthetic dataset."""
    spaces = {"random_forest": SMALL_SPACE}
    monkeypatch.setattr(tune, "SEARCH_SPACES", spaces)
    monkeypatch.setattr(tune, "TUNED_MODELS", ("random_forest",))
    monkeypatch.setattr(tune, "load_xy", lambda: make_xy(n=400))
    monkeypatch.setattr(tune, "build_candidates", cheap_candidates)
    # Running the module as a script re-imports it, so patch the sources too.
    monkeypatch.setattr(models, "SEARCH_SPACES", spaces)
    monkeypatch.setattr(models, "TUNED_MODELS", ("random_forest",))
    monkeypatch.setattr(data, "load_xy", lambda: make_xy(n=400))
    monkeypatch.setattr(models, "build_candidates", cheap_candidates)


@pytest.fixture(scope="module")
def results(make_xy: MakeXY) -> dict[str, TuningResult]:
    """Tune once and share the result between the tests that only read it."""
    X, y = make_xy(n=400)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tune, "SEARCH_SPACES", {"random_forest": SMALL_SPACE})
        patch.setattr(tune, "build_candidates", cheap_candidates)
        return tune_models(X, y, ("random_forest",), n_iter=2, inner_splits=2)


def test_tune_models_scores_default_and_tuned_on_the_same_folds(
    results: dict[str, TuningResult],
) -> None:
    result = results["random_forest"]

    assert result.default_scores.shape == result.tuned_scores.shape == (5, len(METRICS))
    assert set(result.best_settings) == {"n_estimators", "max_depth"}
    assert 0.5 < result.search_score <= 1.0


def test_tune_models_reports_progress(
    make_xy: MakeXY, small_run: None, capsys: pytest.CaptureFixture[str]
) -> None:
    X, y = make_xy(n=400)

    tune_models(X, y, ("random_forest",), n_iter=2, inner_splits=2, verbose=True)

    err = capsys.readouterr().err
    assert "random_forest: scoring the default settings" in err
    assert "random_forest: nested cross-validation" in err
    assert "random_forest: searching on the whole training set" in err


def test_comparison_table_has_a_default_and_a_tuned_row_per_model(
    results: dict[str, TuningResult],
) -> None:
    table = comparison_table(results)

    assert list(table.index) == ["random_forest (default)", "random_forest (tuned)"]
    assert list(table.to_dict(orient="list")) == [
        (metric, stat) for metric in METRICS for stat in ("mean", "std")
    ]


def test_format_results_shows_the_table_and_the_best_settings(
    results: dict[str, TuningResult],
) -> None:
    text = format_results(results)

    assert "random_forest (default)" in text and "random_forest (tuned)" in text
    assert "±" in text
    assert "random_forest: best settings {" in text
    assert "(optimistic) against" in text


def test_main_prints_the_comparison_for_the_training_set(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["tune", "--n-iter", "2", "--inner-splits", "2"]
    monkeypatch.setattr(sys, "argv", argv)

    tune.main()

    out = capsys.readouterr().out
    assert "320 rows, 32 failures" in out  # 80% of 400 rows, 10% failures
    assert "2 random settings per search" in out
    assert "random_forest (tuned)" in out


def test_main_accepts_models_repeats_and_jobs(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["tune", "--models", "random_forest", "--n-iter", "2", "--inner-splits", "2"]
    monkeypatch.setattr(sys, "argv", argv + ["--repeats", "2", "--n-jobs", "2"])

    tune.main()

    assert "2 repeat(s)" in capsys.readouterr().out


def test_main_rejects_an_unknown_model(
    small_run: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["tune", "--models", "dummy"])

    with pytest.raises(SystemExit):
        tune.main()


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_runs_main_when_executed_as_script(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["tune", "--n-iter", "2", "--inner-splits", "2"]
    monkeypatch.setattr(sys, "argv", argv)

    runpy.run_module("machine_failure.experiments.tune", run_name="__main__")

    assert "random_forest (default)" in capsys.readouterr().out
