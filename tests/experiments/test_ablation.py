import logging
import sys
from collections.abc import Callable

import pandas as pd
import pytest

from machine_failure.evaluation import METRICS
from machine_failure.experiments import app
from machine_failure.experiments.ablation import (
    compare_feature_sets,
    format_feature_set_table,
)
from machine_failure.pipelines import splits

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]
BASELINES = ("dummy", "logistic_regression")
BASELINE_ARGS = ["--models", *BASELINES]


def make_table() -> pd.DataFrame:
    """Two models on two feature sets with known PR-AUC means and stds."""
    columns = pd.MultiIndex.from_product([["raw", "all"], METRICS, ["mean", "std"]])
    table = pd.DataFrame(0.0, index=["a", "b"], columns=columns)
    table[("raw", "pr_auc", "mean")] = [0.50, 0.20]
    table[("all", "pr_auc", "mean")] = [0.60, 0.80]
    table[("raw", "pr_auc", "std")] = [0.01, 0.02]
    table[("all", "pr_auc", "std")] = [0.03, 0.04]
    table[("all", "recall", "mean")] = [0.9, 0.1]
    return table


def test_compare_feature_sets_has_a_column_per_set_metric_and_statistic(
    make_xy: MakeXY,
) -> None:
    X, y = make_xy()

    table = compare_feature_sets(X, y, models=BASELINES)

    assert list(table.index) == ["dummy", "logistic_regression"]
    expected = [
        (fs, metric, stat)
        for fs in ("raw", "computed", "all")
        for metric in METRICS
        for stat in ("mean", "std")
    ]
    assert list(table.to_dict(orient="list")) == expected
    for feature_set in ("raw", "computed", "all"):
        pr_auc = table[(feature_set, "pr_auc", "mean")].to_dict()
        assert pr_auc["dummy"] == pytest.approx(0.1)
        assert pr_auc["logistic_regression"] > 0.5


def test_compare_feature_sets_accepts_a_subset_and_logs_progress(
    make_xy: MakeXY, caplog: pytest.LogCaptureFixture
) -> None:
    X, y = make_xy()

    with caplog.at_level(logging.INFO, logger="machine_failure"):
        table = compare_feature_sets(X, y, feature_sets=("raw",), models=BASELINES)

    assert {column[0] for column in table.columns} == {"raw"}
    assert "cross-validating on feature set: raw" in caplog.text


def test_format_orders_models_by_their_best_score_and_shows_mean_and_std() -> None:
    text = format_feature_set_table(make_table())

    lines = text.splitlines()
    assert lines[0].split() == ["raw", "all"]
    assert lines[1].startswith("b")  # best score 0.80 against 0.60
    assert "0.800 ± 0.040" in lines[1]
    assert "0.200 ± 0.020" in lines[1]
    assert "0.500 ± 0.010" in lines[2]


def test_format_can_show_another_metric() -> None:
    text = format_feature_set_table(make_table(), "recall")

    lines = text.splitlines()
    assert lines[1].startswith("a")  # recall 0.9 against 0.1
    assert "0.900 ± 0.000" in lines[1]


@pytest.fixture
def fast_run(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small synthetic dataset instead of the downloaded one."""
    monkeypatch.setattr(splits, "load_xy", lambda path: make_xy())


def test_main_prints_the_table_for_the_training_set(
    fast_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["mf", "compare", "--feature-sets", "raw", "computed", "all", *BASELINE_ARGS],
    )

    app.main()

    captured = capsys.readouterr()
    assert "480 rows, 48 failures" in captured.out
    assert "Feature sets: raw" in captured.out
    assert "Showing pr_auc" in captured.out
    assert "raw" in captured.out and "computed" in captured.out
    assert "logistic_regression" in captured.out
    assert "feature set: all" in captured.err


def test_main_accepts_metric_and_repeats(
    fast_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mf",
            "compare",
            "--feature-sets",
            "raw",
            "computed",
            "all",
            *BASELINE_ARGS,
            "--metric",
            "recall",
            "--repeats",
            "2",
        ],
    )

    app.main()

    out = capsys.readouterr().out
    assert "Showing recall" in out
    assert "2 repeat(s)" in out


def test_main_rejects_an_unknown_metric(
    fast_run: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mf",
            "compare",
            "--feature-sets",
            "raw",
            "computed",
            "all",
            "--metric",
            "luck",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        app.main()
    assert exit_info.value.code == 2  # argparse: not one of the choices
