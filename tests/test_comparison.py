import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from machine_failure import comparison, data, models
from machine_failure.baseline import METRICS, build_baselines
from machine_failure.comparison import sort_by_metric
from machine_failure.models import build_all_models


def make_xy(n: int = 600, failure_rate: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic features with the real columns; failures have a higher torque."""
    rng = np.random.default_rng(0)
    n_failures = int(n * failure_rate)
    y = pd.Series([1] * n_failures + [0] * (n - n_failures), name="Machine failure")
    X = pd.DataFrame(
        {
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Air temperature": rng.normal(300, 2, n),
            "Process temperature": rng.normal(310, 1.5, n),
            "Rotational speed": rng.integers(1200, 2800, n),
            "Torque": rng.normal(40, 5, n) + y.to_numpy() * 20,
            "Tool wear": rng.integers(0, 250, n),
        }
    )
    return X, y


def make_table(pr_auc: list[float], recall: list[float]) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([METRICS, ["mean", "std"]])
    table = pd.DataFrame(0.0, index=["a", "b", "c"], columns=columns)
    table[("pr_auc", "mean")] = pr_auc
    table[("recall", "mean")] = recall
    return table


def test_all_models_start_with_the_baselines() -> None:
    names = list(build_all_models())

    assert names[:2] == ["dummy", "logistic_regression"]
    assert names[2:] == list(models.build_candidates())
    assert len(names) == 9


def test_sort_by_metric_puts_the_best_model_first() -> None:
    table = make_table(pr_auc=[0.2, 0.9, 0.5], recall=[0.1, 0.2, 0.3])

    assert list(sort_by_metric(table).index) == ["b", "c", "a"]


def test_sort_by_metric_can_use_another_metric() -> None:
    table = make_table(pr_auc=[0.2, 0.9, 0.5], recall=[0.1, 0.2, 0.3])

    assert list(sort_by_metric(table, "recall").index) == ["c", "b", "a"]


def test_sort_by_metric_keeps_the_order_of_ties() -> None:
    table = make_table(pr_auc=[0.5, 0.5, 0.5], recall=[0.0, 0.0, 0.0])

    assert list(sort_by_metric(table).index) == ["a", "b", "c"]


@pytest.fixture
def fast_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small synthetic dataset and only the two baselines, for speed."""
    monkeypatch.setattr(comparison, "load_xy", lambda: make_xy())
    monkeypatch.setattr(comparison, "build_all_models", build_baselines)
    # Running the module as a script re-imports it, so patch the sources too.
    monkeypatch.setattr(data, "load_xy", lambda: make_xy())
    monkeypatch.setattr(models, "build_all_models", build_baselines)


def test_main_prints_models_sorted_by_pr_auc(
    fast_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["comparison"])

    comparison.main()

    out = capsys.readouterr().out
    assert "480 rows, 48 failures" in out  # 80% of 600 rows, 10% failures
    assert "ordered by mean PR-AUC" in out
    # logistic regression beats the dummy, so it is listed first
    assert out.index("logistic_regression") < out.index("dummy")


def test_main_accepts_repeats(
    fast_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["comparison", "--repeats", "2"])

    comparison.main()

    assert "2 repeat(s)" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_runs_main_when_executed_as_script(
    fast_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["comparison"])

    runpy.run_module("machine_failure.comparison", run_name="__main__")

    assert "logistic_regression" in capsys.readouterr().out
