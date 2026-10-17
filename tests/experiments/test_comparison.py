import runpy
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from machine_failure import models
from machine_failure.config import ComparisonConfig, load_config
from machine_failure.evaluation import METRICS
from machine_failure.experiments import comparison
from machine_failure.experiments.comparison import sort_by_metric
from machine_failure.models import build_all_models, build_baselines
from machine_failure.pipelines import ComparisonPipeline, splits

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


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
    assert len(names) == 10


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
def fast_run(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small synthetic dataset and only the two baselines, for speed."""
    monkeypatch.setattr(splits, "load_xy", lambda path: make_xy())
    monkeypatch.setattr(
        ComparisonPipeline, "build_models", lambda self: build_baselines()
    )


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


@pytest.fixture
def synthetic_data(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """A small synthetic dataset; the models are whatever the command asks for."""
    monkeypatch.setattr(splits, "load_xy", lambda path: make_xy())


@pytest.mark.usefixtures("synthetic_data")
def test_main_compares_only_the_requested_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["comparison", "--models", "dummy", "logistic_regression"]
    monkeypatch.setattr(sys, "argv", argv)

    comparison.main()

    out = capsys.readouterr().out
    assert "logistic_regression" in out and "random_forest" not in out


@pytest.mark.usefixtures("synthetic_data")
def test_main_reads_the_settings_from_a_config_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config = tmp_path / "compare.toml"
    config.write_text(
        'models = ["dummy", "logistic_regression"]\nn_splits = 3\nn_repeats = 2\n'
        "[data]\ntest_size = 0.5\n"
    )
    monkeypatch.setattr(sys, "argv", ["comparison", "--config", str(config)])

    comparison.main()

    out = capsys.readouterr().out
    assert "3-fold" in out and "300 rows, 30 failures" in out and "2 repeat(s)" in out
    assert "random_forest" not in out


@pytest.mark.usefixtures("synthetic_data")
def test_command_line_options_override_the_config_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config = tmp_path / "compare.toml"
    config.write_text('models = ["dummy", "knn"]\nn_repeats = 2\n')
    argv = [
        "comparison",
        "--config",
        str(config),
        "--models",
        "dummy",
        "--repeats",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    comparison.main()

    out = capsys.readouterr().out
    assert "1 repeat(s)" in out and "knn" not in out


def test_the_example_config_is_the_default_comparison() -> None:
    example = Path(__file__).parents[2] / "configs" / "compare_default.toml"

    assert load_config(example, ComparisonConfig) == ComparisonConfig()


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_runs_main_when_executed_as_script(
    fast_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["comparison"])

    runpy.run_module("machine_failure.experiments.comparison", run_name="__main__")

    assert "logistic_regression" in capsys.readouterr().out
