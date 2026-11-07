from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle

from machine_failure.evaluation import (
    METRICS,
    compare_models,
    comparison_figure,
    describe_cross_validation,
    format_comparison,
    plot_comparison,
)
from machine_failure.evaluation.comparison import BASELINE_COLOR, CANDIDATE_COLOR
from machine_failure.models import build_baselines

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


def test_compare_models_scores_each_model_on_every_metric(make_xy: MakeXY) -> None:
    X, y = make_xy()

    table = compare_models(build_baselines(), X, y)

    assert list(table.index) == ["dummy", "logistic_regression"]
    assert list(table.to_dict(orient="list")) == [
        (metric, stat) for metric in METRICS for stat in ("mean", "std")
    ]
    pr_auc = table[("pr_auc", "mean")].to_dict()
    roc_auc = table[("roc_auc", "mean")].to_dict()
    assert pr_auc["dummy"] == pytest.approx(0.1)
    assert roc_auc["dummy"] == pytest.approx(0.5)
    assert pr_auc["logistic_regression"] > 0.8


def test_format_comparison_shows_mean_and_std(make_xy: MakeXY) -> None:
    X, y = make_xy()
    table = compare_models(build_baselines(), X, y)

    text = format_comparison(table)

    assert "±" in text
    for name in ("dummy", "logistic_regression"):
        assert name in text
    for metric in METRICS:
        assert metric in text


def test_describe_cross_validation_names_the_folds_and_the_rows() -> None:
    assert describe_cross_validation(1_234, 42, 3, n_splits=4) == (
        "4-fold stratified cross-validation on the training set "
        "(1,234 rows, 42 failures), 3 repeat(s). Mean ± std across folds."
    )


def scored_table() -> pd.DataFrame:
    """Three models with known PR-AUC means and standard deviations."""
    columns = pd.MultiIndex.from_product([METRICS, ["mean", "std"]])
    table = pd.DataFrame(0.1, index=["dummy", "forest", "boosting"], columns=columns)
    table[("pr_auc", "mean")] = [0.03, 0.89, 0.85]
    table[("pr_auc", "std")] = [0.0, 0.03, 0.02]
    return table


def test_the_chart_puts_the_best_model_at_the_top() -> None:
    fig = comparison_figure(scored_table(), baselines=["dummy"])
    ax = fig.axes[0]

    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    bars = [bar for bar in ax.patches if isinstance(bar, Rectangle)]
    widths = [round(bar.get_width(), 2) for bar in bars]
    assert labels == ["dummy", "boosting", "forest"]  # bottom to top
    assert widths == [0.03, 0.85, 0.89]


def test_baselines_are_gray_with_a_legend_and_candidates_blue() -> None:
    fig = comparison_figure(scored_table(), baselines=["dummy"])
    ax = fig.axes[0]

    colors = [to_hex(bar.get_facecolor()) for bar in ax.patches]
    assert colors == [BASELINE_COLOR, CANDIDATE_COLOR, CANDIDATE_COLOR]
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "candidate",
        "baseline",
    ]


def test_only_the_best_value_is_printed() -> None:
    ax = comparison_figure(scored_table()).axes[0]

    assert [text.get_text() for text in ax.texts] == ["0.890"]
    assert ax.get_legend() is None  # no baselines, no legend


def test_the_axis_names_the_metric() -> None:
    ax = comparison_figure(scored_table(), metric="pr_auc").axes[0]

    assert ax.get_xlabel().startswith("PR-AUC, mean across folds")
    assert ax.get_xlim() == (0.0, 1.0)


def test_plot_comparison_writes_a_png_file(tmp_path: Path) -> None:
    path = tmp_path / "comparison.png"

    plot_comparison(scored_table(), path)

    assert path.read_bytes().startswith(b"\x89PNG")
