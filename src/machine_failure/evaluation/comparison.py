"""Compare several models with cross-validation, on the same folds."""

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.base import BaseEstimator

from machine_failure.data import RANDOM_STATE
from machine_failure.evaluation.cross_validation import (
    N_SPLITS,
    SCORING,
    cross_validate_model,
    summarize_scores,
)
from machine_failure.evaluation.thresholds import (
    GRID,
    INK,
    INK_MUTED,
    LINE_COLORS,
    SURFACE,
)

logger = logging.getLogger(__name__)

METRICS = list(SCORING)
METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "accuracy": "Accuracy",
}
CANDIDATE_COLOR = LINE_COLORS[0]
BASELINE_COLOR = "#8a8984"  # neutral gray, 3.4:1 against the chart surface


def compare_models(
    models: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = 1,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Cross-validate each model on the same folds.

    Returns one row per model and a (metric, mean/std) column for every metric.
    """
    summaries = {}
    for name, model in models.items():
        logger.info("cross-validating %s", name)
        scores = cross_validate_model(
            model,
            X,
            y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        summaries[name] = summarize_scores(scores).stack()
    return pd.DataFrame(summaries).T


def format_comparison(table: pd.DataFrame) -> str:
    """Render the comparison as ``mean ± std`` per metric."""
    cells = pd.DataFrame(
        {
            metric: table[metric]["mean"].map("{:.3f}".format)
            + " ± "
            + table[metric]["std"].map("{:.3f}".format)
            for metric in METRICS
        }
    )
    return cells.to_string()


def describe_cross_validation(
    n_rows: int, n_failures: int, n_repeats: int, n_splits: int = N_SPLITS
) -> str:
    """One line describing how the models were cross-validated."""
    return (
        f"{n_splits}-fold stratified cross-validation on the training set "
        f"({n_rows:,} rows, {n_failures} failures), "
        f"{n_repeats} repeat(s). Mean ± std across folds."
    )


def describe_run(X_train: pd.DataFrame, y_train: pd.Series, n_repeats: int) -> str:
    """``describe_cross_validation`` for a training set and the default folds."""
    return describe_cross_validation(len(X_train), int(y_train.sum()), n_repeats)


def comparison_figure(
    table: pd.DataFrame, metric: str = "pr_auc", baselines: Sequence[str] = ()
) -> Figure:
    """A bar per model: the mean of ``metric`` across folds, best at the top.

    Each bar has a line for one standard deviation either side. Models named in
    ``baselines`` are gray and the others blue, with a legend; only the best
    value is printed, the axis carries the rest. Drawn with matplotlib's
    ``Figure`` directly, without pyplot, so the plotting backend is left alone.
    """
    means = table[(metric, "mean")].sort_values(kind="stable")  # best last = top
    stds = table[(metric, "std")].reindex(means.index)
    names = [str(name) for name in means.index]
    colors = [BASELINE_COLOR if n in baselines else CANDIDATE_COLOR for n in names]

    fig = Figure(figsize=(7.5, 1.4 + 0.42 * len(names)), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    positions = list(range(len(names)))
    ax.barh(positions, means.to_numpy(), height=0.34, color=colors)
    ax.errorbar(
        means.to_numpy(),
        positions,
        xerr=stds.to_numpy(),
        fmt="none",
        ecolor=INK_MUTED,
        elinewidth=1,
        capsize=3,
    )
    best = float(means.iloc[-1])
    ax.text(
        best + float(stds.iloc[-1]) + 0.015,
        positions[-1],
        f"{best:.3f}",
        va="center",
        color=INK,
        fontsize=9,
    )
    ax.set_yticks(positions, names)
    ax.set_xlim(0, 1)
    label = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel(
        f"{label}, mean across folds (lines: one standard deviation)", color=INK_MUTED
    )
    ax.set_title(
        f"{label} of each model, cross-validated on the training set",
        loc="left",
        color=INK,
        fontweight="bold",
    )
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    if any(name in baselines for name in names):
        handles = [
            Patch(color=CANDIDATE_COLOR, label="candidate"),
            Patch(color=BASELINE_COLOR, label="baseline"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False, labelcolor=INK)
    fig.tight_layout()
    return fig


def plot_comparison(
    table: pd.DataFrame,
    path: Path,
    metric: str = "pr_auc",
    baselines: Sequence[str] = (),
) -> None:
    """Save ``comparison_figure`` as a PNG file at ``path``."""
    comparison_figure(table, metric, baselines).savefig(
        path, dpi=150, facecolor=SURFACE
    )
