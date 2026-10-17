"""Choose a decision threshold from out-of-fold predictions.

A model gives each row a failure probability; a row is flagged as a failure when
its probability is at least the threshold. A low threshold catches more failures
(high recall) but raises more false alarms (low precision), and a high threshold
does the opposite. These functions show that trade-off on the training set.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from machine_failure.baseline import describe_run
from machine_failure.data import load_xy, split_train_test
from machine_failure.predictions import out_of_fold_predictions
from machine_failure.tuned_models import build_tuned_models

DEFAULT_TARGETS = (0.80, 0.85, 0.90)
# The first four slots of the categorical palette, in order.
LINE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def threshold_curve(labels: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
    """Counts and scores at every threshold that changes the predictions.

    A row is flagged when its probability is at least the threshold. Returns one
    row per distinct probability, from the highest threshold (flags the fewest
    rows) to the lowest (flags every row), with ``tp``, ``fp``, ``fn``, ``tn``,
    ``precision``, ``recall`` and ``f1``.
    """
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    order = np.argsort(-p, kind="stable")
    p_sorted, y_sorted = p[order], y[order]
    last_of_each_value = np.r_[np.flatnonzero(np.diff(p_sorted)), len(p_sorted) - 1]

    tp = np.cumsum(y_sorted)[last_of_each_value]
    fp = np.cumsum(1 - y_sorted)[last_of_each_value]
    positives = int(y.sum())
    fn = positives - tp
    tn = (len(y) - positives) - fp
    return pd.DataFrame(
        {
            "threshold": p_sorted[last_of_each_value],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp),
            "recall": tp / positives,
            "f1": 2 * tp / (2 * tp + fp + fn),
        }
    )


def threshold_for_recall(curve: pd.DataFrame, target: float) -> pd.Series:
    """The highest threshold that still reaches at least ``target`` recall.

    Among the thresholds that reach the recall target, the highest one flags the
    fewest rows, so it has the fewest false alarms.
    """
    if not 0 < target <= 1:
        raise ValueError(f"the recall target must be in (0, 1], not {target}")
    return curve[curve["recall"] >= target].iloc[0]


def best_f1_threshold(curve: pd.DataFrame) -> pd.Series:
    """The threshold with the highest F1 (the first one if several tie)."""
    return curve.iloc[int(np.argmax(curve["f1"].to_numpy()))]


def compare_at_recall_targets(
    predictions: dict[str, pd.DataFrame],
    targets: Sequence[float] = DEFAULT_TARGETS,
) -> pd.DataFrame:
    """Each model's threshold, counts and scores at each recall target.

    ``predictions`` maps a model name to the output of ``out_of_fold_predictions``.
    Returns one row per (recall target, model).
    """
    rows = []
    for target in targets:
        for name, frame in predictions.items():
            curve = threshold_curve(frame["label"], frame["probability"])
            point = threshold_for_recall(curve, target).to_dict()
            rows.append({"target": target, "model": name, **point})
    return pd.DataFrame(rows).set_index(["target", "model"])


def best_f1_per_model(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each model's best-F1 threshold, counts and scores."""
    rows = {}
    for name, frame in predictions.items():
        curve = threshold_curve(frame["label"], frame["probability"])
        rows[name] = best_f1_threshold(curve)
    return pd.DataFrame(rows).T


def format_table(table: pd.DataFrame) -> str:
    """Counts as integers, scores to three decimals, recall targets as percentages."""
    counts = ["tp", "fp", "fn", "tn"]
    shown = table.copy()
    shown[counts] = shown[counts].astype(int)
    if "target" in shown.index.names:
        shown = shown.rename(index=lambda target: f"{target:.0%}", level="target")
    return shown.to_string(float_format=lambda value: f"{value:.3f}")


def plot_precision_recall(
    predictions: dict[str, pd.DataFrame],
    path: Path,
    targets: Sequence[float] = DEFAULT_TARGETS,
) -> None:
    """Save precision against recall for each model, with the recall targets marked.

    Draws with matplotlib's ``Figure`` directly, without pyplot, so importing this
    module never changes the plotting backend of a notebook or a script.
    """
    fig = Figure(figsize=(7.5, 5), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    for color, (name, frame) in zip(LINE_COLORS, predictions.items(), strict=False):
        curve = threshold_curve(frame["label"], frame["probability"])
        ax.plot(curve["recall"], curve["precision"], color=color, lw=2, label=name)
    for target in targets:
        ax.axvline(target, color=INK_MUTED, lw=1, ls=":")
        ax.text(
            target,
            0.02,
            f"{target:.0%} recall",
            rotation=90,
            ha="right",
            va="bottom",
            color=INK_MUTED,
            fontsize=9,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of failures caught)", color=INK_MUTED)
    ax.set_ylabel("Precision (share of alerts that are real)", color=INK_MUTED)
    ax.set_title(
        "Precision against recall, out-of-fold on the training set",
        loc="left",
        color=INK,
        fontweight="bold",
    )
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(loc="lower left", frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGETS),
        help="recall targets to compare (default: 0.80 0.85 0.90)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="save the precision-recall plot to this PNG file",
    )
    args = parser.parse_args()

    X, y = load_xy()
    X_train, _, y_train, _ = split_train_test(X, y)  # the test set is not used here
    predictions = {}
    for name, model in build_tuned_models().items():
        print(f"{name}: out-of-fold predictions", file=sys.stderr, flush=True)
        predictions[name] = out_of_fold_predictions(model, X_train, y_train)

    print(describe_run(X_train, y_train, 1))
    print(
        "Tuned models on all 9 features. A row is flagged when its probability "
        "is at least the threshold."
    )
    print("\nAt each recall target: the highest threshold that reaches it")
    print(format_table(compare_at_recall_targets(predictions, args.targets)))
    print("\nAt the threshold with the best F1")
    print(format_table(best_f1_per_model(predictions)))
    if args.plot is not None:
        plot_precision_recall(predictions, args.plot, args.targets)
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
