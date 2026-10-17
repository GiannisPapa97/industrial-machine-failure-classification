import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_recall_curve

from machine_failure.thresholds import (
    best_f1_per_model,
    best_f1_threshold,
    compare_at_recall_targets,
    format_table,
    plot_precision_recall,
    threshold_curve,
    threshold_for_recall,
)

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]

# A small case that can be worked out by hand: 3 failures and 3 normal rows.
LABELS = pd.Series([1, 1, 0, 1, 0, 0])
PROBABILITIES = pd.Series([0.9, 0.8, 0.7, 0.6, 0.4, 0.2])


def example_predictions() -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame({"label": LABELS, "probability": PROBABILITIES})
    worse = frame.assign(probability=[0.9, 0.3, 0.7, 0.6, 0.4, 0.2])
    return {"good": frame, "worse": worse}


def test_curve_counts_match_a_hand_calculation() -> None:
    curve = threshold_curve(LABELS, PROBABILITIES)

    assert curve["threshold"].tolist() == [0.9, 0.8, 0.7, 0.6, 0.4, 0.2]
    assert curve["tp"].tolist() == [1, 2, 2, 3, 3, 3]
    assert curve["fp"].tolist() == [0, 0, 1, 1, 2, 3]
    assert curve["fn"].tolist() == [2, 1, 1, 0, 0, 0]
    assert curve["tn"].tolist() == [3, 3, 2, 2, 1, 0]
    assert curve["precision"].tolist() == pytest.approx(
        [1, 1, 2 / 3, 3 / 4, 3 / 5, 1 / 2]
    )
    assert curve["recall"].tolist() == pytest.approx([1 / 3, 2 / 3, 2 / 3, 1, 1, 1])
    assert curve["f1"].tolist() == pytest.approx(
        [1 / 2, 4 / 5, 2 / 3, 6 / 7, 3 / 4, 2 / 3]
    )


def test_equal_probabilities_share_one_threshold() -> None:
    labels = pd.Series([1, 0, 1, 0])
    probabilities = pd.Series([0.5, 0.5, 0.2, 0.2])

    curve = threshold_curve(labels, probabilities)

    assert curve["threshold"].tolist() == [0.5, 0.2]
    assert curve["tp"].tolist() == [1, 2]
    assert curve["fp"].tolist() == [1, 2]


def test_curve_invariants_hold_on_random_data() -> None:
    rng = np.random.default_rng(1)
    labels = pd.Series(rng.random(500) < 0.1).astype(int)
    probabilities = pd.Series(rng.random(500) + 0.5 * labels)

    curve = threshold_curve(labels, probabilities)

    assert (curve["tp"] + curve["fn"] == labels.sum()).all()
    assert (curve["fp"] + curve["tn"] == (labels == 0).sum()).all()
    assert curve["threshold"].is_monotonic_decreasing
    assert curve["recall"].is_monotonic_increasing
    last = curve.iloc[-1]  # the lowest threshold flags every row
    assert (last["tp"], last["fp"], last["recall"]) == (
        labels.sum(),
        500 - labels.sum(),
        1,
    )


def test_curve_agrees_with_scikit_learn() -> None:
    rng = np.random.default_rng(2)
    labels = pd.Series(rng.random(400) < 0.15).astype(int)
    probabilities = pd.Series(rng.random(400) + 0.6 * labels)

    curve = threshold_curve(labels, probabilities).sort_values("threshold")
    precision, recall, thresholds_ = precision_recall_curve(labels, probabilities)

    assert curve["threshold"].to_numpy() == pytest.approx(thresholds_)
    assert curve["precision"].to_numpy() == pytest.approx(precision[:-1])
    assert curve["recall"].to_numpy() == pytest.approx(recall[:-1])


@pytest.mark.parametrize(
    ("target", "threshold"),
    [(0.3, 0.9), (0.6, 0.8), (0.7, 0.6), (1.0, 0.6)],
)
def test_threshold_for_recall_is_the_highest_that_reaches_the_target(
    target: float, threshold: float
) -> None:
    curve = threshold_curve(LABELS, PROBABILITIES)

    chosen = threshold_for_recall(curve, target)

    assert chosen["threshold"] == threshold
    assert chosen["recall"] >= target


@pytest.mark.parametrize("target", [0.0, -0.5, 1.2])
def test_an_invalid_recall_target_raises(target: float) -> None:
    curve = threshold_curve(LABELS, PROBABILITIES)

    with pytest.raises(ValueError, match="recall target"):
        threshold_for_recall(curve, target)


def test_best_f1_threshold_picks_the_highest_f1() -> None:
    curve = threshold_curve(LABELS, PROBABILITIES)

    best = best_f1_threshold(curve)

    assert best["threshold"] == 0.6
    assert best["f1"] == pytest.approx(6 / 7)


def test_best_f1_threshold_keeps_the_first_of_a_tie() -> None:
    curve = pd.DataFrame({"threshold": [0.8, 0.5, 0.2], "f1": [0.7, 0.7, 0.4]})

    assert best_f1_threshold(curve)["threshold"] == 0.8


def test_compare_at_recall_targets_has_a_row_per_target_and_model() -> None:
    table = compare_at_recall_targets(example_predictions(), targets=(0.6, 1.0))

    assert list(table.index) == [
        (0.6, "good"),
        (0.6, "worse"),
        (1.0, "good"),
        (1.0, "worse"),
    ]
    assert table.loc[(1.0, "good"), "threshold"] == 0.6
    assert table.loc[(1.0, "worse"), "threshold"] == 0.3
    # reaching full recall costs more false alarms for the weaker ranking
    assert table.loc[(1.0, "good"), "fp"] == 1
    assert table.loc[(1.0, "worse"), "fp"] == 2
    assert {"tp", "fp", "fn", "tn", "precision", "recall", "f1"} <= set(table.columns)


def test_best_f1_per_model_has_a_row_per_model() -> None:
    table = best_f1_per_model(example_predictions())

    assert list(table.index) == ["good", "worse"]
    assert table.loc["good", "threshold"] == 0.6


def test_format_table_shows_counts_as_integers_and_scores_to_three_decimals() -> None:
    table = compare_at_recall_targets(example_predictions(), targets=(1.0,))

    text = format_table(table)

    assert "0.857" in text  # good: F1 6/7 at full recall
    assert "0.750" in text  # good: precision 3/4
    assert "3.000" not in text  # counts are not shown as floats
    assert "100%" in text  # the recall target is shown as a percentage


def test_plot_writes_a_png_file(tmp_path: Path) -> None:
    path = tmp_path / "pr.png"

    plot_precision_recall(example_predictions(), path, targets=(0.6, 0.9))

    assert path.read_bytes().startswith(b"\x89PNG")


def test_importing_the_module_does_not_change_the_plotting_backend() -> None:
    code = (
        "import matplotlib; "
        "matplotlib.use('module://matplotlib_inline.backend_inline'); "
        "before = matplotlib.get_backend(); "
        "import machine_failure.thresholds; "
        "assert matplotlib.get_backend() == before, matplotlib.get_backend()"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
