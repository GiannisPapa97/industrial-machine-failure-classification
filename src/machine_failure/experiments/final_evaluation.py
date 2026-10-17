"""Evaluate the final model once on the held-out test set.

The test set is meant to be used a single time, on the model and threshold chosen
with cross-validation on the training set. Nothing in this module should be used
to tune the model or the threshold.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from machine_failure.data import RANDOM_STATE, load_xy, split_train_test
from machine_failure.evaluation import out_of_fold_predictions
from machine_failure.models import (
    MODEL_PATH,
    FinalModel,
    build_tuned_models,
    load_final_model,
)

METRICS = ("pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy")
COUNTS = ("tp", "fp", "fn", "tn")
N_BOOTSTRAP = 1000


def metrics_at_threshold(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """The metrics and counts of the alerts raised at ``threshold``.

    A row is flagged when its probability is at least the threshold. PR-AUC and
    ROC-AUC do not depend on the threshold; the rest do. Precision and F1 are 0
    when nothing is flagged.
    """
    flagged = probabilities >= threshold
    failure = labels == 1
    tp = int((flagged & failure).sum())
    fp = int((flagged & ~failure).sum())
    fn = int((~flagged & failure).sum())
    tn = int((~flagged & ~failure).sum())
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "accuracy": (tp + tn) / len(labels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample rows with replacement, keeping the number of failures fixed.

    Failures and normal rows are resampled separately, so no resample ends up
    without failures, however few there are.
    """
    failures = np.flatnonzero(labels == 1)
    normal = np.flatnonzero(labels == 0)
    return np.concatenate(
        [
            rng.choice(failures, size=len(failures), replace=True),
            rng.choice(normal, size=len(normal), replace=True),
        ]
    )


@dataclass(frozen=True)
class FinalEvaluation:
    """The test-set result: scores, counts and bootstrap intervals."""

    name: str
    threshold: float
    n_rows: int
    n_failures: int
    scores: dict[str, float]  # the metrics and counts on the test set
    intervals: dict[str, tuple[float, float]]  # 95% bootstrap interval per metric


def evaluate_final(
    final: FinalModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> FinalEvaluation:
    """Score ``final`` on the test set, with a 95% bootstrap interval per metric.

    The model is not changed. The interval comes from resampling the test rows
    ``n_bootstrap`` times, which shows how much the score could move with a
    different sample of machines; with only a few dozen failures, it is wide.
    """
    labels = y_test.to_numpy()
    probabilities = final.probability(X_test).to_numpy()
    scores = metrics_at_threshold(labels, probabilities, final.threshold)

    rng = np.random.default_rng(random_state)
    resampled: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(n_bootstrap):
        rows = bootstrap_indices(labels, rng)
        result = metrics_at_threshold(
            labels[rows], probabilities[rows], final.threshold
        )
        for metric in METRICS:
            resampled[metric].append(result[metric])
    intervals = {
        metric: (float(low), float(high))
        for metric, (low, high) in (
            (metric, np.percentile(values, [2.5, 97.5]))
            for metric, values in resampled.items()
        )
    }
    return FinalEvaluation(
        name=final.name,
        threshold=final.threshold,
        n_rows=len(labels),
        n_failures=int(labels.sum()),
        scores=scores,
        intervals=intervals,
    )


def cross_validated_reference(
    name: str, threshold: float, X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, float]:
    """The same metrics from out-of-fold predictions on the training set.

    This is the estimate the threshold was chosen with, to compare with the test
    result.
    """
    frame = out_of_fold_predictions(build_tuned_models()[name], X_train, y_train)
    return metrics_at_threshold(
        frame["label"].to_numpy(), frame["probability"].to_numpy(), threshold
    )


def format_evaluation(
    result: FinalEvaluation, reference: dict[str, float] | None = None
) -> str:
    """The metrics with their intervals, and the counts of alerts."""
    rows = {}
    for metric in METRICS:
        low, high = result.intervals[metric]
        rows[metric] = {
            "test set": f"{result.scores[metric]:.3f}",
            "95% interval": f"[{low:.3f}, {high:.3f}]",
        }
        if reference is not None:
            rows[metric]["cross-validation"] = f"{reference[metric]:.3f}"
    scores = result.scores
    lines = [
        f"{result.name} at threshold {result.threshold} on the test set "
        f"({result.n_rows:,} rows, {result.n_failures} failures)",
        pd.DataFrame(rows).T.to_string(),
        "",
        f"correct alerts {scores['tp']:.0f}, false alarms {scores['fp']:.0f}, "
        f"missed failures {scores['fn']:.0f}, correct non-alerts {scores['tn']:,.0f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="required: confirms you want to use the test set, which is meant to "
        "be done once",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"the saved final model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help=f"bootstrap resamples for the intervals (default: {N_BOOTSTRAP})",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="do not compute the cross-validation reference (saves about a minute)",
    )
    args = parser.parse_args()

    if not args.confirm:
        print(
            "The test set is meant to be used once, on the final model. "
            "Run again with --confirm to evaluate on it.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    X, y = load_xy()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    final = load_final_model(args.model_path)
    result = evaluate_final(final, X_test, y_test, n_bootstrap=args.n_bootstrap)
    reference = None
    if not args.skip_reference:
        reference = cross_validated_reference(
            final.name, final.threshold, X_train, y_train
        )
    print(format_evaluation(result, reference))


if __name__ == "__main__":
    main()
