"""Choose a decision threshold from out-of-fold predictions.

A model gives each row a failure probability; a row is flagged as a failure when
its probability is at least the threshold. A low threshold catches more failures
(high recall) but raises more false alarms (low precision), and a high threshold
does the opposite. These functions show that trade-off on the training set.
"""

import argparse
import sys
from pathlib import Path

from machine_failure.cli import command
from machine_failure.data import load_xy, split_train_test
from machine_failure.evaluation import (
    DEFAULT_TARGETS,
    best_f1_per_model,
    compare_at_recall_targets,
    describe_run,
    format_table,
    out_of_fold_predictions,
    plot_precision_recall,
)
from machine_failure.models import build_tuned_models


@command
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
