"""What the final model relies on, and which failures it misses.

Everything here uses the **training set only**, on out-of-fold predictions and
held-out folds, so it never looks at the test set. Studying the test set's
mistakes would invite adjusting the model to it, which would make the final
evaluation no longer honest.
"""

import argparse

from machine_failure.cli import command
from machine_failure.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_raw,
    split_train_test,
)
from machine_failure.evaluation import (
    group_medians,
    missed_by_failure_mode,
    out_of_fold_predictions,
    permutation_importances,
)
from machine_failure.models import FINAL_MODEL, FINAL_THRESHOLD, build_tuned_models


@command
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=FINAL_MODEL,
        help=f"tuned model to study (default: {FINAL_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=FINAL_THRESHOLD,
        help=f"decision threshold (default: {FINAL_THRESHOLD})",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="shuffles per feature and fold in the importance (default: 5)",
    )
    args = parser.parse_args()

    raw = load_raw()
    X, y = raw[FEATURE_COLUMNS].copy(), raw[TARGET_COLUMN].copy()
    X_train, _, y_train, _ = split_train_test(X, y)  # the test set is not used here
    model = build_tuned_models()[args.model]

    importance = permutation_importances(
        model, X_train, y_train, n_repeats=args.n_repeats
    )
    predictions = out_of_fold_predictions(model, X_train, y_train)
    flagged = predictions["probability"] >= args.threshold
    missed = int(((predictions["label"] == 1) & ~flagged).sum())

    print(
        f"{args.model} on the training set ({len(X_train):,} rows, "
        f"{int(y_train.sum())} failures), threshold {args.threshold}."
    )
    print("\nDrop in PR-AUC when one input feature is shuffled (held-out folds)")
    print(importance.to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"\nFailures caught and missed at the threshold ({missed} missed in total)")
    modes = missed_by_failure_mode(predictions, args.threshold, raw)
    print(modes.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nMedian measurements of normal rows, caught failures and missed failures")
    print(group_medians(X_train, predictions, args.threshold).round(2).to_string())


if __name__ == "__main__":
    main()
