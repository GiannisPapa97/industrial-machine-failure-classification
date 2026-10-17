"""Evaluate the final model once on the held-out test set.

The test set is meant to be used a single time, on the model and threshold chosen
with cross-validation on the training set. Nothing in this module should be used
to tune the model or the threshold.
"""

import argparse
import sys
from pathlib import Path

from machine_failure.config import N_BOOTSTRAP
from machine_failure.data import load_xy, split_train_test
from machine_failure.models import MODEL_PATH, load_final_model
from machine_failure.pipelines.evaluation import (
    cross_validated_reference,
    evaluate_final,
    format_evaluation,
)


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
