"""Baseline models and their cross-validated comparison."""

import argparse

from machine_failure.cli import command
from machine_failure.data import load_xy, split_train_test
from machine_failure.evaluation import compare_models, describe_run, format_comparison
from machine_failure.models import build_baselines


@command
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeat the cross-validation this many times (default: 1)",
    )
    args = parser.parse_args()

    X, y = load_xy()
    X_train, _, y_train, _ = split_train_test(X, y)  # the test set is not used here
    table = compare_models(build_baselines(), X_train, y_train, n_repeats=args.repeats)

    print(describe_run(X_train, y_train, args.repeats))
    print(format_comparison(table))


if __name__ == "__main__":
    main()
