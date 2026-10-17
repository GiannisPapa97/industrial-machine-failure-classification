"""Compare the baselines and the candidate models with cross-validation."""

import argparse

import pandas as pd

from machine_failure.data import load_xy, split_train_test
from machine_failure.evaluation import compare_models, describe_run, format_comparison
from machine_failure.models import build_all_models


def sort_by_metric(table: pd.DataFrame, metric: str = "pr_auc") -> pd.DataFrame:
    """Order the models from best to worst by the mean of ``metric``."""
    order = table[(metric, "mean")].sort_values(ascending=False, kind="stable")
    return table.reindex(order.index)


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
    table = compare_models(build_all_models(), X_train, y_train, n_repeats=args.repeats)

    print(describe_run(X_train, y_train, args.repeats))
    print("Models are ordered by mean PR-AUC, best first.")
    print(format_comparison(sort_by_metric(table)))


if __name__ == "__main__":
    main()
