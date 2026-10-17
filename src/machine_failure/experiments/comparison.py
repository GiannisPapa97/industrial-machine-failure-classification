"""Compare the baselines and the candidate models with cross-validation."""

import argparse

import pandas as pd

from machine_failure.config import ComparisonConfig
from machine_failure.evaluation import format_comparison
from machine_failure.pipelines import ComparisonPipeline


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

    # the test set is not used here
    result = ComparisonPipeline(ComparisonConfig(n_repeats=args.repeats)).run()

    print(result.describe())
    print("Models are ordered by mean PR-AUC, best first.")
    print(format_comparison(sort_by_metric(result.table)))


if __name__ == "__main__":
    main()
