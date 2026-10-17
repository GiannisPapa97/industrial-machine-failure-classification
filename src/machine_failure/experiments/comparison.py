"""Compare the baselines and the candidate models with cross-validation."""

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from machine_failure.config import ComparisonConfig, load_config
from machine_failure.evaluation import format_comparison
from machine_failure.pipelines import ComparisonPipeline


def sort_by_metric(table: pd.DataFrame, metric: str = "pr_auc") -> pd.DataFrame:
    """Order the models from best to worst by the mean of ``metric``."""
    order = table[(metric, "mean")].sort_values(ascending=False, kind="stable")
    return table.reindex(order.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML file with the comparison settings (see configs/compare_default."
        "toml); the options below override it",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="compare only these models (default: all of them)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="repeat the cross-validation this many times (default: 1)",
    )
    args = parser.parse_args()

    config = load_config(args.config, ComparisonConfig) if args.config else None
    config = config or ComparisonConfig()
    overrides = {
        "models": tuple(args.models) if args.models else None,
        "n_repeats": args.repeats,
    }
    config = dataclasses.replace(
        config, **{key: value for key, value in overrides.items() if value is not None}
    )
    result = ComparisonPipeline(config).run()  # the test set is not used here

    print(result.describe())
    print("Models are ordered by mean PR-AUC, best first.")
    print(format_comparison(sort_by_metric(result.table)))


if __name__ == "__main__":
    main()
