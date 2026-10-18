"""Compare models with cross-validation on the training rows.

By default every registered model (the two baselines and the eight candidates)
is compared with default settings on all features. ``--feature-sets`` compares
them on each of several feature sets instead. The held-out test rows are not
used.
"""

import argparse

import pandas as pd

from machine_failure.config import ComparisonConfig, load_or_default
from machine_failure.evaluation import (
    METRICS,
    describe_cross_validation,
    format_comparison,
)
from machine_failure.experiments.ablation import (
    FEATURE_SET_DESCRIPTION,
    compare_feature_sets,
    format_feature_set_table,
)
from machine_failure.experiments.options import (
    add_config_option,
    add_data_option,
    data_option,
)
from machine_failure.features import FEATURE_SETS
from machine_failure.pipelines import ComparisonPipeline

HELP = "compare models with cross-validation"


def sort_by_metric(table: pd.DataFrame, metric: str = "pr_auc") -> pd.DataFrame:
    """Order the models from best to worst by the mean of ``metric``."""
    order = table[(metric, "mean")].sort_values(ascending=False, kind="stable")
    return table.reindex(order.index)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of ``mf compare``."""
    add_config_option(parser, "compare_default.toml")
    add_data_option(parser)
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
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SETS,
        help="compare the models on each of these feature sets, in one table",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="pr_auc",
        help="metric to order the models by (default: pr_auc)",
    )


def run(args: argparse.Namespace) -> None:
    """``mf compare``: cross-validate the models and print the comparison."""
    config = load_or_default(
        args.config,
        ComparisonConfig,
        models=tuple(args.models) if args.models else None,
        n_repeats=args.repeats,
        data=data_option(args),
    )
    pipeline = ComparisonPipeline(config)
    if args.feature_sets:
        X_train, y_train = pipeline.load_training_data()  # no test rows
        table = compare_feature_sets(
            X_train,
            y_train,
            feature_sets=tuple(args.feature_sets),
            models=config.models,
            n_splits=config.n_splits,
            n_repeats=config.n_repeats,
            random_state=config.random_state,
        )
        print(
            describe_cross_validation(
                len(X_train), int(y_train.sum()), config.n_repeats, config.n_splits
            )
        )
        print(FEATURE_SET_DESCRIPTION)
        print(
            f"Showing {args.metric}; models are ordered by their best mean across sets."
        )
        print(format_feature_set_table(table, args.metric))
        return
    result = pipeline.run()  # the test set is not used here
    print(result.describe())
    print(f"Models are ordered by mean {args.metric}, best first.")
    print(format_comparison(sort_by_metric(result.table, args.metric)))
