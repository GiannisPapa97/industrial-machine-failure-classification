"""Compare all models on the raw, computed and combined feature sets."""

import argparse
import sys

import pandas as pd

from machine_failure.config import ComparisonConfig
from machine_failure.data import RANDOM_STATE
from machine_failure.evaluation import METRICS, N_SPLITS, describe_run
from machine_failure.features import FEATURE_SETS, FeatureSet
from machine_failure.models import all_models
from machine_failure.pipelines import ComparisonPipeline

FEATURE_SET_DESCRIPTION = (
    "Feature sets: raw = Type and the five measurements; "
    "computed = temp_diff, power and strain only; all = raw and computed."
)


def compare_feature_sets(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    feature_sets: tuple[FeatureSet, ...] = FEATURE_SETS,
    models: tuple[str, ...] | None = None,
    n_splits: int = N_SPLITS,
    n_repeats: int = 1,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
) -> pd.DataFrame:
    """Cross-validate the models (all by default) on each feature set.

    Returns one row per model and a (feature set, metric, mean/std) column for
    every combination. ``verbose`` reports progress on stderr, since a full run
    takes a few minutes.
    """
    tables = {}
    for feature_set in feature_sets:
        if verbose:
            print(f"cross-validating on feature set: {feature_set}", file=sys.stderr)
        config = ComparisonConfig(
            models=models or tuple(all_models()),
            feature_set=feature_set,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        tables[feature_set] = ComparisonPipeline(config).compare(X, y).table
    return pd.concat(tables, axis=1)


def format_feature_set_table(table: pd.DataFrame, metric: str = "pr_auc") -> str:
    """Models as rows, feature sets as columns, ``mean ± std`` of ``metric``.

    Models are ordered by their best mean across the feature sets.
    """
    feature_sets = list(dict.fromkeys(column[0] for column in table.columns))
    means = pd.DataFrame({fs: table[(fs, metric, "mean")] for fs in feature_sets})
    order = means.max(axis=1).sort_values(ascending=False, kind="stable").index
    cells = pd.DataFrame(
        {
            fs: table[(fs, metric, "mean")].map("{:.3f}".format)
            + " ± "
            + table[(fs, metric, "std")].map("{:.3f}".format)
            for fs in feature_sets
        }
    )
    return cells.reindex(order).to_string()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="pr_auc",
        help="metric to show in the table (default: pr_auc)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="compare only these models (default: all of them)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeat the cross-validation this many times (default: 1)",
    )
    args = parser.parse_args()

    # the test set is not used here
    pipeline = ComparisonPipeline(ComparisonConfig(n_repeats=args.repeats))
    X_train, y_train = pipeline.load_training_data()
    models = tuple(args.models) if args.models else None
    table = compare_feature_sets(
        X_train, y_train, models=models, n_repeats=args.repeats, verbose=True
    )

    print(describe_run(X_train, y_train, args.repeats))
    print(FEATURE_SET_DESCRIPTION)
    print(f"Showing {args.metric}; models are ordered by their best mean across sets.")
    print(format_feature_set_table(table, args.metric))


if __name__ == "__main__":
    main()
