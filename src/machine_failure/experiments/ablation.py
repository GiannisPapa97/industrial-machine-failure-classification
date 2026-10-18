"""Compare models on the raw, computed and combined feature sets.

Used by ``mf compare --feature-sets``.
"""

import logging

import pandas as pd

from machine_failure.config import ComparisonConfig
from machine_failure.data import RANDOM_STATE
from machine_failure.evaluation import N_SPLITS
from machine_failure.features import FEATURE_SETS, FeatureSet
from machine_failure.models import all_models
from machine_failure.pipelines import ComparisonPipeline

logger = logging.getLogger(__name__)

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
) -> pd.DataFrame:
    """Cross-validate the models (all by default) on each feature set.

    Returns one row per model and a (feature set, metric, mean/std) column for
    every combination. Progress is logged, since a full run takes a few minutes.
    """
    tables = {}
    for feature_set in feature_sets:
        logger.info("cross-validating on feature set: %s", feature_set)
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
