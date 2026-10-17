"""Compare several models with cross-validation, on the same folds."""

import pandas as pd
from sklearn.base import BaseEstimator

from machine_failure.data import RANDOM_STATE
from machine_failure.evaluation.cross_validation import (
    N_SPLITS,
    SCORING,
    cross_validate_model,
    summarize_scores,
)

METRICS = list(SCORING)


def compare_models(
    models: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = 1,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Cross-validate each model on the same folds.

    Returns one row per model and a (metric, mean/std) column for every metric.
    """
    summaries = {
        name: summarize_scores(
            cross_validate_model(
                model,
                X,
                y,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        ).stack()
        for name, model in models.items()
    }
    return pd.DataFrame(summaries).T


def format_comparison(table: pd.DataFrame) -> str:
    """Render the comparison as ``mean ± std`` per metric."""
    cells = pd.DataFrame(
        {
            metric: table[metric]["mean"].map("{:.3f}".format)
            + " ± "
            + table[metric]["std"].map("{:.3f}".format)
            for metric in METRICS
        }
    )
    return cells.to_string()


def describe_run(X_train: pd.DataFrame, y_train: pd.Series, n_repeats: int) -> str:
    """One line describing how the models were cross-validated."""
    return (
        f"{N_SPLITS}-fold stratified cross-validation on the training set "
        f"({len(X_train):,} rows, {int(y_train.sum())} failures), "
        f"{n_repeats} repeat(s). Mean ± std across folds."
    )
