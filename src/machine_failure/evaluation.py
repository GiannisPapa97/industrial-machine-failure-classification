"""Cross-validated evaluation of classifiers on the training set."""

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from machine_failure.data import RANDOM_STATE

N_SPLITS = 5

# PR-AUC (average precision) is the main metric: it needs no threshold and
# reflects the rare failure class. Precision, recall and F1 use the default
# decision threshold; precision is 0 when a model predicts no failures at all.
SCORING: dict[str, Any] = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "accuracy": "accuracy",
}


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = 1,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Score ``model`` with stratified k-fold cross-validation.

    Pass the training set only; the test set stays untouched until the final
    evaluation. ``model`` is cloned and refit on every training fold, so put the
    preprocessing inside it (a ``Pipeline``) to keep it from seeing the held-out
    fold. Returns one row per fold and one column per metric; ``n_repeats`` > 1
    repeats the whole procedure with different shuffles for steadier estimates.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    result = cross_validate(model, X, y, cv=cv, scoring=SCORING, error_score="raise")
    return pd.DataFrame({name: result[f"test_{name}"] for name in SCORING})


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across the folds."""
    return scores.agg(["mean", "std"]).T
