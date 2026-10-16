"""Randomized hyperparameter search with cross-validation on the training set."""

from typing import Any

import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from machine_failure.data import RANDOM_STATE
from machine_failure.evaluation import N_SPLITS, SCORING, cross_validate_model

MAIN_METRIC = "pr_auc"
SETTING_PREFIX = "model__"


def make_search(
    model: Pipeline,
    search_space: dict[str, list[Any]],
    *,
    n_iter: int,
    cv: Any,
    random_state: int,
    n_jobs: int | None = None,
) -> RandomizedSearchCV:
    """An unfitted randomized search that scores every metric and refits on PR-AUC."""
    return RandomizedSearchCV(
        model,
        search_space,
        n_iter=n_iter,
        scoring=SCORING,
        refit=MAIN_METRIC,
        cv=cv,
        random_state=random_state,
        error_score="raise",
        n_jobs=n_jobs,
    )


def tune_model(
    model: Pipeline,
    search_space: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 30,
    n_splits: int = N_SPLITS,
    n_repeats: int = 1,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = None,
) -> RandomizedSearchCV:
    """Try ``n_iter`` random settings from ``search_space`` and rank them by PR-AUC.

    Every setting is scored with stratified cross-validation on ``X`` and ``y``,
    which should be the training set only. All metrics in ``SCORING`` are
    recorded for each setting. The returned search is fitted: the best setting,
    chosen by PR-AUC, is refit on all of ``X`` and ``y`` as ``best_estimator_``.

    The best setting's cross-validation score is optimistic, because the setting
    was picked on those same folds. Use nested cross-validation to estimate how
    the tuned model performs on new data.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    search = make_search(
        model,
        search_space,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return search.fit(X, y)


def nested_cross_validate(
    model: Pipeline,
    search_space: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 20,
    n_splits: int = N_SPLITS,
    inner_splits: int = 3,
    n_repeats: int = 1,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Estimate how a tuned model performs on new data.

    In each outer fold the randomized search runs on the outer training part only
    (with ``inner_splits``-fold cross-validation inside it) and the setting it
    picks is scored on the held-out outer fold, which the search never saw. The
    outer folds are the same as in ``cross_validate_model``, so the scores are
    directly comparable with an untuned model. Returns one row per outer fold and
    one column per metric. ``n_jobs`` runs the outer folds in parallel.
    """
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state
    )
    search = make_search(
        model, search_space, n_iter=n_iter, cv=inner_cv, random_state=random_state
    )
    return cross_validate_model(
        search,
        X,
        y,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def best_settings(search: RandomizedSearchCV) -> dict[str, Any]:
    """The best setting found, with the pipeline prefix removed."""
    return {
        key.removeprefix(SETTING_PREFIX): value
        for key, value in search.best_params_.items()
    }


def search_results(search: RandomizedSearchCV) -> pd.DataFrame:
    """One row per setting tried, best PR-AUC first.

    Has a column per setting, then ``<metric>_mean`` and ``<metric>_std`` for
    every metric.
    """
    results = search.cv_results_
    table = pd.DataFrame(
        {
            key.removeprefix("param_").removeprefix(SETTING_PREFIX): values
            for key, values in results.items()
            if key.startswith("param_")
        }
    )
    for metric in SCORING:
        table[f"{metric}_mean"] = results[f"mean_test_{metric}"]
        table[f"{metric}_std"] = results[f"std_test_{metric}"]
    return table.sort_values(
        f"{MAIN_METRIC}_mean", ascending=False, kind="stable"
    ).reset_index(drop=True)
