from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RandomizedSearchCV

from machine_failure.evaluation import SCORING
from machine_failure.models import build_candidates
from machine_failure.tuning import (
    best_settings,
    search_results,
    tune_model,
)

SPACE: dict[str, list[Any]] = {
    "model__n_estimators": [10, 20],
    "model__max_depth": [3, None],
    "model__min_samples_leaf": [1, 5],
}


def make_xy(n: int = 400, failure_rate: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic features with the real columns; failures have a higher torque."""
    rng = np.random.default_rng(0)
    n_failures = int(n * failure_rate)
    y = pd.Series([1] * n_failures + [0] * (n - n_failures), name="Machine failure")
    X = pd.DataFrame(
        {
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Air temperature": rng.normal(300, 2, n),
            "Process temperature": rng.normal(310, 1.5, n),
            "Rotational speed": rng.integers(1200, 2800, n),
            "Torque": rng.normal(40, 5, n) + y.to_numpy() * 20,
            "Tool wear": rng.integers(0, 250, n),
        }
    )
    return X, y


@pytest.fixture(scope="module")
def search() -> RandomizedSearchCV:
    X, y = make_xy()
    return tune_model(build_candidates()["random_forest"], SPACE, X, y, n_iter=4)


def test_tune_model_returns_a_fitted_search_with_a_refit_best_model(
    search: RandomizedSearchCV,
) -> None:
    X, _ = make_xy()

    assert set(search.best_params_) == set(SPACE)
    assert search.best_estimator_.predict(X).shape == (len(X),)


def test_every_setting_is_scored_on_every_metric(search: RandomizedSearchCV) -> None:
    results = search.cv_results_

    assert len(results["params"]) == 4  # n_iter settings
    for metric in SCORING:
        assert f"mean_test_{metric}" in results
        assert f"std_test_{metric}" in results


def test_the_best_setting_is_chosen_by_pr_auc(search: RandomizedSearchCV) -> None:
    pr_auc = search.cv_results_["mean_test_pr_auc"]

    assert search.best_index_ == int(np.argmax(pr_auc))
    assert search.best_score_ == pytest.approx(pr_auc.max())


def test_best_settings_drops_the_pipeline_prefix(search: RandomizedSearchCV) -> None:
    settings = best_settings(search)

    assert set(settings) == {"n_estimators", "max_depth", "min_samples_leaf"}
    assert settings["n_estimators"] == search.best_params_["model__n_estimators"]


def test_search_results_lists_settings_and_metrics_best_first(
    search: RandomizedSearchCV,
) -> None:
    table = search_results(search)

    assert len(table) == 4
    assert set(table.columns[:3]) == {"n_estimators", "max_depth", "min_samples_leaf"}
    for metric in SCORING:
        assert f"{metric}_mean" in table.columns
        assert f"{metric}_std" in table.columns
    assert table["pr_auc_mean"].is_monotonic_decreasing
    assert table.loc[0, "pr_auc_mean"] == pytest.approx(search.best_score_)


def test_the_same_seed_gives_the_same_search() -> None:
    X, y = make_xy()
    model = build_candidates()["random_forest"]

    first = tune_model(model, SPACE, X, y, n_iter=3)
    again = tune_model(model, SPACE, X, y, n_iter=3)
    other = tune_model(model, SPACE, X, y, n_iter=3, random_state=7)

    assert first.cv_results_["params"] == again.cv_results_["params"]
    assert first.cv_results_["params"] != other.cv_results_["params"]


def test_an_invalid_setting_raises_instead_of_being_ignored() -> None:
    X, y = make_xy()

    with pytest.raises(ValueError, match="not_a_setting"):
        tune_model(
            build_candidates()["random_forest"],
            {"model__not_a_setting": [1, 2]},
            X,
            y,
            n_iter=2,
        )
