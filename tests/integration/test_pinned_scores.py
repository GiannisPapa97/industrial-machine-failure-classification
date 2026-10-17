"""Cross-validation scores of the models on the real training set, as they are today.

These are 5-fold stratified cross-validation scores on the 8,000 training rows, with
the fixed seeds. The values are rounded to four decimals, hence the tolerance.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from sklearn.model_selection import ParameterSampler

from machine_failure.evaluation import cross_validate_model, summarize_scores
from machine_failure.models import (
    SEARCH_SPACES,
    build_all_models,
    build_baselines,
    build_candidates,
    build_tuned_models,
)

TOLERANCE = 2e-4

# mean of every metric, then the standard deviation of PR-AUC across the folds
DEFAULT_MODELS: dict[str, tuple[dict[str, float], float]] = {
    "dummy": (
        {
            "pr_auc": 0.0339,
            "roc_auc": 0.5,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "accuracy": 0.9661,
        },
        0.0003,
    ),
    "logistic_regression": (
        {
            "pr_auc": 0.4586,
            "roc_auc": 0.9233,
            "recall": 0.8415,
            "precision": 0.1618,
            "f1": 0.2713,
            "accuracy": 0.8466,
        },
        0.0466,
    ),
    "random_forest": (
        {
            "pr_auc": 0.8902,
            "roc_auc": 0.9713,
            "recall": 0.749,
            "precision": 0.9852,
            "f1": 0.8501,
            "accuracy": 0.9911,
        },
        0.0304,
    ),
    "extra_trees": (
        {
            "pr_auc": 0.8243,
            "roc_auc": 0.9708,
            "recall": 0.5461,
            "precision": 0.9474,
            "f1": 0.6922,
            "accuracy": 0.9836,
        },
        0.0263,
    ),
    "hist_gradient_boosting": (
        {
            "pr_auc": 0.8886,
            "roc_auc": 0.9715,
            "recall": 0.8228,
            "precision": 0.8991,
            "f1": 0.858,
            "accuracy": 0.9907,
        },
        0.0166,
    ),
    "xgboost": (
        {
            "pr_auc": 0.8553,
            "roc_auc": 0.9707,
            "recall": 0.8007,
            "precision": 0.8285,
            "f1": 0.8141,
            "accuracy": 0.9876,
        },
        0.0291,
    ),
    "knn": (
        {
            "pr_auc": 0.6153,
            "roc_auc": 0.9155,
            "recall": 0.24,
            "precision": 0.9175,
            "f1": 0.3797,
            "accuracy": 0.9735,
        },
        0.047,
    ),
    "svm": (
        {
            "pr_auc": 0.6825,
            "roc_auc": 0.962,
            "recall": 0.8707,
            "precision": 0.3086,
            "f1": 0.4552,
            "accuracy": 0.9293,
        },
        0.0252,
    ),
    "mlp": (
        {
            "pr_auc": 0.7949,
            "roc_auc": 0.9713,
            "recall": 0.6681,
            "precision": 0.7828,
            "f1": 0.7188,
            "accuracy": 0.9822,
        },
        0.0307,
    ),
    "qda": (
        {
            "pr_auc": 0.4442,
            "roc_auc": 0.9227,
            "recall": 0.3615,
            "precision": 0.5608,
            "f1": 0.4391,
            "accuracy": 0.9688,
        },
        0.0419,
    ),
}

TUNED_MODELS: dict[str, tuple[dict[str, float], float]] = {
    "random_forest": (
        {
            "pr_auc": 0.911,
            "roc_auc": 0.9781,
            "recall": 0.8192,
            "precision": 0.9868,
            "f1": 0.8947,
            "accuracy": 0.9935,
        },
        0.0247,
    ),
    "extra_trees": (
        {
            "pr_auc": 0.8723,
            "roc_auc": 0.9764,
            "recall": 0.7123,
            "precision": 0.9801,
            "f1": 0.8242,
            "accuracy": 0.9898,
        },
        0.0295,
    ),
    "hist_gradient_boosting": (
        {
            "pr_auc": 0.8933,
            "roc_auc": 0.9763,
            "recall": 0.8301,
            "precision": 0.8959,
            "f1": 0.8605,
            "accuracy": 0.9909,
        },
        0.018,
    ),
    "xgboost": (
        {
            "pr_auc": 0.8693,
            "roc_auc": 0.977,
            "recall": 0.7971,
            "precision": 0.8785,
            "f1": 0.8354,
            "accuracy": 0.9894,
        },
        0.0329,
    ),
}

# PR-AUC of two models on each feature set
FEATURE_SET_SCORES = {
    ("raw", "logistic_regression"): 0.4409,
    ("computed", "logistic_regression"): 0.3241,
    ("all", "logistic_regression"): 0.4586,
    ("raw", "hist_gradient_boosting"): 0.7926,
    ("computed", "hist_gradient_boosting"): 0.7017,
    ("all", "hist_gradient_boosting"): 0.8886,
}

# The first of the 20 random settings the searches try, with the fixed seed
FIRST_DRAW: dict[str, dict[str, Any]] = {
    "random_forest": {
        "model__n_estimators": 300,
        "model__min_samples_leaf": 5,
        "model__max_features": "sqrt",
        "model__max_depth": 8,
        "model__class_weight": None,
    },
    "extra_trees": {
        "model__n_estimators": 500,
        "model__min_samples_leaf": 2,
        "model__max_features": 0.8,
        "model__max_depth": 8,
        "model__class_weight": None,
    },
    "hist_gradient_boosting": {
        "model__min_samples_leaf": 10,
        "model__max_leaf_nodes": 31,
        "model__max_iter": 500,
        "model__learning_rate": 0.02,
        "model__l2_regularization": 0.1,
        "model__class_weight": None,
    },
    "xgboost": {
        "model__subsample": 0.6,
        "model__scale_pos_weight": 28.5,
        "model__n_estimators": 300,
        "model__min_child_weight": 1,
        "model__max_depth": 8,
        "model__learning_rate": 0.05,
        "model__colsample_bytree": 1.0,
    },
}


def scores_of(model: Any, data: SimpleNamespace) -> tuple[dict[str, float], float]:
    """Mean of every metric and the standard deviation of PR-AUC over the folds."""
    summary = summarize_scores(cross_validate_model(model, data.X_train, data.y_train))
    means = {str(metric): float(value) for metric, value in summary["mean"].items()}
    return means, float(summary["std"].to_dict()["pr_auc"])


@pytest.mark.parametrize("name", list(DEFAULT_MODELS))
def test_the_default_model_scores(real_data: SimpleNamespace, name: str) -> None:
    expected_means, expected_std = DEFAULT_MODELS[name]

    means, std = scores_of(build_all_models()[name], real_data)

    assert means == pytest.approx(expected_means, abs=TOLERANCE)
    assert std == pytest.approx(expected_std, abs=TOLERANCE)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("random_forest", marks=pytest.mark.slow),
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
    ],
)
def test_the_tuned_model_scores(real_data: SimpleNamespace, name: str) -> None:
    expected_means, expected_std = TUNED_MODELS[name]

    means, std = scores_of(build_tuned_models()[name], real_data)

    assert means == pytest.approx(expected_means, abs=TOLERANCE)
    assert std == pytest.approx(expected_std, abs=TOLERANCE)


@pytest.mark.parametrize(("feature_set", "name"), list(FEATURE_SET_SCORES))
def test_the_feature_set_scores(
    real_data: SimpleNamespace, feature_set: Any, name: str
) -> None:
    models = {**build_baselines(feature_set), **build_candidates(feature_set)}

    means, _ = scores_of(models[name], real_data)

    assert means["pr_auc"] == pytest.approx(
        FEATURE_SET_SCORES[(feature_set, name)], abs=TOLERANCE
    )


@pytest.mark.parametrize("name", list(FIRST_DRAW))
def test_the_search_samples_the_same_settings(name: str) -> None:
    first = list(ParameterSampler(SEARCH_SPACES[name], 20, random_state=42))[0]

    assert first == FIRST_DRAW[name]
