"""The tuned model configurations found by ``python -m machine_failure.tune``.

Each setting is the best configuration of a randomized search over the training
set (see the README for the nested cross-validation scores). The names match the
output of ``tuning.best_settings`` and the ``search_spaces`` they come from.
"""

from typing import Any

from sklearn.pipeline import Pipeline

from machine_failure.features import FeatureSet
from machine_failure.models import build_candidates
from machine_failure.tuning import SETTING_PREFIX

TUNED_SETTINGS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "class_weight": None,
    },
    "extra_trees": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": 0.8,
        "class_weight": None,
    },
    "hist_gradient_boosting": {
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "class_weight": "balanced",
    },
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "scale_pos_weight": 5.0,
    },
}


def build_tuned_models(feature_set: FeatureSet = "all") -> dict[str, Pipeline]:
    """The four tuned models, each in the preprocessing pipeline.

    Every call builds new, unfitted pipelines, so the models can be cloned,
    fitted and cross-validated independently.
    """
    candidates = build_candidates(feature_set)
    tuned = {}
    for name, settings in TUNED_SETTINGS.items():
        pipeline = candidates[name]
        pipeline.set_params(
            **{f"{SETTING_PREFIX}{key}": value for key, value in settings.items()}
        )
        tuned[name] = pipeline
    return tuned
