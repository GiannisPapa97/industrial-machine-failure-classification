"""Hyperparameter search spaces for the models selected for tuning.

Each space maps a pipeline parameter name (``model__<setting>``) to the values a
randomized search may try. Lists are sampled uniformly. Every space includes the
default setting of the matching candidate in ``machine_failure.models``, so the
untuned model is always one of the options. The class-imbalance settings are part
of the search: class weights for the tree models, and ``scale_pos_weight`` for
XGBoost.
"""

from machine_failure.models.base import SearchSpace
from machine_failure.models.candidates import SCALE_POS_WEIGHT

SEARCH_SPACES: dict[str, SearchSpace] = {
    "random_forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_leaf": [1, 2, 3, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    },
    "extra_trees": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_leaf": [1, 2, 3, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
        # "balanced_subsample" needs bootstrap sampling, which extra trees skips
        "model__class_weight": [None, "balanced"],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 300, 500],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 40],
        "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "xgboost": {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 4, 6, 8],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__scale_pos_weight": [1.0, 5.0, SCALE_POS_WEIGHT],
    },
}

TUNED_MODELS = tuple(SEARCH_SPACES)
