"""Hyperparameter search spaces for the models selected for tuning.

Each space maps a pipeline parameter name (``model__<setting>``) to the values a
randomized search may try. Lists are sampled uniformly. Every space includes the
default setting of the matching candidate in ``machine_failure.models``, so the
untuned model is always one of the options. The class-imbalance settings are part
of the search: class weights for the tree models, and ``scale_pos_weight`` for
XGBoost.

The spaces are defined on the model classes (``Model.search_space``); this module
collects them by model name.
"""

import machine_failure.models.candidates  # noqa: F401  (registers the models)
from machine_failure.models.base import SearchSpace, all_models

SEARCH_SPACES: dict[str, SearchSpace] = {
    name: model.search_space for name, model in all_models().items() if model.is_tuned
}

TUNED_MODELS = tuple(SEARCH_SPACES)
