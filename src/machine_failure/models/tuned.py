"""The tuned model configurations found by ``uv run mf tune``.

Each setting is the best configuration of a randomized search over the training
set (see the README for the nested cross-validation scores). The settings are
defined on the model classes (``Model.tuned_settings``), next to the search space
they come from; this module collects them by model name.
"""

from typing import Any

from sklearn.pipeline import Pipeline

import machine_failure.models.candidates  # noqa: F401  (registers the models)
from machine_failure.features import FeatureSet
from machine_failure.models.base import all_models

TUNED_SETTINGS: dict[str, dict[str, Any]] = {
    name: model.tuned_settings for name, model in all_models().items() if model.is_tuned
}


def build_tuned_models(feature_set: FeatureSet = "all") -> dict[str, Pipeline]:
    """The four tuned models, each in the preprocessing pipeline.

    Every call builds new, unfitted pipelines, so the models can be cloned,
    fitted and cross-validated independently.
    """
    return {
        name: model.build_tuned(feature_set)
        for name, model in all_models().items()
        if model.is_tuned
    }
