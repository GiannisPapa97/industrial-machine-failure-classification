"""The models: baselines, candidates, tuned configurations and the final model."""

from machine_failure.models.base import (
    Model,
    Role,
    SearchSpace,
    all_models,
    get_model,
    register,
)
from machine_failure.models.baselines import build_baselines
from machine_failure.models.candidates import (
    SCALE_POS_WEIGHT,
    build_all_models,
    build_candidates,
    with_preprocessing,
)
from machine_failure.models.final import (
    FINAL_MODEL,
    FINAL_THRESHOLD,
    MODEL_PATH,
    FinalModel,
    load_final_model,
    save_final_model,
    train_final_model,
)
from machine_failure.models.search_spaces import SEARCH_SPACES, TUNED_MODELS
from machine_failure.models.tuned import TUNED_SETTINGS, build_tuned_models

__all__ = [
    "FINAL_MODEL",
    "FINAL_THRESHOLD",
    "MODEL_PATH",
    "SCALE_POS_WEIGHT",
    "SEARCH_SPACES",
    "TUNED_MODELS",
    "TUNED_SETTINGS",
    "FinalModel",
    "Model",
    "Role",
    "SearchSpace",
    "all_models",
    "build_all_models",
    "build_baselines",
    "build_candidates",
    "build_tuned_models",
    "get_model",
    "load_final_model",
    "register",
    "save_final_model",
    "train_final_model",
    "with_preprocessing",
]
