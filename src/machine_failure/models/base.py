"""The common interface of every model, and the registry that lists them."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from machine_failure.evaluation import SETTING_PREFIX
from machine_failure.features import FeatureSet, build_preprocessor

SearchSpace = dict[str, list[Any]]
Role = Literal["baseline", "candidate"]


def with_preprocessing(
    model: BaseEstimator, feature_set: FeatureSet = "all"
) -> Pipeline:
    """Put ``model`` behind the preprocessing pipeline for ``feature_set``."""
    return Pipeline([("preprocess", build_preprocessor(feature_set)), ("model", model)])


class Model(ABC):
    """A model the project can compare, tune and train.

    A subclass names itself, says whether it is a baseline or a candidate, and
    builds its bare estimator. Everything else is shared: ``build`` puts the
    estimator behind the preprocessing pipeline, with optional settings applied.
    Models that are tuned also carry the space a search explores (keyed by pipeline
    parameter name, ``model__<setting>``) and the settings it found (keyed by the
    bare setting name).
    """

    name: ClassVar[str]
    role: ClassVar[Role]
    search_space: ClassVar[SearchSpace] = {}
    tuned_settings: ClassVar[dict[str, Any]] = {}

    @abstractmethod
    def make_estimator(self) -> BaseEstimator:
        """A fresh estimator with the model's default settings."""

    @property
    def is_tuned(self) -> bool:
        """Whether the model has a search space and tuned settings."""
        return bool(self.search_space)

    def build(
        self,
        feature_set: FeatureSet = "all",
        settings: dict[str, Any] | None = None,
    ) -> Pipeline:
        """The preprocessing pipeline for ``feature_set`` followed by the estimator.

        ``settings`` override estimator settings, by bare name (``max_depth``).
        """
        pipeline = with_preprocessing(self.make_estimator(), feature_set)
        if settings:
            pipeline.set_params(
                **{f"{SETTING_PREFIX}{key}": value for key, value in settings.items()}
            )
        return pipeline

    def build_tuned(self, feature_set: FeatureSet = "all") -> Pipeline:
        """``build`` with the tuned settings applied."""
        if not self.is_tuned:
            raise ValueError(f"{self.name} has no tuned settings")
        return self.build(feature_set, self.tuned_settings)


_REGISTRY: dict[str, Model] = {}


def register[M: type[Model]](model_class: M) -> M:
    """Class decorator that adds a model to the registry, in definition order."""
    model = model_class()
    if model.name in _REGISTRY:
        raise ValueError(f"a model named {model.name} is already registered")
    _REGISTRY[model.name] = model
    return model_class


def get_model(name: str) -> Model:
    """The registered model called ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; known: {list(_REGISTRY)}") from None


def all_models(role: Role | None = None) -> dict[str, Model]:
    """The registered models in registration order, optionally of one role."""
    return {
        name: model
        for name, model in _REGISTRY.items()
        if role is None or model.role == role
    }
