"""The baseline models every other model has to beat."""

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from machine_failure.data import RANDOM_STATE
from machine_failure.features import FeatureSet
from machine_failure.models.base import Model, all_models, register


@register
class Dummy(Model):
    """Ignores the features and always predicts the majority class.

    It shows what "no skill" scores on this imbalanced data.
    """

    name = "dummy"
    role = "baseline"

    def make_estimator(self) -> BaseEstimator:
        return DummyClassifier(strategy="prior")


@register
class LogisticRegressionModel(Model):
    """A simple, interpretable model, with classes weighted inversely to their
    frequency to counter the imbalance."""

    name = "logistic_regression"
    role = "baseline"

    def make_estimator(self) -> BaseEstimator:
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        )


def build_baselines(feature_set: FeatureSet = "all") -> dict[str, Pipeline]:
    """The two reference models every later model has to beat."""
    return {
        name: model.build(feature_set) for name, model in all_models("baseline").items()
    }
