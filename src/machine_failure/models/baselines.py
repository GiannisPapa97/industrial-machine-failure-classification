"""The baseline models every other model has to beat."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from machine_failure.data import RANDOM_STATE
from machine_failure.features import FeatureSet, build_preprocessor


def build_baselines(feature_set: FeatureSet = "all") -> dict[str, Pipeline]:
    """The two reference models every later model has to beat.

    - ``dummy``: ignores the features and always predicts the majority class. It
      shows what "no skill" scores on this imbalanced data.
    - ``logistic_regression``: a simple, interpretable model, with classes
      weighted inversely to their frequency to counter the imbalance.
    """
    return {
        "dummy": Pipeline(
            [
                ("preprocess", build_preprocessor(feature_set)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor(feature_set)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
