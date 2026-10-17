"""Candidate models, each in the shared preprocessing pipeline."""

from typing import Any, ClassVar

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from machine_failure.data import RANDOM_STATE
from machine_failure.features import FeatureSet
from machine_failure.models.base import Model, SearchSpace, all_models, register
from machine_failure.models.baselines import build_baselines

# Roughly the number of normal rows per failure (about 28.5 in the training
# data). XGBoost has no ``class_weight``; this scales the weight of failures.
SCALE_POS_WEIGHT = 28.5


@register
class RandomForestModel(Model):
    name = "random_forest"
    role = "candidate"
    # The exact best-F1 threshold is 0.4746. 0.475 is the value that was chosen and
    # evaluated once on the test set, so it stays.
    threshold: ClassVar[float | None] = 0.475
    search_space: ClassVar[SearchSpace] = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_leaf": [1, 2, 3, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }
    tuned_settings: ClassVar[dict[str, Any]] = {
        "n_estimators": 500,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "class_weight": None,
    }

    def make_estimator(self) -> BaseEstimator:
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
        )


@register
class ExtraTreesModel(Model):
    name = "extra_trees"
    role = "candidate"
    threshold: ClassVar[float | None] = 0.416
    search_space: ClassVar[SearchSpace] = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_leaf": [1, 2, 3, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
        # "balanced_subsample" needs bootstrap sampling, which extra trees skips
        "model__class_weight": [None, "balanced"],
    }
    tuned_settings: ClassVar[dict[str, Any]] = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": 0.8,
        "class_weight": None,
    }

    def make_estimator(self) -> BaseEstimator:
        return ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        )


@register
class HistGradientBoostingModel(Model):
    name = "hist_gradient_boosting"
    role = "candidate"
    threshold: ClassVar[float | None] = 0.864
    search_space: ClassVar[SearchSpace] = {
        "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 300, 500],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 40],
        "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    }
    tuned_settings: ClassVar[dict[str, Any]] = {
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "class_weight": "balanced",
    }

    def make_estimator(self) -> BaseEstimator:
        return HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        )


@register
class XGBoostModel(Model):
    name = "xgboost"
    role = "candidate"
    threshold: ClassVar[float | None] = 0.654
    search_space: ClassVar[SearchSpace] = {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 4, 6, 8],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__scale_pos_weight": [1.0, 5.0, SCALE_POS_WEIGHT],
    }
    tuned_settings: ClassVar[dict[str, Any]] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "scale_pos_weight": 5.0,
    }

    def make_estimator(self) -> BaseEstimator:
        return XGBClassifier(
            n_estimators=300,
            scale_pos_weight=SCALE_POS_WEIGHT,
            random_state=RANDOM_STATE,
        )


@register
class KNNModel(Model):
    name = "knn"
    role = "candidate"

    def make_estimator(self) -> BaseEstimator:
        return KNeighborsClassifier(n_neighbors=15)


@register
class SVMModel(Model):
    name = "svm"
    role = "candidate"

    def make_estimator(self) -> BaseEstimator:
        return SVC(class_weight="balanced")


@register
class MLPModel(Model):
    name = "mlp"
    role = "candidate"

    def make_estimator(self) -> BaseEstimator:
        return MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=1000, random_state=RANDOM_STATE
        )


@register
class QDAModel(Model):
    name = "qda"
    role = "candidate"

    def make_estimator(self) -> BaseEstimator:
        return QuadraticDiscriminantAnalysis(reg_param=0.1)


def build_candidates(feature_set: FeatureSet = "all") -> dict[str, Pipeline]:
    """The candidate models, with default-ish settings and no tuning.

    Failures are about 3.4% of the data, so models that support it weight the
    classes inversely to their frequency (XGBoost through ``scale_pos_weight``).
    KNN, the MLP and quadratic discriminant analysis have no class weights and see
    the imbalance as it is. The SVM is scored through its decision function, which
    PR-AUC and ROC-AUC accept, so it skips the slow internal cross-validation that
    probabilities would need.

    ``feature_set`` chooses the features, as in ``build_preprocessor``.
    """
    return {
        name: model.build(feature_set)
        for name, model in all_models("candidate").items()
    }


def build_all_models(feature_set: FeatureSet = "all") -> dict[str, Pipeline]:
    """The baselines (the references to beat) followed by the candidates."""
    return {**build_baselines(feature_set), **build_candidates(feature_set)}
