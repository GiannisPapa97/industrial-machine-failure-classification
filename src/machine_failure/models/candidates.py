"""Candidate models, each in the shared preprocessing pipeline."""

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
from machine_failure.models.base import Model, all_models, register
from machine_failure.models.baselines import build_baselines

# Roughly the number of normal rows per failure (about 28.5 in the training
# data). XGBoost has no ``class_weight``; this scales the weight of failures.
SCALE_POS_WEIGHT = 28.5


@register
class RandomForestModel(Model):
    name = "random_forest"
    role = "candidate"

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

    def make_estimator(self) -> BaseEstimator:
        return ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        )


@register
class HistGradientBoostingModel(Model):
    name = "hist_gradient_boosting"
    role = "candidate"

    def make_estimator(self) -> BaseEstimator:
        return HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        )


@register
class XGBoostModel(Model):
    name = "xgboost"
    role = "candidate"

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
