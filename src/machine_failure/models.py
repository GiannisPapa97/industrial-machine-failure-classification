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

from machine_failure.baseline import build_baselines
from machine_failure.data import RANDOM_STATE
from machine_failure.preprocessing import build_preprocessor

# Roughly the number of normal rows per failure (about 28.5 in the training
# data). XGBoost has no ``class_weight``; this scales the weight of failures.
SCALE_POS_WEIGHT = 28.5


def with_preprocessing(model: BaseEstimator) -> Pipeline:
    """Put ``model`` behind the preprocessing pipeline."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def build_candidates() -> dict[str, Pipeline]:
    """The candidate models, with default-ish settings and no tuning.

    Failures are about 3.4% of the data, so models that support it weight the
    classes inversely to their frequency (XGBoost through ``scale_pos_weight``).
    KNN, the MLP and quadratic discriminant analysis have no class weights and see
    the imbalance as it is. The SVM is
    scored through its decision function, which PR-AUC and ROC-AUC accept, so it
    skips the slow internal cross-validation that probabilities would need.
    """
    return {
        "random_forest": with_preprocessing(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
            )
        ),
        "extra_trees": with_preprocessing(
            ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "hist_gradient_boosting": with_preprocessing(
            HistGradientBoostingClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
        "xgboost": with_preprocessing(
            XGBClassifier(
                n_estimators=300,
                scale_pos_weight=SCALE_POS_WEIGHT,
                random_state=RANDOM_STATE,
            )
        ),
        "knn": with_preprocessing(KNeighborsClassifier(n_neighbors=15)),
        "svm": with_preprocessing(SVC(class_weight="balanced")),
        "mlp": with_preprocessing(
            MLPClassifier(
                hidden_layer_sizes=(32, 16), max_iter=1000, random_state=RANDOM_STATE
            )
        ),
        "qda": with_preprocessing(QuadraticDiscriminantAnalysis(reg_param=0.1)),
    }


def build_all_models() -> dict[str, Pipeline]:
    """The baselines (the references to beat) followed by the candidates."""
    return {**build_baselines(), **build_candidates()}
