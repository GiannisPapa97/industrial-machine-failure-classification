from collections.abc import Callable

import pandas as pd
import pytest
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from machine_failure.evaluation import cross_validate_model
from machine_failure.features import FEATURE_SETS, FeatureSet, build_preprocessor
from machine_failure.models import (
    SCALE_POS_WEIGHT,
    all_models,
    build_all_models,
    build_candidates,
    with_preprocessing,
)

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]

EXPECTED_MODELS = {
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "xgboost": XGBClassifier,
    "knn": KNeighborsClassifier,
    "svm": SVC,
    "mlp": MLPClassifier,
    "qda": QuadraticDiscriminantAnalysis,
}


def test_candidates_are_the_expected_models() -> None:
    candidates = build_candidates()

    assert list(candidates) == list(EXPECTED_MODELS)
    for name, model_type in EXPECTED_MODELS.items():
        assert candidates[name].steps[0][0] == "preprocess"
        assert isinstance(candidates[name].steps[-1][1], model_type)


@pytest.mark.parametrize(
    "name", ["random_forest", "extra_trees", "hist_gradient_boosting", "svm"]
)
def test_models_that_support_it_weight_the_classes(name: str) -> None:
    model = build_candidates()[name].steps[-1][1]

    assert str(model.class_weight).startswith("balanced")


def test_xgboost_weights_failures_through_scale_pos_weight() -> None:
    model = build_candidates()["xgboost"].steps[-1][1]

    assert model.scale_pos_weight == SCALE_POS_WEIGHT > 1


@pytest.mark.parametrize(
    "name",
    ["random_forest", "extra_trees", "hist_gradient_boosting", "xgboost", "mlp"],
)
def test_random_models_are_seeded(name: str) -> None:
    model = build_candidates()[name].steps[-1][1]

    assert model.random_state is not None


def test_candidates_are_independent_objects() -> None:
    first, second = build_candidates(), build_candidates()

    assert first["knn"] is not second["knn"]
    assert first["knn"].steps[0][1] is not first["svm"].steps[0][1]


def test_with_preprocessing_puts_the_model_last() -> None:
    model = KNeighborsClassifier()

    pipeline = with_preprocessing(model)

    assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]
    assert pipeline.steps[-1][1] is model


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("name", list(EXPECTED_MODELS))
def test_every_candidate_learns_a_clear_signal(make_xy: MakeXY, name: str) -> None:
    X, y = make_xy()

    scores = cross_validate_model(build_candidates()[name], X, y, n_splits=3)

    assert len(scores) == 3
    assert scores["pr_auc"].mean() > 0.5  # the failure rate is 0.1


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_feature_set_reaches_every_model(
    make_xy: MakeXY, feature_set: FeatureSet
) -> None:
    X, _ = make_xy()
    expected = list(build_preprocessor(feature_set).fit_transform(X).columns)

    for model in build_all_models(feature_set).values():
        assert (
            list(model.named_steps["preprocess"].fit_transform(X).columns) == expected
        )


def test_with_preprocessing_accepts_a_feature_set(make_xy: MakeXY) -> None:
    X, _ = make_xy()

    pipeline = with_preprocessing(KNeighborsClassifier(), "computed")

    columns = pipeline.named_steps["preprocess"].fit_transform(X).columns
    assert list(columns) == ["temp_diff", "power", "strain"]


def test_registry_lists_the_candidates_after_the_baselines() -> None:
    assert list(all_models()) == ["dummy", "logistic_regression", *EXPECTED_MODELS]
    assert list(all_models("candidate")) == list(EXPECTED_MODELS)
