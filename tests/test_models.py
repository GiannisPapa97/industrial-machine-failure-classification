import numpy as np
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
from machine_failure.models import (
    SCALE_POS_WEIGHT,
    build_candidates,
    with_preprocessing,
)

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


def make_xy(n: int = 600, failure_rate: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic features with the real columns; failures have a higher torque."""
    rng = np.random.default_rng(0)
    n_failures = int(n * failure_rate)
    y = pd.Series([1] * n_failures + [0] * (n - n_failures), name="Machine failure")
    X = pd.DataFrame(
        {
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Air temperature": rng.normal(300, 2, n),
            "Process temperature": rng.normal(310, 1.5, n),
            "Rotational speed": rng.integers(1200, 2800, n),
            "Torque": rng.normal(40, 5, n) + y.to_numpy() * 20,
            "Tool wear": rng.integers(0, 250, n),
        }
    )
    return X, y


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
def test_every_candidate_learns_a_clear_signal(name: str) -> None:
    X, y = make_xy()

    scores = cross_validate_model(build_candidates()[name], X, y, n_splits=3)

    assert len(scores) == 3
    assert scores["pr_auc"].mean() > 0.5  # the failure rate is 0.1
