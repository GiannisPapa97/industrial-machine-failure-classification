import numpy as np
import pandas as pd
import pytest

from machine_failure.models import build_candidates
from machine_failure.preprocessing import FEATURE_SETS, FeatureSet, build_preprocessor
from machine_failure.search_spaces import SEARCH_SPACES, TUNED_MODELS
from machine_failure.tuned_models import TUNED_SETTINGS, build_tuned_models
from machine_failure.tuning import SETTING_PREFIX


def make_xy(n: int = 300, failure_rate: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
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


def test_the_four_tuned_models_are_the_searched_ones() -> None:
    assert tuple(TUNED_SETTINGS) == TUNED_MODELS
    assert list(build_tuned_models()) == list(TUNED_MODELS)


@pytest.mark.parametrize("name", TUNED_MODELS)
def test_settings_are_options_of_the_search_space(name: str) -> None:
    space = SEARCH_SPACES[name]
    settings = TUNED_SETTINGS[name]

    # every searched setting has a value, and it is one the search could pick
    assert {f"{SETTING_PREFIX}{key}" for key in settings} == set(space)
    for key, value in settings.items():
        assert value in space[f"{SETTING_PREFIX}{key}"]


@pytest.mark.parametrize("name", TUNED_MODELS)
def test_models_carry_the_tuned_settings(name: str) -> None:
    params = build_tuned_models()[name].get_params()

    for key, value in TUNED_SETTINGS[name].items():
        assert params[f"{SETTING_PREFIX}{key}"] == value


def test_building_tuned_models_does_not_change_the_candidates() -> None:
    build_tuned_models()

    forest = build_candidates()["random_forest"].get_params()
    assert forest["model__n_estimators"] == 300
    assert forest["model__class_weight"] == "balanced_subsample"


def test_each_call_returns_new_unfitted_models() -> None:
    first, second = build_tuned_models(), build_tuned_models()

    assert first["random_forest"] is not second["random_forest"]
    assert (
        first["random_forest"].steps[-1][1] is not second["random_forest"].steps[-1][1]
    )


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_feature_set_reaches_every_tuned_model(feature_set: FeatureSet) -> None:
    X, _ = make_xy()
    expected = list(build_preprocessor(feature_set).fit_transform(X).columns)

    for model in build_tuned_models(feature_set).values():
        assert (
            list(model.named_steps["preprocess"].fit_transform(X).columns) == expected
        )


@pytest.mark.parametrize("name", TUNED_MODELS)
def test_every_tuned_model_fits_and_predicts_probabilities(name: str) -> None:
    X, y = make_xy()

    model = build_tuned_models()[name].fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]
    assert probabilities.shape == (len(X),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities[y == 1].mean() > probabilities[y == 0].mean()
