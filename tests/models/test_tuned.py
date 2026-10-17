from collections.abc import Callable

import pandas as pd
import pytest

from machine_failure.evaluation import SETTING_PREFIX
from machine_failure.features import FEATURE_SETS, FeatureSet, build_preprocessor
from machine_failure.models import (
    FINAL_THRESHOLD,
    SEARCH_SPACES,
    TUNED_MODELS,
    TUNED_SETTINGS,
    all_models,
    build_candidates,
    build_tuned_models,
    get_model,
)

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


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
def test_feature_set_reaches_every_tuned_model(
    make_xy: MakeXY, feature_set: FeatureSet
) -> None:
    X, _ = make_xy(n=300)
    expected = list(build_preprocessor(feature_set).fit_transform(X).columns)

    for model in build_tuned_models(feature_set).values():
        assert (
            list(model.named_steps["preprocess"].fit_transform(X).columns) == expected
        )


@pytest.mark.parametrize("name", TUNED_MODELS)
def test_every_tuned_model_fits_and_predicts_probabilities(
    make_xy: MakeXY, name: str
) -> None:
    X, y = make_xy(n=300)

    model = build_tuned_models()[name].fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]
    assert probabilities.shape == (len(X),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities[y == 1].mean() > probabilities[y == 0].mean()


def test_search_spaces_and_settings_come_from_the_model_classes() -> None:
    tuned = {name: model for name, model in all_models().items() if model.is_tuned}

    assert list(tuned) == list(TUNED_MODELS)
    for name, model in tuned.items():
        assert SEARCH_SPACES[name] is model.search_space
        assert TUNED_SETTINGS[name] is model.tuned_settings
    assert not any(
        model.search_space or model.tuned_settings
        for name, model in all_models().items()
        if name not in tuned
    )


def test_every_tuned_model_has_its_own_threshold_and_the_others_none() -> None:
    for model in all_models().values():
        if model.is_tuned:
            assert 0 < model.own_threshold() < 1
        else:
            assert model.threshold is None
            with pytest.raises(ValueError, match="no threshold"):
                model.own_threshold()


def test_the_final_threshold_is_the_random_forest_threshold() -> None:
    assert FINAL_THRESHOLD == get_model("random_forest").own_threshold() == 0.475
