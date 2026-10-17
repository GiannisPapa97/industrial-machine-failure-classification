from collections.abc import Callable

import pandas as pd
import pytest
from sklearn.base import clone

from machine_failure.models import (
    SCALE_POS_WEIGHT,
    SEARCH_SPACES,
    TUNED_MODELS,
    build_candidates,
)

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]

# Settings that make a model cheap to fit, so every option can be fitted in tests.
CHEAP = {
    "random_forest": {"model__n_estimators": 10},
    "extra_trees": {"model__n_estimators": 10},
    "hist_gradient_boosting": {"model__max_iter": 10},
    "xgboost": {"model__n_estimators": 10},
}
IMBALANCE_SETTINGS = ("model__class_weight", "model__scale_pos_weight")


def test_spaces_cover_the_four_selected_models() -> None:
    assert TUNED_MODELS == (
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
    )
    assert set(SEARCH_SPACES) <= set(build_candidates())


@pytest.mark.parametrize("name", TUNED_MODELS)
def test_every_setting_is_a_pipeline_parameter_with_several_values(name: str) -> None:
    pipeline = build_candidates()[name]
    valid = pipeline.get_params()

    for key, values in SEARCH_SPACES[name].items():
        assert key.startswith("model__")
        assert key in valid, f"{key} is not a parameter of {name}"
        assert len(values) > 1
        assert len(set(map(repr, values))) == len(values)  # no duplicates


@pytest.mark.parametrize("name", TUNED_MODELS)
def test_the_default_setting_is_one_of_the_options(name: str) -> None:
    params = build_candidates()[name].get_params()

    for key, values in SEARCH_SPACES[name].items():
        if params[key] is not None:  # None means the library's own default
            assert params[key] in values, f"{key}: default {params[key]!r}"


def test_xgboost_space_includes_the_current_class_imbalance_setting() -> None:
    assert SCALE_POS_WEIGHT in SEARCH_SPACES["xgboost"]["model__scale_pos_weight"]


@pytest.mark.parametrize("name", TUNED_MODELS)
def test_every_imbalance_option_can_be_fitted(make_xy: MakeXY, name: str) -> None:
    X, y = make_xy(n=400)
    settings = [k for k in SEARCH_SPACES[name] if k in IMBALANCE_SETTINGS]
    assert settings  # every tuned model searches a class-imbalance setting

    for key in settings:
        for value in SEARCH_SPACES[name][key]:
            pipeline = clone(build_candidates()[name])
            pipeline.set_params(**CHEAP[name], **{key: value})
            pipeline.fit(X, y)
            assert pipeline.predict(X).shape == (len(X),)
