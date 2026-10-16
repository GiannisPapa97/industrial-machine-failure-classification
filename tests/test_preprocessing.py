import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from machine_failure.preprocessing import (
    SCALED_FEATURES,
    TYPE_CATEGORIES,
    build_preprocessor,
)

EXPECTED_COLUMNS = [*(f"Type_{c}" for c in TYPE_CATEGORIES), *SCALED_FEATURES]


def make_features(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Type": rng.choice(TYPE_CATEGORIES, size=n),
            "Air temperature": rng.normal(300, 2, n),
            "Process temperature": rng.normal(310, 1.5, n),
            "Rotational speed": rng.integers(1200, 2800, n),
            "Torque": rng.normal(40, 10, n),
            "Tool wear": rng.integers(0, 250, n),
        },
        index=np.arange(n) + 100,
    )


def test_output_is_a_named_dataframe_with_eleven_columns() -> None:
    features = make_features()

    result = build_preprocessor().fit_transform(features)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == EXPECTED_COLUMNS
    assert len(EXPECTED_COLUMNS) == 11
    assert result.index.equals(features.index)
    assert not result.isna().any().any()


def test_type_is_one_hot_encoded() -> None:
    features = make_features()

    result = build_preprocessor().fit_transform(features)

    one_hot = result[["Type_L", "Type_M", "Type_H"]]
    assert (one_hot.sum(axis=1) == 1).all()
    for category in TYPE_CATEGORIES:
        assert (one_hot[f"Type_{category}"] == (features["Type"] == category)).all()


def test_numeric_features_are_standardized_on_the_fit_data() -> None:
    result = build_preprocessor().fit_transform(make_features())

    scaled = result[SCALED_FEATURES]
    assert scaled.mean().to_numpy() == pytest.approx(0.0, abs=1e-9)
    assert scaled.std(ddof=0).to_numpy() == pytest.approx(1.0)


def test_statistics_come_from_the_training_data_only() -> None:
    train = make_features(seed=0)
    other = make_features(seed=1).assign(**{"Air temperature": 400.0})
    preprocessor = build_preprocessor().fit(train)

    result = preprocessor.transform(other)

    mean = train["Air temperature"].mean()
    std = train["Air temperature"].std(ddof=0)
    assert result["Air temperature"].to_numpy() == pytest.approx((400.0 - mean) / std)


def test_all_type_columns_exist_when_a_type_is_missing_from_the_data() -> None:
    features = make_features()
    only_l = features.assign(Type="L")

    result = build_preprocessor().fit_transform(only_l)

    assert list(result.columns) == EXPECTED_COLUMNS
    assert (result["Type_L"] == 1).all()
    assert (result[["Type_M", "Type_H"]] == 0).all().all()


def test_unknown_type_raises() -> None:
    preprocessor = build_preprocessor().fit(make_features())

    with pytest.raises(ValueError, match="Found unknown categories"):
        preprocessor.transform(make_features().assign(Type="X"))


def test_can_be_cloned_and_does_not_share_fitted_state() -> None:
    fitted = build_preprocessor().fit(make_features())

    copy = clone(fitted)

    assert not hasattr(copy.steps[0][1], "feature_names_in_")
