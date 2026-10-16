import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from machine_failure.features import DERIVED_FEATURES, FeatureEngineer


def make_features() -> pd.DataFrame:
    """A small feature table with the columns that load_xy returns."""
    return pd.DataFrame(
        {
            "Type": ["M", "L", "H"],
            "Air temperature": [298.0, 300.0, 302.5],
            "Process temperature": [308.0, 310.5, 311.0],
            "Rotational speed": [1500, 2000, 1200],
            "Torque": [40.0, 20.0, 60.0],
            "Tool wear": [100, 0, 250],
        },
        index=[10, 20, 30],
    )


def test_computes_the_derived_features() -> None:
    result = FeatureEngineer().fit_transform(make_features())

    assert result["temp_diff"].tolist() == pytest.approx([10.0, 10.5, 8.5])
    # torque [Nm] * speed [rpm] * 2 * pi / 60 gives power in watts
    expected_power = [40 * 1500, 20 * 2000, 60 * 1200]
    assert result["power"].tolist() == pytest.approx(
        [p * 2 * np.pi / 60 for p in expected_power]
    )
    assert result["strain"].tolist() == pytest.approx([4000.0, 0.0, 15000.0])


def test_keeps_input_columns_and_appends_new_ones() -> None:
    features = make_features()

    result = FeatureEngineer().fit_transform(features)

    assert list(result.columns) == [*features.columns, *DERIVED_FEATURES]
    assert_frame_equal(result[features.columns], features)
    assert result.index.tolist() == [10, 20, 30]


def test_does_not_modify_the_input() -> None:
    features = make_features()
    original = features.copy()

    FeatureEngineer().fit_transform(features)

    assert_frame_equal(features, original)


def test_get_feature_names_out() -> None:
    features = make_features()
    engineer = FeatureEngineer().fit(features)

    assert list(engineer.get_feature_names_out()) == [
        *features.columns,
        *DERIVED_FEATURES,
    ]
    assert list(engineer.get_feature_names_out(["a", "b"])) == [
        "a",
        "b",
        *DERIVED_FEATURES,
    ]


def test_transform_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        FeatureEngineer().transform(make_features())


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_missing_columns_raise(method: str) -> None:
    features = make_features().drop(columns=["Torque"])

    with pytest.raises(ValueError, match="missing columns.*Torque"):
        getattr(FeatureEngineer(), method)(features)


def test_transform_with_missing_columns_raises() -> None:
    engineer = FeatureEngineer().fit(make_features())

    with pytest.raises(ValueError, match="missing columns.*Tool wear"):
        engineer.transform(make_features().drop(columns=["Tool wear"]))


def test_existing_derived_column_raises() -> None:
    features = make_features().assign(power=0.0)

    with pytest.raises(ValueError, match="already present.*power"):
        FeatureEngineer().fit(features)


def test_transform_columns_must_match_fit() -> None:
    features = make_features()
    engineer = FeatureEngineer().fit(features)
    reordered = features[features.columns[::-1]]
    extra = features.assign(Extra=1)

    with pytest.raises(ValueError, match="differ from those seen in fit"):
        engineer.transform(reordered)
    with pytest.raises(ValueError, match="differ from those seen in fit"):
        engineer.transform(extra)


def test_can_be_cloned() -> None:
    assert isinstance(clone(FeatureEngineer()), FeatureEngineer)


def test_works_inside_a_pipeline() -> None:
    pipeline = Pipeline(
        [
            ("features", FeatureEngineer()),
            (
                "scale",
                ColumnTransformer(
                    [("derived", StandardScaler(), DERIVED_FEATURES)],
                    remainder="drop",
                ),
            ),
        ]
    )

    result = pipeline.fit_transform(make_features())

    assert result.shape == (3, 3)
    assert result.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0])
