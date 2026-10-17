"""The derived features and the preprocessing, as they are today."""

from types import SimpleNamespace

import pytest

from machine_failure.features import (
    DERIVED_FEATURES,
    FEATURE_SETS,
    SCALED_FEATURES,
    FeatureEngineer,
    build_preprocessor,
)

COLUMNS = {
    "raw": [
        "Type_L",
        "Type_M",
        "Type_H",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ],
    "computed": ["temp_diff", "power", "strain"],
    "all": ["Type_L", "Type_M", "Type_H", *SCALED_FEATURES],
}
SHAPES = {"raw": (8000, 8), "computed": (8000, 3), "all": (8000, 11)}


def test_the_derived_features_of_the_first_rows(real_data: SimpleNamespace) -> None:
    derived = FeatureEngineer().fit_transform(real_data.X)[DERIVED_FEATURES]

    first = derived.head(3).to_dict("list")

    assert first["temp_diff"] == pytest.approx([10.5, 10.5, 10.4])
    assert first["power"] == pytest.approx([6951.59056, 6826.722724, 7749.387543])
    assert first["strain"] == pytest.approx([0.0, 138.9, 247.0])


def test_the_derived_features_over_the_whole_dataset(
    real_data: SimpleNamespace,
) -> None:
    derived = FeatureEngineer().fit_transform(real_data.X)[DERIVED_FEATURES]

    assert derived.mean().round(3).to_dict() == pytest.approx(
        {"temp_diff": 10.001, "power": 6279.745, "strain": 4314.665}, abs=1e-3
    )
    assert derived.max().to_dict() == pytest.approx(
        {"temp_diff": 12.1, "power": 10469.923, "strain": 16497.0}, abs=1e-3
    )


def test_the_feature_engineer_keeps_the_input_columns(
    real_data: SimpleNamespace,
) -> None:
    result = FeatureEngineer().fit_transform(real_data.X)

    assert list(result.columns) == [*real_data.X.columns, *DERIVED_FEATURES]
    assert len(result) == 10_000


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_the_preprocessing_output_per_feature_set(
    real_data: SimpleNamespace, feature_set: str
) -> None:
    preprocessor = build_preprocessor(feature_set).fit(real_data.X_train)  # type: ignore[arg-type]

    train = preprocessor.transform(real_data.X_train)
    test = preprocessor.transform(real_data.X_test)

    assert train.shape == SHAPES[feature_set]
    assert test.shape == (2000, SHAPES[feature_set][1])
    assert list(train.columns) == COLUMNS[feature_set]
    assert list(test.columns) == COLUMNS[feature_set]
    assert not train.isna().any().any()


def test_the_scaling_is_learned_on_the_training_set_only(
    real_data: SimpleNamespace,
) -> None:
    preprocessor = build_preprocessor().fit(real_data.X_train)

    train = preprocessor.transform(real_data.X_train)[SCALED_FEATURES]
    test = preprocessor.transform(real_data.X_test)[SCALED_FEATURES]

    assert train.mean().abs().max() < 1e-9
    assert train.std(ddof=0).to_numpy() == pytest.approx(1.0)
    # the test set is scaled with the training statistics, so it is not exactly 0 and 1
    assert test.mean().round(4).to_dict() == pytest.approx(
        {
            "Air temperature": -0.0013,
            "Process temperature": -0.0017,
            "Rotational speed": -0.016,
            "Torque": -0.0082,
            "Tool wear": 0.0209,
            "temp_diff": 0.0001,
            "power": -0.0134,
            "strain": 0.0212,
        },
        abs=1e-4,
    )
