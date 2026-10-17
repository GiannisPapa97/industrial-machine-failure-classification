import numpy as np
import pandas as pd
import pytest

from machine_failure.data import DataValidationError, validate_features


def rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H"],
            "Air temperature": [298.1, 299.0, 300.5],
            "Process temperature": [308.6, 309.2, 310.0],
            "Rotational speed": [1551, 1408, 1500],
            "Torque": [42.8, 46.3, 40.1],
            "Tool wear": [0, 3, 5],
        }
    )


def test_valid_rows_pass() -> None:
    validate_features(rows())


def test_extra_columns_and_a_missing_target_are_fine() -> None:
    validate_features(rows().assign(UID=[1, 2, 3], note="x"))


def test_missing_columns_are_named() -> None:
    with pytest.raises(DataValidationError, match="Torque"):
        validate_features(rows().drop(columns="Torque"))


def test_null_values_are_named() -> None:
    frame = rows()
    frame.loc[1, "Tool wear"] = None

    with pytest.raises(DataValidationError, match="null values.*Tool wear"):
        validate_features(frame)


def test_non_numeric_measurements_are_named() -> None:
    with pytest.raises(DataValidationError, match="numeric.*Torque"):
        validate_features(rows().assign(Torque=["a", "b", "c"]))
    with pytest.raises(DataValidationError, match="numeric.*Tool wear"):
        validate_features(rows().assign(**{"Tool wear": [True, False, True]}))


def test_an_unknown_machine_type_is_named() -> None:
    with pytest.raises(DataValidationError, match="'X'"):
        validate_features(rows().assign(Type=["L", "X", "H"]))


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_measurements_are_named(value: float) -> None:
    with pytest.raises(DataValidationError, match="infinite values.*Torque"):
        validate_features(rows().assign(Torque=[40.0, value, 41.0]))


def test_unexpected_types_of_mixed_kinds_are_listed() -> None:
    with pytest.raises(DataValidationError, match=r"\['5', 'X'\]"):
        validate_features(rows().assign(Type=["L", "X", 5]))
