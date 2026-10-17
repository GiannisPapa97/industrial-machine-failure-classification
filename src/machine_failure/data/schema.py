"""The columns of the AI4I 2020 dataset and the checks that its values are valid."""

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from machine_failure.cli import UserError

ID_COLUMNS = ["UID", "Product ID"]
TYPE_COLUMN = "Type"
NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
FEATURE_COLUMNS = [TYPE_COLUMN, *NUMERIC_FEATURES]
TARGET_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

TYPE_VALUES = ("L", "M", "H")  # low, medium and high quality variants
INTEGER_COLUMNS = ["UID", "Rotational speed", "Tool wear"]
FLOAT_COLUMNS = ["Air temperature", "Process temperature", "Torque"]
STRING_COLUMNS = ["Product ID", TYPE_COLUMN]
BINARY_COLUMNS = [TARGET_COLUMN, *FAILURE_MODE_COLUMNS]
EXPECTED_COLUMNS = [
    *ID_COLUMNS,
    TYPE_COLUMN,
    *NUMERIC_FEATURES,
    TARGET_COLUMN,
    *FAILURE_MODE_COLUMNS,
]


class DataValidationError(UserError, ValueError):
    """The raw dataset does not have the expected structure or values."""


def validate_raw(df: pd.DataFrame) -> None:
    """Raise DataValidationError if ``df`` is not the expected raw dataset."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    with_nulls = [c for c in EXPECTED_COLUMNS if df[c].isna().any()]
    if with_nulls:
        raise DataValidationError(f"columns contain null values: {with_nulls}")

    wrong_dtype = (
        [c for c in INTEGER_COLUMNS if not pdt.is_integer_dtype(df[c])]
        + [c for c in FLOAT_COLUMNS if not pdt.is_float_dtype(df[c])]
        + [c for c in STRING_COLUMNS if not pdt.is_string_dtype(df[c])]
        + [c for c in BINARY_COLUMNS if not pdt.is_integer_dtype(df[c])]
    )
    if wrong_dtype:
        raise DataValidationError(f"columns have unexpected dtypes: {wrong_dtype}")

    check_type_values(df)

    non_binary = [c for c in BINARY_COLUMNS if not df[c].isin([0, 1]).all()]
    if non_binary:
        raise DataValidationError(f"columns must contain only 0 or 1: {non_binary}")


def validate_features(df: pd.DataFrame) -> None:
    """Raise DataValidationError if ``df`` cannot be scored by a model.

    Only the feature columns are checked: ``Type`` and the five measurements. Other
    columns may be present (they are ignored), and no target is needed, so new rows
    and the raw dataset both pass.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing feature columns: {missing}")

    with_nulls = [c for c in FEATURE_COLUMNS if df[c].isna().any()]
    if with_nulls:
        raise DataValidationError(f"feature columns contain null values: {with_nulls}")

    not_numeric = [
        c
        for c in NUMERIC_FEATURES
        if not pdt.is_numeric_dtype(df[c]) or pdt.is_bool_dtype(df[c])
    ]
    if not_numeric:
        raise DataValidationError(f"feature columns must be numeric: {not_numeric}")

    not_finite = [c for c in NUMERIC_FEATURES if not np.isfinite(df[c]).all()]
    if not_finite:
        raise DataValidationError(
            f"feature columns contain infinite values: {not_finite}"
        )

    check_type_values(df)


def check_type_values(df: pd.DataFrame) -> None:
    """Raise DataValidationError unless every ``Type`` is one of ``TYPE_VALUES``.

    The unexpected values are listed as text, so values of mixed kinds (a number
    among the letters) are reported instead of failing to sort.
    """
    bad_types = set(df[TYPE_COLUMN].unique()) - set(TYPE_VALUES)
    if bad_types:
        raise DataValidationError(
            f"{TYPE_COLUMN!r} has unexpected values: {sorted(map(str, bad_types))}"
        )
