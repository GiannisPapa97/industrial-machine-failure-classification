"""Download and load the AI4I 2020 Predictive Maintenance dataset."""

import argparse
from pathlib import Path

import pandas as pd
from pandas.api import types as pdt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

UCI_DATASET_ID = 601
RAW_FILENAME = "ai4i2020.csv"
DEFAULT_RAW_DIR = Path("data/raw")
TEST_SIZE = 0.2
RANDOM_STATE = 42

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

TYPE_VALUES = {"L", "M", "H"}
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


class DataValidationError(ValueError):
    """The raw dataset does not have the expected structure or values."""


def download_dataset(
    dest_dir: Path = DEFAULT_RAW_DIR, *, overwrite: bool = False
) -> Path:
    """Fetch the dataset from the UCI repository and save it as a CSV.

    An existing file is never replaced unless ``overwrite`` is True.
    Returns the path of the CSV file.
    """
    path = dest_dir / RAW_FILENAME
    if path.exists() and not overwrite:
        return path

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset.data.original.to_csv(path, index=False)
    return path


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

    bad_types = set(df[TYPE_COLUMN].unique()) - TYPE_VALUES
    if bad_types:
        raise DataValidationError(
            f"{TYPE_COLUMN!r} has unexpected values: {sorted(bad_types)}"
        )

    non_binary = [c for c in BINARY_COLUMNS if not df[c].isin([0, 1]).all()]
    if non_binary:
        raise DataValidationError(f"columns must contain only 0 or 1: {non_binary}")


def load_raw(path: Path = DEFAULT_RAW_DIR / RAW_FILENAME) -> pd.DataFrame:
    """Read the raw CSV and validate it. Returns all columns unchanged."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python -m machine_failure.data` to download it"
        )
    df = pd.read_csv(path)
    validate_raw(df)
    return df


def load_xy(
    path: Path = DEFAULT_RAW_DIR / RAW_FILENAME,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset as features ``X`` and binary target ``y``.

    ``X`` holds ``Type`` and the five numeric measurements. The identifier
    columns and the failure-mode flags (TWF, HDF, PWF, OSF, RNF) are dropped:
    the flags describe the failure itself, so using them would leak the target.
    """
    df = load_raw(path)
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into a training set and a held-out test set.

    The split is stratified on ``y``, so both parts keep the failure rate, and
    it is reproducible for a given ``random_state``. The test set should be used
    only once, for the final evaluation; use cross-validation on the training
    set to compare and tune models.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"directory for the CSV file (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the file if it already exists",
    )
    args = parser.parse_args()
    print(download_dataset(args.dest, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
