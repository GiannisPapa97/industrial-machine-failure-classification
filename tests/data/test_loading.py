from pathlib import Path

import pandas as pd
import pytest

from machine_failure import data


def make_raw_frame() -> pd.DataFrame:
    """A tiny frame with the same columns and dtypes as the real dataset."""
    return pd.DataFrame(
        {
            "UID": [1, 2, 3],
            "Product ID": ["M14860", "L47181", "H29424"],
            "Type": ["M", "L", "H"],
            "Air temperature": [298.1, 298.2, 298.1],
            "Process temperature": [308.6, 308.7, 308.5],
            "Rotational speed": [1551, 1408, 1498],
            "Torque": [42.8, 46.3, 49.4],
            "Tool wear": [0, 3, 5],
            "Machine failure": [0, 0, 1],
            "TWF": [0, 0, 0],
            "HDF": [0, 0, 0],
            "PWF": [0, 0, 1],
            "OSF": [0, 0, 0],
            "RNF": [0, 0, 0],
        }
    )


def write_csv(df: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / data.RAW_FILENAME
    df.to_csv(path, index=False)
    return path


def test_load_raw_returns_all_columns(tmp_path: Path) -> None:
    expected = make_raw_frame()
    path = write_csv(expected, tmp_path)

    df = data.load_raw(path)

    assert list(df.columns) == data.EXPECTED_COLUMNS
    assert len(df) == len(expected)
    assert df["Machine failure"].tolist() == [0, 0, 1]


def test_load_raw_missing_file_points_to_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="machine_failure.data"):
        data.load_raw(tmp_path / "nope.csv")


def test_load_raw_missing_column(tmp_path: Path) -> None:
    df = make_raw_frame().drop(columns=["Torque"])

    with pytest.raises(data.DataValidationError, match="missing columns.*Torque"):
        data.load_raw(write_csv(df, tmp_path))


def test_load_raw_null_values(tmp_path: Path) -> None:
    df = make_raw_frame()
    df.loc[1, "Torque"] = None

    with pytest.raises(data.DataValidationError, match="null values.*Torque"):
        data.load_raw(write_csv(df, tmp_path))


def test_load_raw_unexpected_dtype(tmp_path: Path) -> None:
    df = make_raw_frame()
    df["Torque"] = df["Torque"].astype(str)
    df.loc[0, "Torque"] = "high"

    with pytest.raises(data.DataValidationError, match="dtypes.*Torque"):
        data.load_raw(write_csv(df, tmp_path))


def test_load_raw_unexpected_type_value(tmp_path: Path) -> None:
    df = make_raw_frame()
    df.loc[0, "Type"] = "X"

    with pytest.raises(data.DataValidationError, match="unexpected values.*X"):
        data.load_raw(write_csv(df, tmp_path))


@pytest.mark.parametrize(
    "column", ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
)
def test_load_raw_non_binary_column(tmp_path: Path, column: str) -> None:
    df = make_raw_frame()
    df.loc[0, column] = 2

    with pytest.raises(data.DataValidationError, match=f"only 0 or 1.*{column}"):
        data.load_raw(write_csv(df, tmp_path))


def test_load_xy_splits_features_and_target(tmp_path: Path) -> None:
    path = write_csv(make_raw_frame(), tmp_path)

    X, y = data.load_xy(path)

    assert list(X.columns) == data.FEATURE_COLUMNS
    assert y.name == data.TARGET_COLUMN
    assert y.tolist() == [0, 0, 1]
    assert len(X) == len(y) == 3


def test_load_xy_excludes_ids_flags_and_target(tmp_path: Path) -> None:
    path = write_csv(make_raw_frame(), tmp_path)

    X, _ = data.load_xy(path)

    excluded = [
        *data.ID_COLUMNS,
        *data.FAILURE_MODE_COLUMNS,
        data.TARGET_COLUMN,
    ]
    assert not set(excluded) & set(X.columns)


def test_load_xy_validates_the_data(tmp_path: Path) -> None:
    df = make_raw_frame().drop(columns=["Torque"])

    with pytest.raises(data.DataValidationError):
        data.load_xy(write_csv(df, tmp_path))
