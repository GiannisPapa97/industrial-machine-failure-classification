from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from machine_failure import data


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the network call with a tiny DataFrame and record the calls."""
    calls: list[int] = []
    frame = pd.DataFrame({"Type": ["L", "M"], "Machine failure": [0, 1]})

    def fetch(id: int) -> SimpleNamespace:  # noqa: A002
        calls.append(id)
        return SimpleNamespace(data=SimpleNamespace(original=frame))

    monkeypatch.setattr(data, "fetch_ucirepo", fetch)
    return calls


def test_download_writes_csv(tmp_path: Path, fake_fetch: list[int]) -> None:
    dest = tmp_path / "raw"

    path = data.download_dataset(dest)

    assert path == dest / data.RAW_FILENAME
    assert pd.read_csv(path).shape == (2, 2)
    assert fake_fetch == [data.UCI_DATASET_ID]


def test_download_does_not_overwrite_existing_file(
    tmp_path: Path, fake_fetch: list[int]
) -> None:
    path = tmp_path / data.RAW_FILENAME
    path.write_text("keep,me\n")

    result = data.download_dataset(tmp_path)

    assert result == path
    assert path.read_text() == "keep,me\n"
    assert fake_fetch == []


def test_download_overwrite_replaces_file(
    tmp_path: Path, fake_fetch: list[int]
) -> None:
    path = tmp_path / data.RAW_FILENAME
    path.write_text("old\n")

    data.download_dataset(tmp_path, overwrite=True)

    assert pd.read_csv(path).shape == (2, 2)
    assert fake_fetch == [data.UCI_DATASET_ID]


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
