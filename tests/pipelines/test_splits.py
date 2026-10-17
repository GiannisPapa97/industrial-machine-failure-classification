from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from machine_failure.config import DataConfig
from machine_failure.data import split_train_test
from machine_failure.pipelines import splits
from machine_failure.pipelines.splits import load_split, split_fingerprint

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture
def seen_paths(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    paths: list[Path] = []

    def load(path: Path) -> tuple[pd.DataFrame, pd.Series]:
        paths.append(path)
        return make_xy(n=400)

    monkeypatch.setattr(splits, "load_xy", load)
    return paths


def test_the_split_reads_the_configured_path_and_holds_out_20_percent(
    seen_paths: list[Path],
) -> None:
    X_train, X_test, y_train, y_test = load_split(DataConfig(path=Path("my.csv")))

    assert seen_paths == [Path("my.csv")]
    assert (len(X_train), len(X_test)) == (320, 80)
    assert (len(y_train), len(y_test)) == (320, 80)
    assert not set(X_train.index) & set(X_test.index)


@pytest.mark.usefixtures("seen_paths")
def test_the_split_is_stratified_and_reproducible() -> None:
    first = load_split(DataConfig())
    second = load_split(DataConfig())

    assert first[0].index.equals(second[0].index)
    assert int(first[2].sum()) == 32 and int(first[3].sum()) == 8  # 10% failures


def test_the_split_is_the_project_split(
    seen_paths: list[Path], make_xy: MakeXY
) -> None:
    X_train, X_test, _, _ = load_split(DataConfig())

    expected_train, expected_test, _, _ = split_train_test(*make_xy(n=400))
    assert X_train.index.equals(expected_train.index)
    assert X_test.index.equals(expected_test.index)


def test_the_fingerprint_identifies_the_data_and_the_held_out_rows(
    make_xy: MakeXY,
) -> None:
    parts = split_train_test(*make_xy(n=400))

    first = split_fingerprint(*parts)

    assert first == split_fingerprint(*parts)
    assert set(first) == {"data_sha256", "test_rows_sha256"}


def test_other_held_out_rows_change_the_fingerprint(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400)
    default = split_fingerprint(*split_train_test(X, y))

    other = split_fingerprint(*split_train_test(X, y, random_state=7))

    assert other["data_sha256"] == default["data_sha256"]  # same data
    assert other["test_rows_sha256"] != default["test_rows_sha256"]


def test_changed_data_changes_the_fingerprint(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400)
    default = split_fingerprint(*split_train_test(X, y))

    changed = X.copy()
    changed.loc[changed.index[0], "Torque"] = float(changed["Torque"].iloc[0]) + 1.0
    other = split_fingerprint(*split_train_test(changed, y))

    assert other["data_sha256"] != default["data_sha256"]
    assert other["test_rows_sha256"] == default["test_rows_sha256"]
