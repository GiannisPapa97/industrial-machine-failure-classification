"""The real dataset and its split, as they are today."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from machine_failure.data import FAILURE_MODE_COLUMNS, load_raw


def test_the_raw_data_has_the_recorded_counts(dataset_path: Path) -> None:
    raw = load_raw(dataset_path)

    assert raw.shape == (10_000, 14)
    assert raw["Machine failure"].sum() == 339
    assert raw["Type"].value_counts().to_dict() == {"L": 6000, "M": 2997, "H": 1003}
    assert raw.groupby("Type")["Machine failure"].sum().to_dict() == {
        "H": 21,
        "L": 235,
        "M": 83,
    }


def test_the_failure_mode_flags_have_the_recorded_counts(dataset_path: Path) -> None:
    raw = load_raw(dataset_path)

    flags = {column: int(raw[column].sum()) for column in FAILURE_MODE_COLUMNS}

    assert flags == {"TWF": 46, "HDF": 115, "PWF": 95, "OSF": 98, "RNF": 19}


def test_the_flags_do_not_match_the_target_exactly(dataset_path: Path) -> None:
    raw = load_raw(dataset_path)
    any_flag = (raw[FAILURE_MODE_COLUMNS].sum(axis=1) > 0).astype(int)

    table = pd.crosstab(raw["Machine failure"], any_flag)

    # 9 failures have no flag; 18 rows have a flag but no failure (random failures)
    assert table.to_dict() == {0: {0: 9643, 1: 9}, 1: {0: 18, 1: 330}}


def test_the_split_has_the_recorded_sizes(real_data: SimpleNamespace) -> None:
    assert real_data.X_train.shape == (8000, 6)
    assert real_data.X_test.shape == (2000, 6)
    assert int(real_data.y_train.sum()) == 271
    assert int(real_data.y_test.sum()) == 68


def test_the_split_assigns_the_same_rows_as_before(real_data: SimpleNamespace) -> None:
    train, test = real_data.X_train.index, real_data.X_test.index

    assert train[:5].tolist() == [4058, 1221, 6895, 9863, 8711]
    assert test[:5].tolist() == [2997, 4871, 3858, 951, 6463]
    assert int(np.sum(train.to_numpy())) == 40_015_578
    assert int(np.sum(test.to_numpy())) == 9_979_422
    assert set(train).isdisjoint(test)
