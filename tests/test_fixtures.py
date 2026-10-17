"""The shared fixtures in conftest.py."""

from collections.abc import Callable

import pandas as pd
import pytest

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]
FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def test_make_xy_returns_the_real_columns_and_a_failure_target(
    make_xy: MakeXY,
) -> None:
    X, y = make_xy()

    assert list(X.columns) == FEATURES
    assert X.shape == (600, 6)
    assert y.name == "Machine failure"
    assert len(y) == 600
    assert int(y.sum()) == 60  # 10% failures, and they come first
    assert y.iloc[:60].eq(1).all() and y.iloc[60:].eq(0).all()


def test_make_xy_respects_the_size_and_failure_rate(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400, failure_rate=0.05)

    assert len(X) == len(y) == 400
    assert int(y.sum()) == 20


def test_the_signal_shifts_the_torque_of_failures_only(make_xy: MakeXY) -> None:
    none, _ = make_xy(signal=0.0)
    strong, y = make_xy(signal=20.0)

    shift = (strong["Torque"] - none["Torque"]).round(9)

    assert shift[y == 1].eq(20.0).all()
    assert shift[y == 0].eq(0.0).all()


def test_the_default_index_is_the_row_number(make_xy: MakeXY) -> None:
    X, y = make_xy(n=50)

    assert X.index.equals(pd.RangeIndex(50))
    assert y.index.equals(X.index)


def test_an_index_offset_moves_both_indexes(make_xy: MakeXY) -> None:
    X, y = make_xy(n=50, index_offset=1000)

    assert X.index.tolist() == list(range(1000, 1050))
    assert y.index.equals(X.index)


def test_the_data_is_reproducible_and_independent(make_xy: MakeXY) -> None:
    first_X, first_y = make_xy(n=100)
    second_X, second_y = make_xy(n=100)

    pd.testing.assert_frame_equal(first_X, second_X)
    pd.testing.assert_series_equal(first_y, second_y)
    first_X.loc[0, "Torque"] = -1.0  # changing one copy must not change the next
    assert make_xy(n=100)[0].loc[0, "Torque"] != -1.0


@pytest.mark.parametrize("n", [10, 100])
def test_every_row_has_a_valid_type(make_xy: MakeXY, n: int) -> None:
    X, _ = make_xy(n=n)

    assert set(X["Type"]) <= {"L", "M", "H"}
