import pandas as pd
import pytest

from machine_failure import data


def make_xy(
    n_failures: int = 50, n_normal: int = 950
) -> tuple[pd.DataFrame, pd.Series]:
    n = n_failures + n_normal
    X = pd.DataFrame({"a": range(n), "b": [float(i) for i in range(n)]})
    y = pd.Series([1] * n_failures + [0] * n_normal, name="Machine failure")
    return X, y


def test_split_sizes_and_stratification() -> None:
    X, y = make_xy()

    X_train, X_test, y_train, y_test = data.split_train_test(X, y)

    assert (len(X_train), len(X_test)) == (800, 200)
    assert (len(y_train), len(y_test)) == (800, 200)
    assert y_train.mean() == pytest.approx(0.05)
    assert y_test.mean() == pytest.approx(0.05)
    assert (int(y_train.sum()), int(y_test.sum())) == (40, 10)


def test_split_keeps_rows_aligned_and_disjoint() -> None:
    X, y = make_xy()

    X_train, X_test, y_train, y_test = data.split_train_test(X, y)

    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert sorted([*X_train.index, *X_test.index]) == list(X.index)


def test_split_is_reproducible_and_depends_on_the_seed() -> None:
    X, y = make_xy()

    first = data.split_train_test(X, y)
    again = data.split_train_test(X, y)
    other = data.split_train_test(X, y, random_state=data.RANDOM_STATE + 1)

    assert first[1].index.equals(again[1].index)
    assert not first[1].index.equals(other[1].index)


def test_split_respects_test_size() -> None:
    X, y = make_xy()

    _, X_test, _, _ = data.split_train_test(X, y, test_size=0.5)

    assert len(X_test) == 500
