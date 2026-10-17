"""Fixtures and marking for the integration tests.

Integration tests run the real workflows on the real dataset and compare the
results with the numbers recorded in the README. They protect the behavior of the
project during refactoring. They are skipped when the dataset is not downloaded:
run ``uv run python -m machine_failure.data`` first.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from machine_failure.data import RAW_FILENAME, load_xy, split_train_test

HERE = Path(__file__).resolve().parent
DATASET = HERE.parents[1] / "data" / "raw" / RAW_FILENAME


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test in this folder as an integration test."""
    for item in items:
        if HERE in Path(str(item.path)).resolve().parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def dataset_path() -> Path:
    """The downloaded dataset file; the test is skipped if it is missing."""
    if not DATASET.is_file():
        pytest.skip(
            f"{DATASET} not found; run `uv run python -m machine_failure.data` first"
        )
    return DATASET


@pytest.fixture(scope="session")
def real_data(dataset_path: Path) -> SimpleNamespace:
    """The real dataset and its fixed split, loaded once per test session.

    Has the attributes ``X``, ``y``, ``X_train``, ``X_test``, ``y_train`` and
    ``y_test``.
    """
    X, y = load_xy(dataset_path)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    return SimpleNamespace(
        X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
