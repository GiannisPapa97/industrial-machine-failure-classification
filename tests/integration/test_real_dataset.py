from pathlib import Path
from types import SimpleNamespace

from machine_failure.data import load_raw


def test_the_real_dataset_loads_and_validates(dataset_path: Path) -> None:
    raw = load_raw(dataset_path)

    assert raw.shape == (10_000, 14)


def test_the_fixture_provides_the_split(real_data: SimpleNamespace) -> None:
    assert len(real_data.X) == 10_000
    assert len(real_data.X_train) + len(real_data.X_test) == len(real_data.X)
