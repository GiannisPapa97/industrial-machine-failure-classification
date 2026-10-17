from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from machine_failure.config import DataConfig
from machine_failure.pipelines import splits
from machine_failure.pipelines.splits import load_split

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture
def seen_paths(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    paths: list[Path] = []

    def load(path: Path) -> tuple[pd.DataFrame, pd.Series]:
        paths.append(path)
        return make_xy(n=400)

    monkeypatch.setattr(splits, "load_xy", load)
    return paths


def test_the_split_uses_the_configured_path_and_test_size(
    seen_paths: list[Path],
) -> None:
    X_train, X_test, y_train, y_test = load_split(
        DataConfig(path=Path("my.csv"), test_size=0.25)
    )

    assert seen_paths == [Path("my.csv")]
    assert (len(X_train), len(X_test)) == (300, 100)
    assert (len(y_train), len(y_test)) == (300, 100)
    assert not set(X_train.index) & set(X_test.index)


@pytest.mark.usefixtures("seen_paths")
def test_the_split_is_stratified_and_reproducible() -> None:
    first = load_split(DataConfig())
    second = load_split(DataConfig())

    assert first[0].index.equals(second[0].index)
    assert int(first[2].sum()) == 32 and int(first[3].sum()) == 8  # 10% failures


@pytest.mark.usefixtures("seen_paths")
def test_another_seed_gives_another_split() -> None:
    default = load_split(DataConfig())[0].index
    other = load_split(DataConfig(random_state=7))[0].index

    assert not default.equals(other)
