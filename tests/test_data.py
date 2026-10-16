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
