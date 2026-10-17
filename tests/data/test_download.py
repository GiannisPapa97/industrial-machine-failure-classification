import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import ucimlrepo

from machine_failure.data import download
from machine_failure.experiments import app


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the network call with a tiny DataFrame and record the calls."""
    calls: list[int] = []
    frame = pd.DataFrame({"Type": ["L", "M"], "Machine failure": [0, 1]})

    def fetch(id: int) -> SimpleNamespace:  # noqa: A002
        calls.append(id)
        return SimpleNamespace(data=SimpleNamespace(original=frame))

    monkeypatch.setattr(download, "fetch_ucirepo", fetch)
    # Running the module as a script re-imports it, so patch the source too.
    monkeypatch.setattr(ucimlrepo, "fetch_ucirepo", fetch)
    return calls


def test_download_writes_csv(tmp_path: Path, fake_fetch: list[int]) -> None:
    dest = tmp_path / "raw"

    path = download.download_dataset(dest)

    assert path == dest / download.RAW_FILENAME
    assert pd.read_csv(path).shape == (2, 2)
    assert fake_fetch == [download.UCI_DATASET_ID]


def test_download_does_not_overwrite_existing_file(
    tmp_path: Path, fake_fetch: list[int]
) -> None:
    path = tmp_path / download.RAW_FILENAME
    path.write_text("keep,me\n")

    result = download.download_dataset(tmp_path)

    assert result == path
    assert path.read_text() == "keep,me\n"
    assert fake_fetch == []


def test_download_overwrite_replaces_file(
    tmp_path: Path, fake_fetch: list[int]
) -> None:
    path = tmp_path / download.RAW_FILENAME
    path.write_text("old\n")

    download.download_dataset(tmp_path, overwrite=True)

    assert pd.read_csv(path).shape == (2, 2)
    assert fake_fetch == [download.UCI_DATASET_ID]


def test_main_downloads_to_dest_and_prints_path(
    tmp_path: Path,
    fake_fetch: list[int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["mf", "download", "--dest", str(tmp_path)])

    app.main()

    expected = tmp_path / download.RAW_FILENAME
    assert capsys.readouterr().out.strip() == str(expected)
    assert expected.is_file()
    assert fake_fetch == [download.UCI_DATASET_ID]


def test_main_overwrite_flag_replaces_existing_file(
    tmp_path: Path,
    fake_fetch: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / download.RAW_FILENAME
    path.write_text("old\n")
    monkeypatch.setattr(
        sys, "argv", ["mf", "download", "--dest", str(tmp_path), "--overwrite"]
    )

    app.main()

    assert pd.read_csv(path).shape == (2, 2)
    assert fake_fetch == [download.UCI_DATASET_ID]
