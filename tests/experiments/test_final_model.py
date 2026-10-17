import runpy
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from machine_failure import data, models, tuned_models
from machine_failure.experiments import final_model
from machine_failure.final_model import (
    FINAL_THRESHOLD,
    load_final_model,
)
from machine_failure.models import with_preprocessing

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


def cheap_models() -> dict[str, Pipeline]:
    forest = RandomForestClassifier(n_estimators=10, random_state=0)
    return {"random_forest": with_preprocessing(forest)}


@pytest.fixture
def small_run(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """A small synthetic dataset and a quick forest instead of the tuned one."""
    monkeypatch.setattr(
        final_model, "load_xy", lambda: make_xy(n=400, index_offset=500)
    )
    monkeypatch.setattr(final_model, "build_tuned_models", cheap_models)
    # Running the module as a script re-imports it, so patch the sources too.
    monkeypatch.setattr(data, "load_xy", lambda: make_xy(n=400, index_offset=500))
    monkeypatch.setattr(tuned_models, "build_tuned_models", cheap_models)
    monkeypatch.setattr(models, "build_candidates", cheap_models)


def test_main_trains_on_the_training_set_and_saves(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "models" / "final.joblib"
    monkeypatch.setattr(sys, "argv", ["final_model", "--output", str(path)])

    final_model.main()

    out = capsys.readouterr().out
    assert "320 rows, 32 failures" in out  # 80% of 400 rows: the test set is not used
    assert f"threshold {FINAL_THRESHOLD}" in out
    assert load_final_model(path).threshold == FINAL_THRESHOLD


def test_main_accepts_a_threshold_and_refuses_to_overwrite(
    small_run: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "final.joblib"
    argv = ["final_model", "--output", str(path), "--threshold", "0.3"]
    monkeypatch.setattr(sys, "argv", argv)

    final_model.main()
    assert load_final_model(path).threshold == 0.3

    with pytest.raises(FileExistsError):
        final_model.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--overwrite"])
    final_model.main()


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_runs_main_when_executed_as_script(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "final.joblib"
    monkeypatch.setattr(sys, "argv", ["final_model", "--output", str(path)])

    runpy.run_module("machine_failure.experiments.final_model", run_name="__main__")

    assert "Saved to" in capsys.readouterr().out
    assert path.is_file()
