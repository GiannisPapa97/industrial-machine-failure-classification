import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from machine_failure.experiments import app, thresholds
from machine_failure.models import with_preprocessing
from machine_failure.pipelines import splits

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


def cheap_models() -> dict[str, Pipeline]:
    forest = RandomForestClassifier(n_estimators=10, random_state=0)
    return {
        "forest": with_preprocessing(forest),
        "logistic": with_preprocessing(LogisticRegression()),
    }


@pytest.fixture
def small_run(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """A small synthetic dataset and two quick models instead of the tuned four."""
    monkeypatch.setattr(splits, "load_xy", lambda path: make_xy(n=400))
    monkeypatch.setattr(thresholds, "build_tuned_models", cheap_models)


def test_main_prints_the_comparison_for_the_training_set(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["mf", "thresholds"])

    app.main()

    captured = capsys.readouterr()
    assert "320 rows, 32 failures" in captured.out  # 80% of 400 rows, 10% failures
    assert "At each recall target" in captured.out
    assert "At the threshold with the best F1" in captured.out
    for target in ("80%", "85%", "90%"):
        assert target in captured.out
    assert "forest" in captured.out and "logistic" in captured.out
    assert "forest: out-of-fold predictions" in captured.err
    assert "Plot saved" not in captured.out


def test_main_accepts_targets_and_saves_a_plot(
    small_run: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    path = tmp_path / "pr.png"
    argv = ["mf", "thresholds", "--targets", "0.7", "--plot", str(path)]
    monkeypatch.setattr(sys, "argv", argv)

    app.main()

    out = capsys.readouterr().out
    assert "70%" in out and "85%" not in out
    assert f"Plot saved to {path}" in out
    assert path.is_file()


@pytest.mark.usefixtures("small_run")
def test_main_reads_the_targets_from_a_config_file(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = tmp_path / "thresholds.toml"
    config.write_text("recall_targets = [0.6]\n")

    app.main(["thresholds", "--config", str(config)])

    assert "60%" in capsys.readouterr().out
