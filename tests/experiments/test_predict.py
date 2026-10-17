import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from machine_failure.config import TrainingConfig
from machine_failure.experiments import app
from machine_failure.pipelines import TrainingPipeline

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture
def files(make_xy: MakeXY, tmp_path: Path) -> tuple[Path, Path]:
    """A saved model (threshold 0.3) and a CSV of 40 rows to score."""
    X, y = make_xy(n=300)
    model = tmp_path / "final.joblib"
    config = TrainingConfig(model="knn", tuned=False, threshold=0.3, output_path=model)
    pipeline = TrainingPipeline(config)
    pipeline.save(pipeline.fit(X, y))
    rows = tmp_path / "rows.csv"
    X.iloc[:40].to_csv(rows, index=False)
    return model, rows


def test_main_prints_one_row_of_scores_per_input_row(
    files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, rows = files
    monkeypatch.setattr(
        sys, "argv", ["mf", "predict", str(rows), "--model-path", str(model)]
    )

    app.main()

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0] == "row,probability,flagged"
    assert len(lines) == 41
    assert "40 rows scored" in captured.err and "threshold 0.3 (knn)" in captured.err


def test_main_writes_the_scores_to_a_file_and_refuses_to_overwrite(
    files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model, rows = files
    out = tmp_path / "scores.csv"
    argv = [
        "mf",
        "predict",
        str(rows),
        "--model-path",
        str(model),
        "--output",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    app.main()
    scores = pd.read_csv(out, index_col="row")
    assert list(scores.columns) == ["probability", "flagged"] and len(scores) == 40

    with pytest.raises(SystemExit):
        app.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--overwrite"])
    app.main()


def test_main_rejects_rows_without_the_feature_columns(
    files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    model, _ = files
    bad = tmp_path / "bad.csv"
    bad.write_text("Type\nL\n")
    monkeypatch.setattr(
        sys, "argv", ["mf", "predict", str(bad), "--model-path", str(model)]
    )

    with pytest.raises(SystemExit) as exit_info:
        app.main()

    assert exit_info.value.code == 1
    assert capsys.readouterr().err.startswith("error: missing feature columns")
