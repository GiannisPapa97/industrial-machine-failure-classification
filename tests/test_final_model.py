import runpy
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from machine_failure import data, final_model, models, tuned_models
from machine_failure.final_model import (
    FINAL_MODEL,
    FINAL_THRESHOLD,
    FinalModel,
    load_final_model,
    save_final_model,
    train_final_model,
)
from machine_failure.models import with_preprocessing
from machine_failure.search_spaces import TUNED_MODELS


def make_xy(n: int = 400, failure_rate: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic features with the real columns; failures have a higher torque."""
    rng = np.random.default_rng(0)
    n_failures = int(n * failure_rate)
    y = pd.Series([1] * n_failures + [0] * (n - n_failures), name="Machine failure")
    X = pd.DataFrame(
        {
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Air temperature": rng.normal(300, 2, n),
            "Process temperature": rng.normal(310, 1.5, n),
            "Rotational speed": rng.integers(1200, 2800, n),
            "Torque": rng.normal(40, 5, n) + y.to_numpy() * 20,
            "Tool wear": rng.integers(0, 250, n),
        },
        index=np.arange(n) + 500,
    )
    return X, y.set_axis(X.index)


def cheap_models() -> dict[str, Pipeline]:
    forest = RandomForestClassifier(n_estimators=10, random_state=0)
    return {"random_forest": with_preprocessing(forest)}


@pytest.fixture(scope="module")
def trained() -> FinalModel:
    X, y = make_xy()
    return train_final_model(X, y, "random_forest", 0.5)


def test_the_chosen_model_is_one_of_the_tuned_models() -> None:
    assert FINAL_MODEL in TUNED_MODELS
    assert 0 < FINAL_THRESHOLD < 1


def test_train_final_model_uses_the_tuned_settings_and_is_fitted() -> None:
    X, y = make_xy()

    final = train_final_model(X, y)

    assert final.name == FINAL_MODEL
    assert final.threshold == FINAL_THRESHOLD
    forest = final.model.named_steps["model"]
    assert forest.n_estimators == 500  # the tuned setting, not the default 300
    assert forest.max_depth == 8
    assert hasattr(forest, "classes_")  # fitted


def test_an_unknown_model_or_a_bad_threshold_raises() -> None:
    X, y = make_xy()

    with pytest.raises(ValueError, match="unknown model"):
        train_final_model(X, y, "dummy")
    for threshold in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            train_final_model(X, y, threshold=threshold)


def test_probabilities_are_aligned_with_the_input(trained: FinalModel) -> None:
    X, y = make_xy()

    probability = trained.probability(X)

    assert probability.index.equals(X.index)
    assert probability.between(0, 1).all()
    assert probability[y == 1].mean() > probability[y == 0].mean()


def test_predict_flags_rows_at_or_above_the_threshold(trained: FinalModel) -> None:
    X, _ = make_xy()
    probability = trained.probability(X)
    threshold = float(probability.iloc[0])  # exactly one row's probability

    final = FinalModel(trained.name, trained.model, threshold)
    flagged = final.predict(X)

    assert flagged.iloc[0] == 1  # at the threshold counts as flagged
    assert (flagged == (probability >= threshold)).all()
    assert set(flagged.unique()) <= {0, 1}


def test_the_threshold_controls_how_many_rows_are_flagged(trained: FinalModel) -> None:
    X, _ = make_xy()

    low = FinalModel(trained.name, trained.model, 0.01).predict(X).sum()
    high = FinalModel(trained.name, trained.model, 0.99).predict(X).sum()

    assert low > high


def test_a_saved_model_loads_with_the_same_predictions(
    trained: FinalModel, tmp_path: Path
) -> None:
    X, _ = make_xy()
    path = tmp_path / "models" / "final.joblib"

    save_final_model(trained, path)
    loaded = load_final_model(path)

    assert loaded.name == trained.name
    assert loaded.threshold == trained.threshold
    pd.testing.assert_series_equal(loaded.probability(X), trained.probability(X))
    pd.testing.assert_series_equal(loaded.predict(X), trained.predict(X))


def test_saving_never_overwrites_unless_asked(
    trained: FinalModel, tmp_path: Path
) -> None:
    path = tmp_path / "final.joblib"
    path.write_text("keep me")

    with pytest.raises(FileExistsError, match="already exists"):
        save_final_model(trained, path)
    assert path.read_text() == "keep me"

    save_final_model(trained, path, overwrite=True)
    assert load_final_model(path).name == trained.name


def test_loading_a_missing_file_points_to_the_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="machine_failure.final_model"):
        load_final_model(tmp_path / "nope.joblib")


def test_loading_something_else_raises(tmp_path: Path) -> None:
    path = tmp_path / "other.joblib"
    joblib.dump({"model": 1}, path)

    with pytest.raises(ValueError, match="not a saved final model"):
        load_final_model(path)


@pytest.fixture
def small_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A small synthetic dataset and a quick forest instead of the tuned one."""
    monkeypatch.setattr(final_model, "load_xy", lambda: make_xy())
    monkeypatch.setattr(final_model, "build_tuned_models", cheap_models)
    # Running the module as a script re-imports it, so patch the sources too.
    monkeypatch.setattr(data, "load_xy", lambda: make_xy())
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

    runpy.run_module("machine_failure.final_model", run_name="__main__")

    assert "Saved to" in capsys.readouterr().out
    assert path.is_file()
