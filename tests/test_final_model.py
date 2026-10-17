from collections.abc import Callable
from pathlib import Path

import joblib
import pandas as pd
import pytest

from machine_failure.final_model import (
    FINAL_MODEL,
    FINAL_THRESHOLD,
    FinalModel,
    load_final_model,
    save_final_model,
    train_final_model,
)
from machine_failure.search_spaces import TUNED_MODELS

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture(scope="module")
def trained(make_xy: MakeXY) -> FinalModel:
    X, y = make_xy(n=400, index_offset=500)
    return train_final_model(X, y, "random_forest", 0.5)


def test_the_chosen_model_is_one_of_the_tuned_models() -> None:
    assert FINAL_MODEL in TUNED_MODELS
    assert 0 < FINAL_THRESHOLD < 1


def test_train_final_model_uses_the_tuned_settings_and_is_fitted(
    make_xy: MakeXY,
) -> None:
    X, y = make_xy(n=400, index_offset=500)

    final = train_final_model(X, y)

    assert final.name == FINAL_MODEL
    assert final.threshold == FINAL_THRESHOLD
    forest = final.model.named_steps["model"]
    assert forest.n_estimators == 500  # the tuned setting, not the default 300
    assert forest.max_depth == 8
    assert hasattr(forest, "classes_")  # fitted


def test_an_unknown_model_or_a_bad_threshold_raises(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400, index_offset=500)

    with pytest.raises(ValueError, match="unknown model"):
        train_final_model(X, y, "dummy")
    for threshold in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            train_final_model(X, y, threshold=threshold)


def test_probabilities_are_aligned_with_the_input(
    make_xy: MakeXY, trained: FinalModel
) -> None:
    X, y = make_xy(n=400, index_offset=500)

    probability = trained.probability(X)

    assert probability.index.equals(X.index)
    assert probability.between(0, 1).all()
    assert probability[y == 1].mean() > probability[y == 0].mean()


def test_predict_flags_rows_at_or_above_the_threshold(
    make_xy: MakeXY, trained: FinalModel
) -> None:
    X, _ = make_xy(n=400, index_offset=500)
    probability = trained.probability(X)
    threshold = float(probability.iloc[0])  # exactly one row's probability

    final = FinalModel(trained.name, trained.model, threshold)
    flagged = final.predict(X)

    assert flagged.iloc[0] == 1  # at the threshold counts as flagged
    assert (flagged == (probability >= threshold)).all()
    assert set(flagged.unique()) <= {0, 1}


def test_the_threshold_controls_how_many_rows_are_flagged(
    make_xy: MakeXY, trained: FinalModel
) -> None:
    X, _ = make_xy(n=400, index_offset=500)

    low = FinalModel(trained.name, trained.model, 0.01).predict(X).sum()
    high = FinalModel(trained.name, trained.model, 0.99).predict(X).sum()

    assert low > high


def test_a_saved_model_loads_with_the_same_predictions(
    make_xy: MakeXY, trained: FinalModel, tmp_path: Path
) -> None:
    X, _ = make_xy(n=400, index_offset=500)
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
