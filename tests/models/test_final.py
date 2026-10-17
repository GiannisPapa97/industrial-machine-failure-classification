from collections.abc import Callable
from pathlib import Path

import joblib
import pandas as pd
import pytest

from machine_failure.models import (
    FINAL_MODEL,
    FINAL_THRESHOLD,
    TUNED_MODELS,
    FinalModel,
    get_model,
    load_final_model,
    save_final_model,
)

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture(scope="module")
def trained(make_xy: MakeXY) -> FinalModel:
    X, y = make_xy(n=400, index_offset=500)
    model = get_model("random_forest").build_tuned().fit(X, y)
    return FinalModel("random_forest", model, 0.5)


def test_the_chosen_model_is_one_of_the_tuned_models() -> None:
    assert FINAL_MODEL in TUNED_MODELS
    assert 0 < FINAL_THRESHOLD < 1


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
    with pytest.raises(FileNotFoundError, match="mf-train"):
        load_final_model(tmp_path / "nope.joblib")


def test_loading_something_else_raises(tmp_path: Path) -> None:
    path = tmp_path / "other.joblib"
    joblib.dump({"model": 1}, path)

    with pytest.raises(ValueError, match="not a saved final model"):
        load_final_model(path)
