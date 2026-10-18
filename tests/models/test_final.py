import logging
from collections.abc import Callable
from pathlib import Path

import joblib
import pandas as pd
import pytest
import sklearn

from machine_failure.cli import UserError
from machine_failure.models import (
    FINAL_MODEL,
    FINAL_THRESHOLD,
    TUNED_MODELS,
    FinalModel,
    ModelFileError,
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
    with pytest.raises(FileNotFoundError, match="mf train"):
        load_final_model(tmp_path / "nope.joblib")


def test_loading_something_else_raises(tmp_path: Path) -> None:
    path = tmp_path / "other.joblib"
    joblib.dump({"model": 1}, path)

    with pytest.raises(ModelFileError, match="not a saved final model"):
        load_final_model(path)


def test_the_metadata_is_saved_and_loaded_with_the_model(
    trained: FinalModel, tmp_path: Path
) -> None:
    path = tmp_path / "final.joblib"
    with_metadata = FinalModel(
        trained.name, trained.model, trained.threshold, {"trained_at": "now"}
    )

    save_final_model(with_metadata, path)

    assert joblib.load(path)["format"] == 2
    assert load_final_model(path).metadata == {"trained_at": "now"}


def test_a_model_saved_before_metadata_still_loads_with_a_warning(
    trained: FinalModel, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "old.joblib"
    joblib.dump({"name": trained.name, "model": trained.model, "threshold": 0.4}, path)

    with caplog.at_level(logging.WARNING, logger="machine_failure"):
        loaded = load_final_model(path)

    assert (loaded.threshold, loaded.metadata) == (0.4, {})
    assert "saved without metadata" in caplog.text


def test_an_unknown_format_is_refused(trained: FinalModel, tmp_path: Path) -> None:
    path = tmp_path / "future.joblib"
    save_final_model(trained, path)
    saved = joblib.load(path)
    joblib.dump({**saved, "format": 3}, path)

    with pytest.raises(ModelFileError, match="format 3"):
        load_final_model(path)


def test_a_file_that_is_not_a_pickle_is_a_model_file_error(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("Type,Torque\nL,40\n")

    with pytest.raises(ModelFileError, match="cannot be read") as error_info:
        load_final_model(path)
    assert isinstance(error_info.value, UserError)  # reported in one line


def test_another_scikit_learn_version_gives_a_warning(
    trained: FinalModel, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "final.joblib"
    metadata = {"versions": {"scikit-learn": "0.0.1"}}
    save_final_model(
        FinalModel(trained.name, trained.model, trained.threshold, metadata), path
    )

    with caplog.at_level(logging.WARNING, logger="machine_failure"):
        load_final_model(path)

    assert "saved with scikit-learn 0.0.1" in caplog.text


def test_the_same_scikit_learn_version_is_silent(
    trained: FinalModel, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "final.joblib"
    metadata = {"versions": {"scikit-learn": sklearn.__version__}}
    save_final_model(
        FinalModel(trained.name, trained.model, trained.threshold, metadata), path
    )

    with caplog.at_level(logging.WARNING, logger="machine_failure"):
        load_final_model(path)

    assert caplog.text == ""
