from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from machine_failure.config import InferenceConfig, TrainingConfig
from machine_failure.data import DataValidationError
from machine_failure.models import load_final_model
from machine_failure.pipelines import InferencePipeline, TrainingPipeline

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture(scope="module")
def saved_model(
    make_xy: MakeXY, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, pd.DataFrame]:
    """A small model saved by the training pipeline (threshold 0.3), and its rows."""
    X, y = make_xy(n=300)
    path = tmp_path_factory.mktemp("models") / "final.joblib"
    config = TrainingConfig(model="knn", tuned=False, threshold=0.3, output_path=path)
    pipeline = TrainingPipeline(config)
    pipeline.save(pipeline.fit(X, y))
    return path, X


def test_predict_gives_a_probability_and_an_alert_per_row(
    saved_model: tuple[Path, pd.DataFrame],
) -> None:
    path, X = saved_model
    pipeline = InferencePipeline(InferenceConfig(model_path=path))

    scores = pipeline.predict(X)

    assert list(scores.columns) == ["probability", "flagged"]
    assert scores.index.equals(X.index)
    assert scores["probability"].between(0, 1).all()
    # an alert is a probability at the saved threshold or above
    assert (scores["flagged"] == (scores["probability"] >= 0.3)).all()
    assert scores["flagged"].sum() > 0


def test_predictions_match_the_saved_model(
    saved_model: tuple[Path, pd.DataFrame],
) -> None:
    path, X = saved_model
    final = load_final_model(path)

    scores = InferencePipeline(InferenceConfig(model_path=path)).predict(X)

    pd.testing.assert_series_equal(scores["probability"], final.probability(X))
    pd.testing.assert_series_equal(scores["flagged"], final.predict(X))


def test_extra_columns_and_column_order_do_not_matter(
    saved_model: tuple[Path, pd.DataFrame],
) -> None:
    path, X = saved_model
    pipeline = InferencePipeline(InferenceConfig(model_path=path))
    reordered = X[list(reversed(X.columns))].assign(UID=range(len(X)), label=1)

    pd.testing.assert_frame_equal(pipeline.predict(reordered), pipeline.predict(X))


def test_a_single_row_can_be_scored(saved_model: tuple[Path, pd.DataFrame]) -> None:
    path, X = saved_model

    scores = InferencePipeline(InferenceConfig(model_path=path)).predict(X.iloc[:1])

    assert len(scores) == 1


def test_invalid_rows_fail_before_the_model_is_loaded(tmp_path: Path) -> None:
    pipeline = InferencePipeline(InferenceConfig(model_path=tmp_path / "nope.joblib"))
    X = pd.DataFrame({"Type": ["L"]})

    with pytest.raises(DataValidationError, match="missing feature columns"):
        pipeline.predict(X)


def test_a_missing_model_file_says_how_to_train_one(tmp_path: Path) -> None:
    pipeline = InferencePipeline(InferenceConfig(model_path=tmp_path / "nope.joblib"))

    with pytest.raises(FileNotFoundError, match="mf train"):
        pipeline.final  # noqa: B018


def test_the_model_is_loaded_once(saved_model: tuple[Path, pd.DataFrame]) -> None:
    pipeline = InferencePipeline(InferenceConfig(model_path=saved_model[0]))

    assert pipeline.final is pipeline.final


def test_the_default_config_points_at_the_default_model_path() -> None:
    assert InferencePipeline().config == InferenceConfig()


def test_predict_csv_reads_rows_from_a_file(
    saved_model: tuple[Path, pd.DataFrame], tmp_path: Path
) -> None:
    path, X = saved_model
    csv = tmp_path / "rows.csv"
    X.iloc[:20].to_csv(csv, index=False)
    pipeline = InferencePipeline(InferenceConfig(model_path=path))

    scores = pipeline.predict_csv(csv)

    assert len(scores) == 20
    pd.testing.assert_series_equal(
        scores["probability"].reset_index(drop=True),
        pipeline.predict(X.iloc[:20])["probability"].reset_index(drop=True),
    )


def test_predict_csv_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rows.csv"):
        InferencePipeline().predict_csv(tmp_path / "rows.csv")
