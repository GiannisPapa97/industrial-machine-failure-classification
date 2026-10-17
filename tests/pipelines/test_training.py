from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from machine_failure.config import DataConfig, TrainingConfig
from machine_failure.features import FEATURE_SETS
from machine_failure.models import (
    FINAL_THRESHOLD,
    FinalModel,
    load_final_model,
)
from machine_failure.pipelines import TrainingPipeline, TrainingResult, splits


def test_describe_says_what_was_trained() -> None:
    final = FinalModel("random_forest", object(), 0.475)
    result = TrainingResult(final, TrainingConfig(), n_rows=8000, n_failures=271)

    assert result.describe() == (
        "Trained random_forest on the training set (8,000 rows, 271 failures); "
        "threshold 0.475."
    )


MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


@pytest.fixture
def synthetic_data(make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch) -> None:
    """400 synthetic rows (40 failures) instead of the downloaded dataset."""
    monkeypatch.setattr(splits, "load_xy", lambda path: make_xy(n=400))


@pytest.mark.usefixtures("synthetic_data")
def test_run_trains_on_the_training_rows_only() -> None:
    result = TrainingPipeline().run()

    assert (result.n_rows, result.n_failures) == (320, 32)  # 80% of the rows
    assert result.final.name == "random_forest"
    assert result.final.threshold == FINAL_THRESHOLD
    assert result.config == TrainingConfig()


def test_the_data_config_decides_the_split(
    make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def load(path: Path) -> tuple[pd.DataFrame, pd.Series]:
        seen.append(path)
        return make_xy(n=400)

    monkeypatch.setattr(splits, "load_xy", load)
    config = TrainingConfig(data=DataConfig(path=Path("my.csv")))

    result = TrainingPipeline(config).run()

    assert seen == [Path("my.csv")]
    assert result.n_rows == 320  # the fixed split: 80% for training


@pytest.mark.usefixtures("synthetic_data")
def test_the_same_config_gives_the_same_split_and_predictions() -> None:
    pipeline = TrainingPipeline()
    X_first, y_first = pipeline.load_training_data()
    X_second, _ = pipeline.load_training_data()

    assert X_first.index.equals(X_second.index)
    first = pipeline.fit(X_first, y_first).final.probability(X_first)
    second = pipeline.fit(X_first, y_first).final.probability(X_first)
    pd.testing.assert_series_equal(first, second)


def test_the_tuned_model_carries_the_tuned_settings() -> None:
    pipeline = TrainingPipeline(TrainingConfig(model="random_forest"))

    assert pipeline.build_model().get_params()["model__max_depth"] == 8


def test_an_untuned_model_keeps_its_default_settings() -> None:
    config = TrainingConfig(model="random_forest", tuned=False)

    assert (
        TrainingPipeline(config).build_model().get_params()["model__max_depth"] is None
    )


def test_each_build_gives_a_new_unfitted_model() -> None:
    pipeline = TrainingPipeline()

    assert pipeline.build_model() is not pipeline.build_model()


def test_the_feature_set_decides_the_inputs_the_model_sees(make_xy: MakeXY) -> None:
    X, y = make_xy(n=300)
    widths = {}
    for feature_set in FEATURE_SETS:
        config = TrainingConfig(model="knn", tuned=False, feature_set=feature_set)
        final = TrainingPipeline(config).fit(X, y).final
        widths[feature_set] = final.model.named_steps["model"].n_features_in_

    assert widths["raw"] < widths["all"]


def test_save_writes_the_model_and_refuses_to_overwrite(
    make_xy: MakeXY, tmp_path: Path
) -> None:
    config = TrainingConfig(model="knn", tuned=False, output_path=tmp_path / "m.joblib")
    pipeline = TrainingPipeline(config)
    result = pipeline.fit(*make_xy(n=300))

    path = pipeline.save(result)

    assert path == tmp_path / "m.joblib"
    assert load_final_model(path).threshold == config.threshold
    with pytest.raises(FileExistsError):
        pipeline.save(result)
    assert pipeline.save(result, overwrite=True) == path
