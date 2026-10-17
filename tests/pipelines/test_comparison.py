from collections.abc import Callable

import pandas as pd
import pytest

from machine_failure.config import ComparisonConfig, DataConfig
from machine_failure.evaluation import METRICS, compare_models
from machine_failure.models import build_all_models, get_model
from machine_failure.pipelines import (
    ComparisonPipeline,
    ComparisonResult,
    splits,
)

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]
QUICK = ("dummy", "logistic_regression")


def test_the_default_pipeline_builds_every_model_in_the_old_order() -> None:
    models = ComparisonPipeline().build_models()

    assert list(models) == list(build_all_models())
    assert len(models) == 10


def test_models_come_in_the_order_of_the_config() -> None:
    config = ComparisonConfig(models=("xgboost", "dummy"))

    assert list(ComparisonPipeline(config).build_models()) == ["xgboost", "dummy"]


def test_the_feature_set_reaches_the_models() -> None:
    raw = ComparisonPipeline(ComparisonConfig(models=("knn",), feature_set="raw"))
    full = ComparisonPipeline(ComparisonConfig(models=("knn",)))

    assert repr(raw.build_models()["knn"]) != repr(full.build_models()["knn"])


def test_the_models_are_new_each_time() -> None:
    pipeline = ComparisonPipeline(ComparisonConfig(models=("knn",)))

    assert pipeline.build_models()["knn"] is not pipeline.build_models()["knn"]


def test_compare_gives_one_row_per_model_with_every_metric(make_xy: MakeXY) -> None:
    X, y = make_xy(n=300)

    result = ComparisonPipeline(ComparisonConfig(models=QUICK)).compare(X, y)

    assert isinstance(result, ComparisonResult)
    assert list(result.table.index) == list(QUICK)
    assert list(result.table.to_dict(orient="list")) == [
        (metric, stat) for metric in METRICS for stat in ("mean", "std")
    ]
    assert (result.n_rows, result.n_failures) == (300, 30)


def test_compare_gives_the_scores_compare_models_gives(make_xy: MakeXY) -> None:
    X, y = make_xy(n=300)
    config = ComparisonConfig(models=QUICK, n_splits=3, random_state=5)

    result = ComparisonPipeline(config).compare(X, y)

    expected = compare_models(
        {name: get_model(name).build() for name in QUICK},
        X,
        y,
        n_splits=3,
        random_state=5,
    )
    pd.testing.assert_frame_equal(result.table, expected)


def test_describe_uses_the_configured_folds_and_repeats(make_xy: MakeXY) -> None:
    X, y = make_xy(n=300)
    config = ComparisonConfig(models=("dummy",), n_splits=3, n_repeats=2)

    result = ComparisonPipeline(config).compare(X, y)

    assert result.describe() == (
        "3-fold stratified cross-validation on the training set "
        "(300 rows, 30 failures), 2 repeat(s). Mean ± std across folds."
    )


def test_run_compares_on_the_training_rows_only(
    make_xy: MakeXY, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(splits, "load_xy", lambda path: make_xy(n=400))
    config = ComparisonConfig(models=("dummy",), data=DataConfig(test_size=0.25))

    result = ComparisonPipeline(config).run()

    assert (result.n_rows, result.n_failures) == (300, 30)  # 75% of the rows
    assert result.config is config
