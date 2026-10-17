from collections.abc import Callable

import pandas as pd
import pytest

from machine_failure.evaluation import (
    METRICS,
    compare_models,
    describe_cross_validation,
    format_comparison,
)
from machine_failure.models import build_baselines

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


def test_compare_models_scores_each_model_on_every_metric(make_xy: MakeXY) -> None:
    X, y = make_xy()

    table = compare_models(build_baselines(), X, y)

    assert list(table.index) == ["dummy", "logistic_regression"]
    assert list(table.to_dict(orient="list")) == [
        (metric, stat) for metric in METRICS for stat in ("mean", "std")
    ]
    pr_auc = table[("pr_auc", "mean")].to_dict()
    roc_auc = table[("roc_auc", "mean")].to_dict()
    assert pr_auc["dummy"] == pytest.approx(0.1)
    assert roc_auc["dummy"] == pytest.approx(0.5)
    assert pr_auc["logistic_regression"] > 0.8


def test_format_comparison_shows_mean_and_std(make_xy: MakeXY) -> None:
    X, y = make_xy()
    table = compare_models(build_baselines(), X, y)

    text = format_comparison(table)

    assert "±" in text
    for name in ("dummy", "logistic_regression"):
        assert name in text
    for metric in METRICS:
        assert metric in text


def test_describe_cross_validation_names_the_folds_and_the_rows() -> None:
    assert describe_cross_validation(1_234, 42, 3, n_splits=4) == (
        "4-fold stratified cross-validation on the training set "
        "(1,234 rows, 42 failures), 3 repeat(s). Mean ± std across folds."
    )
