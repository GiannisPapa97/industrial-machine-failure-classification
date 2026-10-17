from collections.abc import Callable

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from machine_failure.baseline import build_baselines

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


def test_build_baselines_returns_dummy_and_logistic_regression() -> None:
    models = build_baselines()

    assert list(models) == ["dummy", "logistic_regression"]
    assert isinstance(models["dummy"].steps[-1][1], DummyClassifier)
    logistic = models["logistic_regression"].steps[-1][1]
    assert isinstance(logistic, LogisticRegression)
    assert logistic.class_weight == "balanced"
