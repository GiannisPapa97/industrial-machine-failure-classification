from collections.abc import Callable

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from machine_failure.models import all_models, build_baselines, get_model

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


def test_build_baselines_returns_dummy_and_logistic_regression() -> None:
    models = build_baselines()

    assert list(models) == ["dummy", "logistic_regression"]
    assert isinstance(models["dummy"].steps[-1][1], DummyClassifier)
    logistic = models["logistic_regression"].steps[-1][1]
    assert isinstance(logistic, LogisticRegression)
    assert logistic.class_weight == "balanced"


def test_baselines_are_registered_in_the_model_registry() -> None:
    assert list(all_models("baseline")) == ["dummy", "logistic_regression"]
    assert get_model("dummy").role == "baseline"
    assert not get_model("logistic_regression").is_tuned
