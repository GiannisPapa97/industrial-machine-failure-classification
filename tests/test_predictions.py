from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.utils.validation import check_is_fitted

from machine_failure.evaluation import cross_validate_model
from machine_failure.models import with_preprocessing
from machine_failure.predictions import out_of_fold_predictions

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


class Memorizer(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Says 1.0 for rows it was trained on and 0.0 for any other row.

    If a row were predicted by a model that had seen it, its probability would
    be 1.0, so this exposes any leakage in the out-of-fold predictions.
    """

    def fit(self, X: pd.DataFrame, y: Any) -> Memorizer:
        self.classes_ = np.array([0, 1])
        self.seen_ = set(X["row_id"])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        seen = X["row_id"].isin(self.seen_).to_numpy().astype(float)
        return np.column_stack([1 - seen, seen])


def test_every_row_gets_one_probability_from_one_fold(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400, index_offset=1000)

    result = out_of_fold_predictions(with_preprocessing(LogisticRegression()), X, y)

    assert list(result.columns) == ["label", "probability", "fold"]
    assert result.index.equals(X.index)
    assert result["label"].equals(y.rename("label"))
    assert not result["probability"].isna().any()
    assert result["probability"].between(0, 1).all()
    assert sorted(result["fold"].unique()) == [0, 1, 2, 3, 4]
    assert (result["fold"].value_counts() == 80).all()


def test_folds_are_stratified(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400, index_offset=1000)

    result = out_of_fold_predictions(with_preprocessing(LogisticRegression()), X, y)

    assert (result.groupby("fold")["label"].sum() == 8).all()  # 40 failures / 5 folds


def test_no_row_is_predicted_by_a_model_that_saw_it(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400, index_offset=1000)
    X = X.assign(row_id=np.arange(len(X)))

    result = out_of_fold_predictions(Memorizer(), X, y)

    assert (result["probability"] == 0.0).all()


def test_the_folds_match_cross_validate_model(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400, index_offset=1000)
    model = with_preprocessing(LogisticRegression())

    result = out_of_fold_predictions(model, X, y)
    scores = cross_validate_model(model, X, y)

    per_fold = [
        average_precision_score(group["label"], group["probability"])
        for _, group in result.groupby("fold")
    ]
    assert per_fold == pytest.approx(scores["pr_auc"].to_numpy())


def test_the_same_seed_gives_the_same_predictions(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400, index_offset=1000)
    model = with_preprocessing(LogisticRegression())

    first = out_of_fold_predictions(model, X, y)
    again = out_of_fold_predictions(model, X, y)
    other = out_of_fold_predictions(model, X, y, random_state=7)

    pd.testing.assert_frame_equal(first, again)
    assert not first["fold"].equals(other["fold"])


def test_n_splits_sets_the_number_of_folds(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400, index_offset=1000)

    result = out_of_fold_predictions(
        with_preprocessing(LogisticRegression()), X, y, n_splits=4
    )

    assert sorted(result["fold"].unique()) == [0, 1, 2, 3]


def test_the_given_model_is_not_fitted(make_xy: MakeXY) -> None:
    X, y = make_xy(n=400, index_offset=1000)
    model = with_preprocessing(LogisticRegression())

    out_of_fold_predictions(model, X, y)

    with pytest.raises(NotFittedError):
        check_is_fitted(model)
