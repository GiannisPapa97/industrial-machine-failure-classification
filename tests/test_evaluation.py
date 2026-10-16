import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from machine_failure.evaluation import SCORING, cross_validate_model, summarize_scores


def make_xy(n: int = 500, failure_rate: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
    """A synthetic task: feature ``signal`` separates failures, ``noise`` does not."""
    rng = np.random.default_rng(0)
    n_failures = int(n * failure_rate)
    y = pd.Series([1] * n_failures + [0] * (n - n_failures), name="Machine failure")
    X = pd.DataFrame(
        {
            "signal": y.to_numpy() * 3.0 + rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    return X, y


def test_returns_one_row_per_fold_and_one_column_per_metric() -> None:
    X, y = make_xy()

    scores = cross_validate_model(DummyClassifier(), X, y)

    assert scores.shape == (5, len(SCORING))
    assert list(scores.columns) == list(SCORING)


def test_repeats_and_splits_set_the_number_of_folds() -> None:
    X, y = make_xy()

    scores = cross_validate_model(DummyClassifier(), X, y, n_splits=4, n_repeats=3)

    assert len(scores) == 12


def test_dummy_classifier_sets_the_no_skill_floor() -> None:
    X, y = make_xy()

    scores = cross_validate_model(DummyClassifier(), X, y)

    assert scores["accuracy"].to_numpy() == pytest.approx(0.9)
    assert scores["pr_auc"].to_numpy() == pytest.approx(0.1)  # the failure rate
    assert scores["roc_auc"].to_numpy() == pytest.approx(0.5)
    assert (scores[["recall", "precision", "f1"]] == 0).all().all()


def test_informative_model_beats_the_floor() -> None:
    X, y = make_xy()

    scores = cross_validate_model(LogisticRegression(), X, y)

    assert (scores["pr_auc"] > 0.7).all()
    assert (scores["roc_auc"] > 0.9).all()


def test_same_seed_gives_same_scores() -> None:
    X, y = make_xy()

    first = cross_validate_model(LogisticRegression(), X, y)
    again = cross_validate_model(LogisticRegression(), X, y)
    other = cross_validate_model(LogisticRegression(), X, y, random_state=1)

    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)


def test_summarize_scores_reports_mean_and_std_per_metric() -> None:
    scores = pd.DataFrame({"pr_auc": [0.2, 0.4], "recall": [1.0, 1.0]})

    summary = summarize_scores(scores)

    assert list(summary.index) == ["pr_auc", "recall"]
    assert list(summary.columns) == ["mean", "std"]
    assert summary.loc["pr_auc", "mean"] == pytest.approx(0.3)
    assert summary.loc["pr_auc", "std"] == pytest.approx(np.std([0.2, 0.4], ddof=1))
    assert summary.loc["recall", "std"] == 0
