"""Out-of-fold predicted probabilities on the training set."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedStratifiedKFold

from machine_failure.data import RANDOM_STATE
from machine_failure.evaluation.cross_validation import N_SPLITS


def out_of_fold_predictions(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Predict the failure probability of every row with a model that never saw it.

    The data is split into stratified folds, the same ones as
    ``cross_validate_model`` uses. For each fold, a fresh copy of ``model`` is fit
    on the other folds and predicts the held-out fold. Every row is therefore
    predicted exactly once, so the probabilities behave like predictions on new
    data and can be used to choose a decision threshold. Pass the training set
    only.

    Returns a DataFrame with the index of ``X`` and the columns ``label`` (the
    true target), ``probability`` (of failure) and ``fold`` (the fold in which the
    row was held out). ``model`` itself is not fitted.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=1, random_state=random_state
    )
    probability = np.full(len(X), np.nan)
    fold = np.full(len(X), -1)
    for number, (train, test) in enumerate(cv.split(X, y)):
        fitted = clone(model).fit(X.iloc[train], y.iloc[train])
        probability[test] = fitted.predict_proba(X.iloc[test])[:, 1]
        fold[test] = number
    return pd.DataFrame(
        {"label": y.to_numpy(), "probability": probability, "fold": fold},
        index=X.index,
    )
