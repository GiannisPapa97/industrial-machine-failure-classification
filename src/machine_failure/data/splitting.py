"""Split the data into a training set and a held-out test set."""

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into a training set and a held-out test set.

    The split is stratified on ``y``, so both parts keep the failure rate, and
    it is reproducible for a given ``random_state``. The test set should be used
    only once, for the final evaluation; use cross-validation on the training
    set to compare and tune models.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
