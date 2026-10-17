"""Loading the dataset and splitting it as a ``DataConfig`` says."""

import logging

import pandas as pd

from machine_failure.config import DataConfig
from machine_failure.data import load_xy, split_train_test

logger = logging.getLogger(__name__)


def load_split(
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the dataset and split it into training and held-out test rows.


    Returns ``X_train, X_test, y_train, y_test``. Pipelines that only train
    or cross-validate use the training rows and leave the test rows alone.
    """
    X, y = load_xy(config.path)
    logger.debug(
        "loaded %d rows from %s; test size %s, seed %d",
        len(X),
        config.path,
        config.test_size,
        config.random_state,
    )
    return split_train_test(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
