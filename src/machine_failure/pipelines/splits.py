"""Loading the dataset and splitting it into the fixed training and test rows."""

import logging

import pandas as pd

from machine_failure.config import DataConfig
from machine_failure.data import load_xy, split_train_test

logger = logging.getLogger(__name__)


def load_split(
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the dataset from ``config.path`` and split it into training and test rows.

    The split is the project's fixed one (see ``DataConfig``). Returns ``X_train,
    X_test, y_train, y_test``. Pipelines that only train or cross-validate use the
    training rows and leave the test rows alone.
    """
    X, y = load_xy(config.path)
    logger.debug("loaded %d rows from %s", len(X), config.path)
    return split_train_test(X, y)
