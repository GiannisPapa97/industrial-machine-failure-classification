"""Loading the dataset and splitting it into the fixed training and test rows."""

import hashlib
import logging

import numpy as np
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


def split_fingerprint(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, str]:
    """Hashes that identify the dataset and which of its rows were held out.

    Saved with a trained model, they let the evaluation check that it scores the
    model on the rows that were held out when it was trained, from the same data.
    The data hash covers the values and the row labels, not the file's bytes.
    """
    rows = pd.concat([X_train, X_test]).join(pd.concat([y_train, y_test]))
    data = pd.util.hash_pandas_object(rows.sort_index(), index=True).to_numpy()
    test_rows = np.sort(X_test.index.to_numpy()).astype(np.int64)
    return {
        "data_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "test_rows_sha256": hashlib.sha256(test_rows.tobytes()).hexdigest(),
    }
