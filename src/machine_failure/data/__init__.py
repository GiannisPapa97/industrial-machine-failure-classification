"""Loading, validating and splitting the AI4I 2020 dataset."""

from machine_failure.data.download import (
    DEFAULT_RAW_DIR,
    RAW_FILENAME,
    UCI_DATASET_ID,
    download_dataset,
)
from machine_failure.data.loading import load_raw, load_xy
from machine_failure.data.schema import (
    BINARY_COLUMNS,
    EXPECTED_COLUMNS,
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    FLOAT_COLUMNS,
    ID_COLUMNS,
    INTEGER_COLUMNS,
    NUMERIC_FEATURES,
    STRING_COLUMNS,
    TARGET_COLUMN,
    TYPE_COLUMN,
    TYPE_VALUES,
    DataValidationError,
    validate_raw,
)
from machine_failure.data.splitting import RANDOM_STATE, TEST_SIZE, split_train_test

__all__ = [
    "BINARY_COLUMNS",
    "DEFAULT_RAW_DIR",
    "EXPECTED_COLUMNS",
    "FAILURE_MODE_COLUMNS",
    "FEATURE_COLUMNS",
    "FLOAT_COLUMNS",
    "ID_COLUMNS",
    "INTEGER_COLUMNS",
    "NUMERIC_FEATURES",
    "RANDOM_STATE",
    "RAW_FILENAME",
    "STRING_COLUMNS",
    "TARGET_COLUMN",
    "TEST_SIZE",
    "TYPE_COLUMN",
    "TYPE_VALUES",
    "UCI_DATASET_ID",
    "DataValidationError",
    "download_dataset",
    "load_raw",
    "load_xy",
    "split_train_test",
    "validate_raw",
]
