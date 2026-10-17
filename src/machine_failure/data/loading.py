"""Read the raw dataset, validate it and split it into features and target."""

from pathlib import Path

import pandas as pd

from machine_failure.data.download import DEFAULT_RAW_DIR, RAW_FILENAME
from machine_failure.data.schema import FEATURE_COLUMNS, TARGET_COLUMN, validate_raw


def load_raw(path: Path = DEFAULT_RAW_DIR / RAW_FILENAME) -> pd.DataFrame:
    """Read the raw CSV and validate it. Returns all columns unchanged."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python -m machine_failure.data` to download it"
        )
    df = pd.read_csv(path)
    validate_raw(df)
    return df


def load_xy(
    path: Path = DEFAULT_RAW_DIR / RAW_FILENAME,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset as features ``X`` and binary target ``y``.

    ``X`` holds ``Type`` and the five numeric measurements. The identifier
    columns and the failure-mode flags (TWF, HDF, PWF, OSF, RNF) are dropped:
    the flags describe the failure itself, so using them would leak the target.
    """
    df = load_raw(path)
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()
