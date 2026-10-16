"""Feature engineering for the AI4I 2020 dataset."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from machine_failure.data import NUMERIC_FEATURES

DERIVED_FEATURES = ["temp_diff", "power", "strain"]


class FeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Add the derived features the failure modes depend on.

    - ``temp_diff``: process temperature minus air temperature (K)
    - ``power``: torque times rotational speed, converted from rpm (W)
    - ``strain``: tool wear times torque (minNm)

    Each feature is computed row by row, so nothing is learned from the data and
    there is no leakage between training and test sets. The input columns are
    kept and the three new columns are appended. The input is not modified.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureEngineer:
        self._check_columns(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        self._check_columns(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                f"columns differ from those seen in fit: {list(X.columns)} "
                f"instead of {list(self.feature_names_in_)}"
            )
        return X.assign(
            temp_diff=X["Process temperature"] - X["Air temperature"],
            power=X["Torque"] * X["Rotational speed"] * 2 * np.pi / 60,
            strain=X["Tool wear"] * X["Torque"],
        )

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> np.ndarray:
        check_is_fitted(self)
        names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray(input_features, dtype=object)
        )
        return np.concatenate([names, np.asarray(DERIVED_FEATURES, dtype=object)])

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = [c for c in NUMERIC_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        existing = [c for c in DERIVED_FEATURES if c in X.columns]
        if existing:
            raise ValueError(f"columns already present: {existing}")
