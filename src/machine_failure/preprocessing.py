"""Preprocessing pipeline: derived features, encoding and scaling."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from machine_failure.data import NUMERIC_FEATURES, TYPE_COLUMN
from machine_failure.features import DERIVED_FEATURES, FeatureEngineer

TYPE_CATEGORIES = ["L", "M", "H"]
SCALED_FEATURES = [*NUMERIC_FEATURES, *DERIVED_FEATURES]


def build_preprocessor() -> Pipeline:
    """Build the preprocessing pipeline for the features from ``load_xy``.

    1. Add ``temp_diff``, ``power`` and ``strain`` (see ``FeatureEngineer``).
    2. One-hot encode ``Type`` into ``Type_L``, ``Type_M`` and ``Type_H``. The
       categories are fixed, so every fold produces the same columns.
    3. Standardize the five measurements and the three derived features to mean 0
       and standard deviation 1.

    The scaler learns its statistics in ``fit``, so use it inside a ``Pipeline``
    with the model: cross-validation then refits it on each training fold only.
    The output is a pandas DataFrame with 11 named columns.
    """
    encode_and_scale = ColumnTransformer(
        [
            (
                "type",
                OneHotEncoder(categories=[TYPE_CATEGORIES], sparse_output=False),
                [TYPE_COLUMN],
            ),
            ("numeric", StandardScaler(), SCALED_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        [("features", FeatureEngineer()), ("encode_and_scale", encode_and_scale)]
    )
    pipeline.set_output(transform="pandas")
    return pipeline
