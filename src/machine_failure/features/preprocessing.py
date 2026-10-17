"""Preprocessing pipeline: derived features, encoding and scaling."""

from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from machine_failure.data import NUMERIC_FEATURES, TYPE_COLUMN
from machine_failure.features.engineer import DERIVED_FEATURES, FeatureEngineer

FeatureSet = Literal["raw", "computed", "all"]
FEATURE_SETS: tuple[FeatureSet, ...] = ("raw", "computed", "all")

TYPE_CATEGORIES = ["L", "M", "H"]
SCALED_FEATURES = [*NUMERIC_FEATURES, *DERIVED_FEATURES]


def build_preprocessor(feature_set: FeatureSet = "all") -> Pipeline:
    """Build the preprocessing pipeline for the features from ``load_xy``.

    ``feature_set`` chooses which features the model sees:

    - ``"raw"``: ``Type`` (one-hot encoded as ``Type_L``, ``Type_M`` and
      ``Type_H``) and the five measurements, standardized. 8 columns.
    - ``"computed"``: only ``temp_diff``, ``power`` and ``strain`` (see
      ``FeatureEngineer``), standardized. 3 columns.
    - ``"all"`` (default): the raw features and the three computed ones.
      11 columns.

    The one-hot categories are fixed, so every fold produces the same columns.
    The scaler learns its statistics in ``fit``, so use the pipeline inside a
    ``Pipeline`` with the model: cross-validation then refits it on each training
    fold only. The output is a pandas DataFrame with named columns.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {FEATURE_SETS}, not {feature_set!r}"
        )

    scaled = {
        "raw": NUMERIC_FEATURES,
        "computed": DERIVED_FEATURES,
        "all": SCALED_FEATURES,
    }[feature_set]
    transformers = [("numeric", StandardScaler(), scaled)]
    if feature_set != "computed":
        one_hot = OneHotEncoder(categories=[TYPE_CATEGORIES], sparse_output=False)
        transformers.insert(0, ("type", one_hot, [TYPE_COLUMN]))
    encode_and_scale = ColumnTransformer(transformers, verbose_feature_names_out=False)

    steps: list[tuple[str, object]] = [("encode_and_scale", encode_and_scale)]
    if feature_set != "raw":
        steps.insert(0, ("features", FeatureEngineer()))
    pipeline = Pipeline(steps)
    pipeline.set_output(transform="pandas")
    return pipeline
