"""Derived features, and the encoding and scaling that prepare them for a model."""

from machine_failure.features.engineer import DERIVED_FEATURES, FeatureEngineer
from machine_failure.features.preprocessing import (
    FEATURE_SETS,
    SCALED_FEATURES,
    TYPE_CATEGORIES,
    FeatureSet,
    build_preprocessor,
)

__all__ = [
    "DERIVED_FEATURES",
    "FEATURE_SETS",
    "SCALED_FEATURES",
    "TYPE_CATEGORIES",
    "FeatureEngineer",
    "FeatureSet",
    "build_preprocessor",
]
