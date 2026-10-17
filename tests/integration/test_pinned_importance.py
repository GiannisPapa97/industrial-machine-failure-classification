"""What the final model relies on, as it is today (training set only)."""

from types import SimpleNamespace

import pytest

from machine_failure.evaluation import permutation_importances
from machine_failure.models import build_tuned_models

pytestmark = pytest.mark.slow

IMPORTANCE = {
    "Torque": 0.803,
    "Rotational speed": 0.636,
    "Tool wear": 0.429,
    "Air temperature": 0.394,
    "Process temperature": 0.242,
    "Type": 0.088,
}


def test_the_permutation_importance_of_the_random_forest(
    real_data: SimpleNamespace,
) -> None:
    table = permutation_importances(
        build_tuned_models()["random_forest"], real_data.X_train, real_data.y_train
    )

    assert list(table.index) == list(IMPORTANCE)  # the order of importance
    drops = {str(name): float(drop) for name, drop in table["drop"].items()}
    assert drops == pytest.approx(IMPORTANCE, abs=5e-3)
