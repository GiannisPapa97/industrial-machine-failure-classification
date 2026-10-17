"""The final model trained on the training set, as it is today.

Only training rows are used here: the test set is evaluated once, by the final
evaluation command, and never by a test.
"""

from types import SimpleNamespace

import pytest
from sklearn.ensemble import RandomForestClassifier

from machine_failure.models import (
    FINAL_MODEL,
    FINAL_THRESHOLD,
    FinalModel,
    train_final_model,
)


@pytest.fixture(scope="module")
def final(real_data: SimpleNamespace) -> FinalModel:
    return train_final_model(real_data.X_train, real_data.y_train)


def test_the_final_model_is_the_tuned_random_forest(final: FinalModel) -> None:
    forest = final.model.named_steps["model"]

    assert final.name == FINAL_MODEL == "random_forest"
    assert final.threshold == FINAL_THRESHOLD == 0.475
    assert isinstance(forest, RandomForestClassifier)
    assert (forest.n_estimators, forest.max_depth) == (500, 8)
    assert (forest.min_samples_leaf, forest.max_features) == (2, 0.8)
    assert forest.class_weight is None


def test_the_final_model_on_the_training_rows(
    final: FinalModel, real_data: SimpleNamespace
) -> None:
    probability = final.probability(real_data.X_train)
    flagged = final.predict(real_data.X_train)

    assert int(flagged.sum()) == 232
    assert probability.sum() == pytest.approx(269.55, abs=0.01)
    assert probability.iloc[:5].tolist() == pytest.approx(
        [0.0011, 0.001, 0.0008, 0.0012, 0.0005], abs=1e-4
    )
