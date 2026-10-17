"""Out-of-fold predictions, thresholds and misses of the tuned models, as they are.

Everything here uses the training set only; the test set is never touched. The
out-of-fold predictions of the four tuned models take about a minute to compute, so
they are computed once per module and the tests are marked slow.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from machine_failure.data import FEATURE_COLUMNS, load_raw
from machine_failure.final_evaluation import metrics_at_threshold
from machine_failure.final_model import FINAL_THRESHOLD
from machine_failure.interpretation import group_medians, missed_by_failure_mode
from machine_failure.predictions import out_of_fold_predictions
from machine_failure.thresholds import best_f1_per_model, compare_at_recall_targets
from machine_failure.tuned_models import build_tuned_models

pytestmark = pytest.mark.slow

# threshold, correct alerts, false alarms, missed failures, correct non-alerts
AT_RECALL_TARGETS = {
    (0.80, "random_forest"): (0.657, 217, 1, 54, 7728),
    (0.80, "extra_trees"): (0.328, 217, 39, 54, 7690),
    (0.80, "hist_gradient_boosting"): (0.923, 217, 6, 54, 7723),
    (0.80, "xgboost"): (0.495, 217, 32, 54, 7697),
    (0.85, "random_forest"): (0.147, 231, 55, 40, 7674),
    (0.85, "extra_trees"): (0.148, 231, 173, 40, 7556),
    (0.85, "hist_gradient_boosting"): (0.100, 231, 112, 40, 7617),
    (0.85, "xgboost"): (0.199, 231, 80, 40, 7649),
    (0.90, "random_forest"): (0.060, 244, 216, 27, 7513),
    (0.90, "extra_trees"): (0.068, 244, 433, 27, 7296),
    (0.90, "hist_gradient_boosting"): (0.008, 244, 331, 27, 7398),
    (0.90, "xgboost"): (0.027, 244, 320, 27, 7409),
}
BEST_F1 = {
    "random_forest": (0.475, 224, 3, 47, 7726),
    "extra_trees": (0.416, 207, 12, 64, 7717),
    "hist_gradient_boosting": (0.864, 221, 7, 50, 7722),
    "xgboost": (0.654, 214, 22, 57, 7707),
}
# failures, caught, missed, for the random forest at the final threshold
MISSES = {
    "HDF": (78, 74, 4),
    "PWF": (68, 68, 0),
    "OSF": (63, 61, 2),
    "TWF": (33, 0, 33),
    "PWF+OSF": (11, 11, 0),
    "none": (7, 0, 7),
    "HDF+OSF": (6, 6, 0),
    "HDF+PWF": (2, 2, 0),
    "TWF+OSF": (1, 0, 1),
    "TWF+PWF+OSF": (1, 1, 0),
    "TWF+RNF": (1, 0, 1),
}
MEDIANS = {
    "normal": (300.0, 310.0, 1507.0, 39.8, 106.0, 7729),
    "caught failure": (301.7, 310.4, 1359.0, 55.7, 144.0, 223),
    "missed failure": (300.65, 310.7, 1484.0, 39.75, 209.5, 48),
}
COUNTS = ["threshold", "tp", "fp", "fn", "tn"]


@pytest.fixture(scope="module")
def oof(real_data: SimpleNamespace) -> dict[str, pd.DataFrame]:
    return {
        name: out_of_fold_predictions(model, real_data.X_train, real_data.y_train)
        for name, model in build_tuned_models().items()
    }


def test_the_out_of_fold_folds(oof: dict[str, pd.DataFrame]) -> None:
    forest = oof["random_forest"]

    assert forest["fold"].value_counts().sort_index().tolist() == [1600] * 5
    assert forest.groupby("fold")["label"].sum().tolist() == [54, 54, 54, 54, 55]
    assert forest["probability"].sum() == pytest.approx(267.45, abs=0.01)


@pytest.mark.parametrize(("target", "name"), list(AT_RECALL_TARGETS))
def test_the_counts_at_each_recall_target(
    oof: dict[str, pd.DataFrame], target: float, name: str
) -> None:
    row = compare_at_recall_targets(oof).to_dict(orient="index")[(target, name)]

    expected = AT_RECALL_TARGETS[(target, name)]
    assert row["threshold"] == pytest.approx(expected[0], abs=1e-3)
    assert [int(row[count]) for count in COUNTS[1:]] == list(expected[1:])


@pytest.mark.parametrize("name", list(BEST_F1))
def test_the_best_f1_point(oof: dict[str, pd.DataFrame], name: str) -> None:
    row = best_f1_per_model(oof).to_dict(orient="index")[name]

    expected = BEST_F1[name]
    assert row["threshold"] == pytest.approx(expected[0], abs=1e-3)
    assert [int(row[count]) for count in COUNTS[1:]] == list(expected[1:])


def test_the_metrics_at_the_final_threshold(oof: dict[str, pd.DataFrame]) -> None:
    forest = oof["random_forest"]

    scores = metrics_at_threshold(
        forest["label"].to_numpy(), forest["probability"].to_numpy(), FINAL_THRESHOLD
    )

    assert (scores["tp"], scores["fp"], scores["fn"], scores["tn"]) == (
        223,
        3,
        48,
        7726,
    )
    assert scores["pr_auc"] == pytest.approx(0.9109, abs=2e-4)
    assert scores["roc_auc"] == pytest.approx(0.976, abs=2e-4)
    assert scores["recall"] == pytest.approx(0.8229, abs=2e-4)
    assert scores["precision"] == pytest.approx(0.9867, abs=2e-4)
    assert scores["f1"] == pytest.approx(0.8974, abs=2e-4)


def test_the_misses_by_failure_mode(
    oof: dict[str, pd.DataFrame], dataset_path: Path
) -> None:
    table = missed_by_failure_mode(
        oof["random_forest"], FINAL_THRESHOLD, load_raw(dataset_path)
    )

    found = {
        str(kind): (int(row["failures"]), int(row["caught"]), int(row["missed"]))
        for kind, row in table.to_dict(orient="index").items()
    }
    assert found == MISSES
    assert list(table.index)[:4] == ["HDF", "PWF", "OSF", "TWF"]  # most failures first


def test_the_medians_of_normal_caught_and_missed_rows(
    oof: dict[str, pd.DataFrame], real_data: SimpleNamespace
) -> None:
    table = group_medians(real_data.X_train, oof["random_forest"], FINAL_THRESHOLD)

    assert list(table.index) == list(MEDIANS)
    rows = table.to_dict(orient="index")
    for group, expected in MEDIANS.items():
        measurements = [rows[group][column] for column in FEATURE_COLUMNS[1:]]
        assert measurements == pytest.approx(list(expected[:5]), abs=0.01)
        assert rows[group]["rows"] == expected[5]
