from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from machine_failure.data import FAILURE_MODE_COLUMNS, NUMERIC_FEATURES
from machine_failure.interpretation import (
    group_medians,
    missed_by_failure_mode,
    permutation_importances,
)
from machine_failure.models import with_preprocessing

MakeXY = Callable[..., tuple[pd.DataFrame, pd.Series]]


def cheap_forest() -> Pipeline:
    forest = RandomForestClassifier(n_estimators=20, random_state=0)
    return with_preprocessing(forest)


def test_the_informative_feature_has_the_largest_importance(make_xy: MakeXY) -> None:
    X, y = make_xy(signal=12.0)

    importance = permutation_importances(cheap_forest(), X, y, n_repeats=3)

    drops = importance["drop"].to_dict()
    assert importance.index[0] == "Torque"
    assert drops["Torque"] > 0.1
    assert all(
        drop < drops["Torque"] / 2 for name, drop in drops.items() if name != "Torque"
    )


def test_importance_has_a_row_per_feature_sorted_with_a_spread(make_xy: MakeXY) -> None:
    X, y = make_xy(signal=12.0)

    importance = permutation_importances(cheap_forest(), X, y, n_repeats=2)

    assert set(importance.index) == set(X.columns)
    assert list(importance.columns) == ["drop", "std"]
    assert importance["drop"].is_monotonic_decreasing
    assert (importance["std"] >= 0).all()


def test_importance_is_reproducible_and_depends_on_the_seed(make_xy: MakeXY) -> None:
    X, y = make_xy(signal=12.0)

    first = permutation_importances(cheap_forest(), X, y, n_repeats=2)
    again = permutation_importances(cheap_forest(), X, y, n_repeats=2)
    other = permutation_importances(cheap_forest(), X, y, n_repeats=2, random_state=7)

    pd.testing.assert_frame_equal(first, again)
    assert not first["drop"].equals(other["drop"])


def test_importance_does_not_change_the_given_model(make_xy: MakeXY) -> None:
    X, y = make_xy(signal=12.0)
    model = cheap_forest()

    permutation_importances(model, X, y, n_repeats=2)

    assert not hasattr(model.named_steps["model"], "estimators_")  # still unfitted


def modes_frame(rows: dict[int, list[str]]) -> pd.DataFrame:
    """Failure-mode flags for the given rows, from lists of flag names."""
    flags = pd.DataFrame(0, index=range(8), columns=list(FAILURE_MODE_COLUMNS))
    for row, names in rows.items():
        flags.loc[row, names] = 1
    return flags


def test_missed_failures_are_counted_by_failure_mode() -> None:
    predictions = pd.DataFrame(
        {
            "label": [1, 1, 1, 1, 1, 0, 0, 1],
            "probability": [0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.1, 0.6],
        }
    )
    modes = modes_frame(
        {0: ["TWF"], 1: ["TWF"], 2: ["HDF"], 4: ["HDF", "PWF"], 7: ["RNF"]}
    )

    table = missed_by_failure_mode(predictions, 0.5, modes)

    assert list(table.columns) == ["failures", "caught", "missed", "missed share"]
    assert list(table.index) == ["TWF", "HDF", "HDF+PWF", "RNF", "none"]
    rows = table.to_dict(orient="index")
    assert rows["TWF"] == {"failures": 2, "caught": 1, "missed": 1, "missed share": 0.5}
    assert rows["HDF"] == {"failures": 1, "caught": 1, "missed": 0, "missed share": 0.0}
    # row 3 has no flag at all
    assert rows["none"] == {
        "failures": 1,
        "caught": 0,
        "missed": 1,
        "missed share": 1.0,
    }
    assert table["failures"].sum() == 6  # the six failures; normal rows are ignored


def test_a_failure_at_the_threshold_counts_as_caught() -> None:
    predictions = pd.DataFrame({"label": [1, 1], "probability": [0.5, 0.49]})
    modes = pd.DataFrame(
        {name: [1, 1] if name == "TWF" else [0, 0] for name in FAILURE_MODE_COLUMNS}
    )

    table = missed_by_failure_mode(predictions, 0.5, modes)

    assert table.loc["TWF", "caught"] == 1
    assert table.loc["TWF", "missed"] == 1


def medians_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.DataFrame({name: np.arange(6) * 1.0 for name in NUMERIC_FEATURES})
    X["Torque"] = [10, 20, 30, 40, 50, 60]
    predictions = pd.DataFrame(
        {
            "label": [0, 0, 1, 1, 1, 0],
            "probability": [0.1, 0.2, 0.9, 0.8, 0.1, 0.3],
        }
    )
    return X, predictions


def test_group_medians_separate_normal_caught_and_missed_rows() -> None:
    X, predictions = medians_inputs()

    table = group_medians(X, predictions, 0.5)

    assert list(table.index) == ["normal", "caught failure", "missed failure"]
    assert table["Torque"].tolist() == [
        20,
        35,
        50,
    ]  # medians of (10, 20, 60), (30, 40), (50)
    assert table["rows"].tolist() == [3, 2, 1]


def test_an_empty_group_has_no_rows_instead_of_failing() -> None:
    X, predictions = medians_inputs()

    table = group_medians(X, predictions, 0.05)  # every row flagged: nothing missed

    assert table.loc["missed failure", "rows"] == 0
    assert table.loc["missed failure"].drop("rows").isna().all()
