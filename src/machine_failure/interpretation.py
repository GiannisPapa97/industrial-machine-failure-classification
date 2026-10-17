"""What the final model relies on, and which failures it misses.

Everything here uses the **training set only**, on out-of-fold predictions and
held-out folds, so it never looks at the test set. Studying the test set's
mistakes would invite adjusting the model to it, which would make the final
evaluation no longer honest.
"""

import argparse

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold

from machine_failure.data import (
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    load_raw,
    split_train_test,
)
from machine_failure.evaluation import N_SPLITS
from machine_failure.final_model import FINAL_MODEL, FINAL_THRESHOLD
from machine_failure.predictions import out_of_fold_predictions
from machine_failure.tuned_models import build_tuned_models


def permutation_importances(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """How much PR-AUC drops when each input feature is shuffled.

    For each cross-validation fold, a fresh copy of ``model`` is fit on the other
    folds, and each feature of the held-out fold is shuffled ``n_repeats`` times,
    one feature at a time. A feature the model relies on makes the PR-AUC drop
    when shuffled; an irrelevant one does not. Shuffling happens before the
    pipeline, so a derived feature is disturbed whenever one of its inputs is.

    Returns one row per feature with the mean drop and its standard deviation
    across the folds, sorted from most to least important.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=1, random_state=random_state
    )
    drops = []
    for train, test in cv.split(X, y):
        fitted = clone(model).fit(X.iloc[train], y.iloc[train])
        result = permutation_importance(
            fitted,
            X.iloc[test],
            y.iloc[test],
            scoring="average_precision",
            n_repeats=n_repeats,
            random_state=random_state,
        )
        drops.append(result.importances_mean)
    per_fold = pd.DataFrame(drops, columns=list(X.columns))
    table = pd.DataFrame({"drop": per_fold.mean(), "std": per_fold.std()})
    return table.sort_values("drop", ascending=False)


def missed_by_failure_mode(
    predictions: pd.DataFrame, threshold: float, modes: pd.DataFrame
) -> pd.DataFrame:
    """For each kind of failure, how many were caught and how many were missed.

    ``predictions`` is the output of ``out_of_fold_predictions``; ``modes`` holds
    the failure-mode flags (TWF, HDF, PWF, OSF, RNF) for the same rows. A failure
    is caught when its probability is at least ``threshold``. A failure with
    several flags is grouped under all of them, e.g. ``HDF+PWF``, and one with no
    flag is ``none``.
    """
    failures = predictions[predictions["label"] == 1]
    flags = modes.loc[failures.index, list(FAILURE_MODE_COLUMNS)]
    kind = flags.apply(
        lambda row: "+".join(row.index[row == 1]) or "none", axis=1
    ).rename("failure mode")
    caught = (failures["probability"] >= threshold).rename("caught")
    table = pd.crosstab(kind, caught).reindex(columns=[True, False], fill_value=0)
    table.columns = ["caught", "missed"]
    table["failures"] = table["caught"] + table["missed"]
    table["missed share"] = table["missed"] / table["failures"]
    table = table.sort_values("failures", ascending=False, kind="stable")
    return table[["failures", "caught", "missed", "missed share"]]


def group_medians(
    X: pd.DataFrame, predictions: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Median of each measurement for normal rows, caught failures and missed ones."""
    flagged = predictions["probability"] >= threshold
    failure = predictions["label"] == 1
    group = pd.Series("normal", index=X.index)
    group[failure & flagged] = "caught failure"
    group[failure & ~flagged] = "missed failure"
    medians = X[list(NUMERIC_FEATURES)].groupby(group).median()
    counts = group.value_counts().rename("rows")
    table = medians.join(counts).reindex(["normal", "caught failure", "missed failure"])
    table["rows"] = table["rows"].fillna(0).astype(int)  # an empty group has 0 rows
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=FINAL_MODEL,
        help=f"tuned model to study (default: {FINAL_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=FINAL_THRESHOLD,
        help=f"decision threshold (default: {FINAL_THRESHOLD})",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="shuffles per feature and fold in the importance (default: 5)",
    )
    args = parser.parse_args()

    raw = load_raw()
    X, y = raw[FEATURE_COLUMNS].copy(), raw[TARGET_COLUMN].copy()
    X_train, _, y_train, _ = split_train_test(X, y)  # the test set is not used here
    model = build_tuned_models()[args.model]

    importance = permutation_importances(
        model, X_train, y_train, n_repeats=args.n_repeats
    )
    predictions = out_of_fold_predictions(model, X_train, y_train)
    flagged = predictions["probability"] >= args.threshold
    missed = int(((predictions["label"] == 1) & ~flagged).sum())

    print(
        f"{args.model} on the training set ({len(X_train):,} rows, "
        f"{int(y_train.sum())} failures), threshold {args.threshold}."
    )
    print("\nDrop in PR-AUC when one input feature is shuffled (held-out folds)")
    print(importance.to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"\nFailures caught and missed at the threshold ({missed} missed in total)")
    modes = missed_by_failure_mode(predictions, args.threshold, raw)
    print(modes.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nMedian measurements of normal rows, caught failures and missed failures")
    print(group_medians(X_train, predictions, args.threshold).round(2).to_string())


if __name__ == "__main__":
    main()
