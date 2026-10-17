"""What the final model relies on, and which failures it misses.

Everything here uses the **training set only**, on out-of-fold predictions and
held-out folds, so it never looks at the test set. Studying the test set's
mistakes would invite adjusting the model to it, which would make the final
evaluation no longer honest.
"""

import argparse

from machine_failure.config import InterpretationConfig, load_or_default
from machine_failure.data import load_raw
from machine_failure.evaluation import (
    group_medians,
    missed_by_failure_mode,
    out_of_fold_predictions,
    permutation_importances,
)
from machine_failure.experiments.options import (
    add_config_option,
    add_data_option,
    data_option,
)
from machine_failure.models import FINAL_MODEL, TUNED_MODELS, build_tuned_models
from machine_failure.pipelines.splits import load_split

HELP = "what the final model relies on, and which failures it misses"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of ``mf interpret``."""
    add_config_option(parser, "interpret_final.toml")
    add_data_option(parser)
    parser.add_argument(
        "--model",
        choices=TUNED_MODELS,
        help=f"tuned model to study (default: {FINAL_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="decision threshold (default: the model's own)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        help="shuffles per feature and fold in the importance (default: 5)",
    )


def run(args: argparse.Namespace) -> None:
    """``mf interpret``: importance and missed failures, on the training rows."""
    config = load_or_default(
        args.config,
        InterpretationConfig,
        model=args.model,
        threshold=args.threshold,
        n_repeats=args.n_repeats,
        data=data_option(args),
    )
    threshold = config.decision_threshold
    X_train, _, y_train, _ = load_split(config.data)  # the test rows are not used
    raw = load_raw(config.data.path)  # for the failure-mode flags of those rows
    model = build_tuned_models()[config.model]

    importance = permutation_importances(
        model, X_train, y_train, n_repeats=config.n_repeats
    )
    predictions = out_of_fold_predictions(model, X_train, y_train)
    flagged = predictions["probability"] >= threshold
    missed = int(((predictions["label"] == 1) & ~flagged).sum())

    print(
        f"{config.model} on the training set ({len(X_train):,} rows, "
        f"{int(y_train.sum())} failures), threshold {threshold}."
    )
    print("\nDrop in PR-AUC when one input feature is shuffled (held-out folds)")
    print(importance.to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"\nFailures caught and missed at the threshold ({missed} missed in total)")
    modes = missed_by_failure_mode(predictions, threshold, raw)
    print(modes.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nMedian measurements of normal rows, caught failures and missed failures")
    print(group_medians(X_train, predictions, threshold).round(2).to_string())
