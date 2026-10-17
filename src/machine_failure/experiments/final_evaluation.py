"""Evaluate the final model once on the held-out test set.

The test set is meant to be used a single time, on the model and threshold chosen
with cross-validation on the training set. Nothing in this module should be used
to tune the model or the threshold.
"""

import argparse
import sys
from pathlib import Path

from machine_failure.config import N_BOOTSTRAP, EvaluationConfig, load_or_default
from machine_failure.experiments.options import (
    add_config_option,
    add_data_option,
    data_option,
)
from machine_failure.models import MODEL_PATH
from machine_failure.pipelines import EvaluationPipeline, UnconfirmedTestSetError

HELP = "evaluate the final model once on the held-out test set"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of ``mf evaluate``."""
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="required: confirms you want to use the test set, which is meant to "
        "be done once",
    )
    add_config_option(parser, "evaluate_final.toml")
    add_data_option(parser)
    parser.add_argument(
        "--model-path",
        type=Path,
        help=f"the saved final model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        help=f"bootstrap resamples for the intervals (default: {N_BOOTSTRAP})",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="do not compute the cross-validation reference (saves about a minute)",
    )


def run(args: argparse.Namespace) -> None:
    """``mf evaluate``: score the saved model once on the test set."""
    config = load_or_default(
        args.config,
        EvaluationConfig,
        model_path=args.model_path,
        n_bootstrap=args.n_bootstrap,
        reference=False if args.skip_reference else None,
        data=data_option(args),
    )
    try:
        result = EvaluationPipeline(config).run(confirm=args.confirm)
    except UnconfirmedTestSetError:
        print(
            "The test set is meant to be used once, on the final model. "
            "Run again with --confirm to evaluate on it.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    print(result.format())
