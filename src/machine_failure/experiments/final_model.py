"""Train the final model on the training rows and save it.

By default this is the chosen tuned model (random forest) with its own decision
threshold. The held-out test rows are not used.
"""

import argparse
from pathlib import Path

from machine_failure.config import TrainingConfig, load_or_default
from machine_failure.experiments.options import (
    add_config_option,
    add_data_option,
    data_option,
)
from machine_failure.models import FINAL_MODEL, FINAL_THRESHOLD, MODEL_PATH
from machine_failure.pipelines import TrainingPipeline

HELP = "train the final model and save it"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of ``mf train``."""
    add_config_option(parser, "train_final.toml")
    add_data_option(parser)
    parser.add_argument(
        "--model",
        help=f"tuned model to train (default: {FINAL_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="decision threshold (default: the tuned model's own, "
        f"{FINAL_THRESHOLD} for {FINAL_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"where to save the model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the file if it already exists",
    )


def run(args: argparse.Namespace) -> None:
    """``mf train``: train on the training rows and save the model."""
    config = load_or_default(
        args.config,
        TrainingConfig,
        model=args.model,
        threshold=args.threshold,
        output_path=args.output,
        data=data_option(args),
    )
    pipeline = TrainingPipeline(config)
    result = pipeline.run()  # the test set is not used here
    path = pipeline.save(result, overwrite=args.overwrite)
    print(result.describe())
    print(f"Saved to {path}")
