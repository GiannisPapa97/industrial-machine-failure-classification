"""Score rows of measurements with the saved final model.

The input is a CSV file with the columns Type, Air temperature, Process
temperature, Rotational speed, Torque and Tool wear (other columns are ignored).
The output has one row per input row, in the same order, with the failure
probability and whether it is flagged at the model's threshold.

Saved models are pickles: use only a model file you created yourself.
"""

import argparse
import logging
import sys
from pathlib import Path

from machine_failure.config import InferenceConfig
from machine_failure.models import MODEL_PATH
from machine_failure.pipelines import InferencePipeline

logger = logging.getLogger(__name__)

HELP = "score new rows with the saved model"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of ``mf predict``."""
    parser.add_argument("input", type=Path, help="CSV file with the rows to score")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"the saved final model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the scores to this CSV file (default: print them)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output file if it already exists",
    )


def run(args: argparse.Namespace) -> None:
    """``mf predict``: score the rows of a CSV file with the saved model."""
    if args.output and args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; pass --overwrite to replace it"
        )
    pipeline = InferencePipeline(InferenceConfig(model_path=args.model_path))
    scores = pipeline.predict_csv(args.input)
    scores.to_csv(args.output or sys.stdout, index_label="row")
    logger.info(
        "%s rows scored, %d flagged at threshold %s (%s).",
        f"{len(scores):,}",
        int(scores["flagged"].sum()),
        pipeline.final.threshold,
        pipeline.final.name,
    )
