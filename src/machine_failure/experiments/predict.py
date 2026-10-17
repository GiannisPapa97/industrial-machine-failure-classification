"""Score rows of measurements with the saved final model.

The input is a CSV file with the columns Type, Air temperature, Process
temperature, Rotational speed, Torque and Tool wear (other columns are ignored).
The output has one row per input row, in the same order, with the failure
probability and whether it is flagged at the model's threshold.

Saved models are pickles: use only a model file you created yourself.
"""

import argparse
import sys
from pathlib import Path

from machine_failure.config import InferenceConfig
from machine_failure.models import MODEL_PATH
from machine_failure.pipelines import InferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    if args.output and args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} already exists; pass --overwrite to replace it")

    pipeline = InferencePipeline(InferenceConfig(model_path=args.model_path))
    scores = pipeline.predict_csv(args.input)
    scores.to_csv(args.output or sys.stdout, index_label="row")
    print(
        f"{len(scores):,} rows scored, {int(scores['flagged'].sum())} flagged at "
        f"threshold {pipeline.final.threshold} ({pipeline.final.name}).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
