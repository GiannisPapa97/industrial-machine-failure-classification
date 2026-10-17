"""The final model: the chosen tuned model, trained on the whole training set."""

import argparse
from pathlib import Path

from machine_failure.config import TrainingConfig
from machine_failure.models import FINAL_MODEL, FINAL_THRESHOLD, MODEL_PATH
from machine_failure.pipelines import TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=FINAL_MODEL,
        help=f"tuned model to train (default: {FINAL_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=FINAL_THRESHOLD,
        help=f"decision threshold (default: {FINAL_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH,
        help=f"where to save the model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the file if it already exists",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        model=args.model, threshold=args.threshold, output_path=args.output
    )
    pipeline = TrainingPipeline(config)
    result = pipeline.run()  # the test set is not used here
    path = pipeline.save(result, overwrite=args.overwrite)
    print(result.describe())
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
