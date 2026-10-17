"""The final model: the chosen tuned model, trained on the whole training set."""

import argparse
from pathlib import Path

from machine_failure.cli import add_verbosity_options, command, configure_logging
from machine_failure.config import TrainingConfig, load_or_default
from machine_failure.models import FINAL_MODEL, FINAL_THRESHOLD, MODEL_PATH
from machine_failure.pipelines import TrainingPipeline


@command
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML file with the training settings (see configs/train_final.toml); "
        "the options below override it",
    )
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
    add_verbosity_options(parser)
    args = parser.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    config = load_or_default(
        args.config,
        TrainingConfig,
        model=args.model,
        threshold=args.threshold,
        output_path=args.output,
    )
    pipeline = TrainingPipeline(config)
    result = pipeline.run()  # the test set is not used here
    path = pipeline.save(result, overwrite=args.overwrite)
    print(result.describe())
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
