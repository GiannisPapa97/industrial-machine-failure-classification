"""The final model: the chosen tuned model, trained on the whole training set."""

import argparse
from pathlib import Path

from machine_failure.data import load_xy, split_train_test
from machine_failure.final_model import (
    FINAL_MODEL,
    FINAL_THRESHOLD,
    MODEL_PATH,
    save_final_model,
    train_final_model,
)


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

    X, y = load_xy()
    X_train, _, y_train, _ = split_train_test(X, y)  # the test set is not used here
    final = train_final_model(X_train, y_train, args.model, args.threshold)
    path = save_final_model(final, args.output, overwrite=args.overwrite)
    print(
        f"Trained {final.name} on the training set ({len(X_train):,} rows, "
        f"{int(y_train.sum())} failures); threshold {final.threshold}."
    )
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
