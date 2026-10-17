"""Download the AI4I 2020 Predictive Maintenance dataset."""

import argparse
from pathlib import Path

from ucimlrepo import fetch_ucirepo

from machine_failure.cli import command

UCI_DATASET_ID = 601
RAW_FILENAME = "ai4i2020.csv"
DEFAULT_RAW_DIR = Path("data/raw")


def download_dataset(
    dest_dir: Path = DEFAULT_RAW_DIR, *, overwrite: bool = False
) -> Path:
    """Fetch the dataset from the UCI repository and save it as a CSV.

    An existing file is never replaced unless ``overwrite`` is True.
    Returns the path of the CSV file.
    """
    path = dest_dir / RAW_FILENAME
    if path.exists() and not overwrite:
        return path

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset.data.original.to_csv(path, index=False)
    return path


@command
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"directory for the CSV file (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the file if it already exists",
    )
    args = parser.parse_args()
    print(download_dataset(args.dest, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
