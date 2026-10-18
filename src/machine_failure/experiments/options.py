"""Options shared by several ``mf`` subcommands."""

import argparse
from pathlib import Path

from machine_failure.config import DataConfig


def add_config_option(parser: argparse.ArgumentParser, example: str) -> None:
    """``--config FILE``: read the settings from a TOML file."""
    parser.add_argument(
        "--config",
        type=Path,
        help=f"TOML file with the settings (see configs/{example}); "
        "the other options override it",
    )


def add_data_option(parser: argparse.ArgumentParser) -> None:
    """``--data FILE``: the dataset to use instead of the configured one."""
    parser.add_argument(
        "--data",
        type=Path,
        help=f"the dataset CSV (default: {DataConfig().path})",
    )


def data_option(args: argparse.Namespace) -> DataConfig | None:
    """The data config ``--data`` asks for, or ``None`` when it was not given."""
    return DataConfig(path=args.data) if args.data is not None else None
