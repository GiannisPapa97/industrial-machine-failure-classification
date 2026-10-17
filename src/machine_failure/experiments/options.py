"""Options shared by several ``mf`` subcommands."""

import argparse
from pathlib import Path


def add_config_option(parser: argparse.ArgumentParser, example: str) -> None:
    """``--config FILE``: read the settings from a TOML file."""
    parser.add_argument(
        "--config",
        type=Path,
        help=f"TOML file with the settings (see configs/{example}); "
        "the other options override it",
    )
