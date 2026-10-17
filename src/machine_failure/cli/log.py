"""Logging for the commands: progress and diagnostics on stderr.

Results (tables, scores, predictions) are the output of a command and are
printed to stdout. Progress messages ("cross-validating random_forest") and
diagnostics go through the standard ``logging`` module, to stderr, under the
``machine_failure`` logger. Nothing is shown below WARNING unless a command (or
you) configures logging, so using the library from a notebook stays quiet.
"""

import argparse
import logging
import sys

LOGGER_NAME = "machine_failure"
HANDLER_NAME = "machine_failure.cli"
FORMAT = "%(message)s"
VERBOSE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def add_verbosity_options(parser: argparse.ArgumentParser) -> None:
    """Add ``--verbose`` and ``--quiet`` (which exclude each other) to ``parser``."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also show diagnostic messages, with time and source",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="show only warnings and errors, no progress messages",
    )


def level_for(*, verbose: bool = False, quiet: bool = False) -> int:
    """DEBUG when verbose, WARNING when quiet, INFO (progress) otherwise."""
    if verbose:
        return logging.DEBUG
    if quiet:
        return logging.WARNING
    return logging.INFO


def configure_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    """Send the project's log messages to stderr at the chosen level.

    Only the ``machine_failure`` logger is configured, so third-party libraries
    keep their own settings. Calling it again replaces the earlier setup.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if handler.get_name() == HANDLER_NAME:
            logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(logging.Formatter(VERBOSE_FORMAT if verbose else FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level_for(verbose=verbose, quiet=quiet))
