import argparse
import logging
from collections.abc import Iterator

import pytest

from machine_failure.cli import (
    LOGGER_NAME,
    add_verbosity_options,
    configure_logging,
    level_for,
)
from machine_failure.cli.log import HANDLER_NAME


@pytest.fixture(autouse=True)
def restore_logger() -> Iterator[None]:
    """Leave the project logger as it was, for the other tests."""
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


def ours(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if h.get_name() == HANDLER_NAME]


@pytest.mark.parametrize(
    ("verbose", "quiet", "level"),
    [
        (False, False, logging.INFO),
        (True, False, logging.DEBUG),
        (False, True, logging.WARNING),
    ],
)
def test_the_flags_choose_the_level(verbose: bool, quiet: bool, level: int) -> None:
    assert level_for(verbose=verbose, quiet=quiet) == level


def test_the_options_parse_and_exclude_each_other() -> None:
    parser = argparse.ArgumentParser()
    add_verbosity_options(parser)

    assert parser.parse_args(["-v"]).verbose
    assert parser.parse_args(["--quiet"]).quiet
    defaults = parser.parse_args([])
    assert (defaults.verbose, defaults.quiet) == (False, False)
    with pytest.raises(SystemExit):
        parser.parse_args(["-v", "-q"])


def test_progress_messages_go_to_stderr_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging()
    logger = logging.getLogger(f"{LOGGER_NAME}.somewhere")

    logger.info("cross-validating random_forest")
    logger.debug("a detail")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "cross-validating random_forest\n"


def test_verbose_adds_diagnostics_with_time_and_source(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(verbose=True)

    logging.getLogger(f"{LOGGER_NAME}.somewhere").debug("a detail")

    err = capsys.readouterr().err
    assert "DEBUG machine_failure.somewhere: a detail" in err


def test_quiet_shows_only_warnings(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(quiet=True)
    logger = logging.getLogger(LOGGER_NAME)

    logger.info("progress")
    logger.warning("careful")

    assert capsys.readouterr().err == "careful\n"


def test_configuring_twice_does_not_duplicate_messages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging()
    configure_logging()

    logging.getLogger(LOGGER_NAME).info("once")

    assert (
        ours(logging.getLogger(LOGGER_NAME))
        and len(ours(logging.getLogger(LOGGER_NAME))) == 1
    )
    assert capsys.readouterr().err == "once\n"


def test_other_libraries_are_left_alone() -> None:
    other = logging.getLogger("sklearn")
    before = (list(other.handlers), other.level)

    configure_logging(verbose=True)

    assert (list(other.handlers), other.level) == before
