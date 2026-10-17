"""What every command shares: reporting errors a user can fix, and logging."""

from machine_failure.cli.errors import EXPECTED_ERRORS, UserError, command
from machine_failure.cli.log import (
    LOGGER_NAME,
    add_verbosity_options,
    configure_logging,
    level_for,
)

__all__ = [
    "EXPECTED_ERRORS",
    "LOGGER_NAME",
    "UserError",
    "add_verbosity_options",
    "command",
    "configure_logging",
    "level_for",
]
