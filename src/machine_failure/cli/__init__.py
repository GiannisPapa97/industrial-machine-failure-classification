"""What every command shares: reporting the errors a user can fix."""

from machine_failure.cli.errors import EXPECTED_ERRORS, UserError, command

__all__ = ["EXPECTED_ERRORS", "UserError", "command"]
