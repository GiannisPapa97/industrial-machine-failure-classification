"""Turning the errors a user can fix into a short message instead of a traceback."""

import functools
import sys
from collections.abc import Callable


class UserError(Exception):
    """An error the user can fix: bad input, a wrong setting, a missing file.

    The project's own errors for such cases (``ConfigError``,
    ``DataValidationError``) derive from it, so the commands can report them
    briefly. Errors that do not derive from it are bugs and keep their traceback.
    """


# Raised for a missing input file or an output file that must not be replaced.
EXPECTED_ERRORS: tuple[type[Exception], ...] = (
    UserError,
    FileNotFoundError,
    FileExistsError,
)


def command(main: Callable[[], None]) -> Callable[[], None]:
    """Decorate a command's ``main`` so expected errors end it with a short message.

    An expected error (see ``EXPECTED_ERRORS``) prints ``error: <message>`` on
    stderr and exits with status 1. Any other error propagates unchanged, with its
    traceback, because it points to a bug.
    """

    @functools.wraps(main)
    def wrapper() -> None:
        try:
            main()
        except EXPECTED_ERRORS as error:
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(1) from None

    return wrapper
