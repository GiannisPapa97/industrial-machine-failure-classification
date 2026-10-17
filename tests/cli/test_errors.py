from collections.abc import Callable

import pytest

from machine_failure.cli import EXPECTED_ERRORS, UserError, command
from machine_failure.config import ConfigError
from machine_failure.data import DataValidationError


def failing_with(error: Exception) -> Callable[[], None]:
    @command
    def main() -> None:
        raise error

    return main


@pytest.mark.parametrize(
    "error",
    [
        UserError("bad input"),
        ConfigError("unknown model 'nope'"),
        DataValidationError("missing feature columns: ['Torque']"),
        FileNotFoundError("models/final_model.joblib not found"),
        FileExistsError("out.csv already exists"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_an_expected_error_ends_with_a_one_line_message(
    error: Exception, capsys: pytest.CaptureFixture[str]
) -> None:
    main = failing_with(error)

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 1
    assert capsys.readouterr().err == f"error: {error}\n"


@pytest.mark.parametrize(
    "error", [KeyError("bug"), TypeError("bug"), ValueError("bug")]
)
def test_any_other_error_keeps_its_traceback(error: Exception) -> None:
    main = failing_with(error)

    with pytest.raises(type(error)):
        main()


def test_a_command_that_succeeds_runs_normally(
    capsys: pytest.CaptureFixture[str],
) -> None:
    @command
    def main() -> None:
        print("done")

    main()

    assert capsys.readouterr().out == "done\n"


def test_the_wrapper_keeps_the_name_and_docstring() -> None:
    @command
    def main() -> None:
        """Do the thing."""

    assert (main.__name__, main.__doc__) == ("main", "Do the thing.")


def test_the_project_errors_are_user_errors_and_still_value_errors() -> None:
    for error_type in (ConfigError, DataValidationError):
        assert issubclass(error_type, UserError)
        assert issubclass(error_type, ValueError)  # existing callers catch these


def test_only_user_fixable_errors_are_expected() -> None:
    assert set(EXPECTED_ERRORS) == {UserError, FileNotFoundError, FileExistsError}
