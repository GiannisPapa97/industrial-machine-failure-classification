"""The ``mf`` command declared in pyproject.toml, and its subcommands."""

import importlib
import pkgutil
import runpy
import sys
import tomllib
from pathlib import Path

import pytest

import machine_failure.experiments as experiments
from machine_failure.experiments import app

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
SCRIPTS: dict[str, str] = tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]


def test_the_only_script_is_mf() -> None:
    assert SCRIPTS == {"mf": "machine_failure.experiments.app:main"}


def test_mf_reports_errors_a_user_can_fix_briefly() -> None:
    # wrapped by machine_failure.cli.command (functools.wraps sets __wrapped__)
    assert hasattr(app.main, "__wrapped__")


def test_every_command_module_is_a_subcommand() -> None:
    modules = {
        importlib.import_module(f"{experiments.__name__}.{info.name}")
        for info in pkgutil.iter_modules(experiments.__path__)
    }
    commands = {module for module in modules if hasattr(module, "run")}

    assert commands <= set(app.COMMANDS.values())


@pytest.mark.parametrize("name", list(app.COMMANDS))
def test_each_subcommand_has_help_options_and_a_run_function(name: str) -> None:
    module = app.COMMANDS[name]

    assert isinstance(module.HELP, str) and module.HELP
    assert callable(module.add_arguments) and callable(module.run)


@pytest.mark.parametrize("name", list(app.COMMANDS))
def test_each_subcommand_shows_its_help(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        app.main([name, "--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: mf {name}")
    assert "--verbose" in out and "--quiet" in out


def test_mf_lists_the_subcommands_in_workflow_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        app.main(["--help"])

    out = capsys.readouterr().out
    positions = [out.index(f"    {name} ") for name in app.COMMANDS]
    assert positions == sorted(positions)


def test_mf_without_a_subcommand_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        app.main([])

    assert exit_info.value.code == 2
    assert "required: COMMAND" in capsys.readouterr().err


def test_an_expected_error_in_a_subcommand_is_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        app.main(["predict", str(tmp_path / "missing.csv")])

    assert exit_info.value.code == 1
    assert capsys.readouterr().err == f"error: {tmp_path / 'missing.csv'} not found\n"


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_the_package_runs_mf(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["mf", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("machine_failure", run_name="__main__")

    assert exit_info.value.code == 0
    assert "usage: mf" in capsys.readouterr().out
