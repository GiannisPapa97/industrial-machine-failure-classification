"""The console scripts declared in pyproject.toml."""

import importlib
import pkgutil
import tomllib
from pathlib import Path

import pytest

import machine_failure.experiments as experiments

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
SCRIPTS: dict[str, str] = tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]


@pytest.mark.parametrize("name", list(SCRIPTS))
def test_each_script_points_to_a_callable(name: str) -> None:
    module_name, function_name = SCRIPTS[name].split(":")

    function = getattr(importlib.import_module(module_name), function_name)

    assert callable(function)
    assert name.startswith("mf-")


def test_every_command_module_has_a_script() -> None:
    commands = {
        f"{experiments.__name__}.{module.name}:main"
        for module in pkgutil.iter_modules(experiments.__path__)
        if hasattr(
            importlib.import_module(f"{experiments.__name__}.{module.name}"), "main"
        )
    }

    assert commands <= set(SCRIPTS.values())


@pytest.mark.parametrize("name", list(SCRIPTS))
def test_each_script_reports_errors_a_user_can_fix_briefly(name: str) -> None:
    module_name, function_name = SCRIPTS[name].split(":")

    function = getattr(importlib.import_module(module_name), function_name)

    # wrapped by machine_failure.cli.command (functools.wraps sets __wrapped__)
    assert hasattr(function, "__wrapped__")


def test_the_download_script_points_at_the_data_package() -> None:
    assert SCRIPTS["mf-download"] == "machine_failure.data.download:main"
