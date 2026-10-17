"""Packages import only from the packages below them (see the README layout)."""

import ast
from pathlib import Path

import pytest

import machine_failure

ROOT = Path(machine_failure.__file__).parent
# From the bottom of the stack to the top.
LAYERS = [
    "data",
    "features",
    "evaluation",
    "models",
    "config",
    "pipelines",
    "experiments",
]


def imported_packages(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
        elif isinstance(node, ast.Import):
            parts = node.names[0].name.split(".")
        else:
            continue
        if parts[0] == "machine_failure" and len(parts) > 1:
            found.add(parts[1])
    return found


@pytest.mark.parametrize("layer", LAYERS)
def test_a_package_imports_only_from_the_packages_below_it(layer: str) -> None:
    allowed = {layer, *LAYERS[: LAYERS.index(layer)]}

    for path in (ROOT / layer).rglob("*.py"):
        assert imported_packages(path) <= allowed, path


def test_every_package_is_in_the_layer_list() -> None:
    packages = {p.name for p in ROOT.iterdir() if (p / "__init__.py").exists()}

    assert packages == set(LAYERS)
