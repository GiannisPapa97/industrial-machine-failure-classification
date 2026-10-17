"""Summarize test coverage by package and fail if a package is below the minimum.

Run it after the tests, from the repository root:

    uv run pytest -m "not integration"
    uv run python scripts/coverage_by_package.py

It reads the coverage data of the last run (the ``.coverage`` file), prints a
Markdown table with one row per package of ``src/machine_failure`` and a total, and
exits with status 1 if any package is below the minimum. The minimum is
``fail_under`` in the ``[tool.package-coverage]`` section of ``pyproject.toml``;
``--fail-under`` overrides it.
"""

import argparse
import json
import math
import subprocess
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = "src/machine_failure/"
ROOT_NAME = "machine_failure"  # files directly in the root package
PYPROJECT = Path("pyproject.toml")


@dataclass(frozen=True)
class Coverage:
    """Statements and branches of a package, and how many of them the tests ran."""

    name: str
    statements: int
    missed: int
    branches: int
    partial: int
    covered_lines: int
    covered_branches: int

    @property
    def percent(self) -> float:
        """Coverage as coverage.py computes it: statements and branches together.

        A package with nothing to cover counts as fully covered.
        """
        total = self.statements + self.branches
        if total == 0:
            return 100.0
        return 100 * (self.covered_lines + self.covered_branches) / total


def package_of(path: str) -> str:
    """The package a source file belongs to: ``src/machine_failure/<package>/...``."""
    relative = path.replace("\\", "/").removeprefix(PACKAGE_ROOT)
    return relative.split("/")[0] if "/" in relative else ROOT_NAME


def by_package(report: dict[str, Any]) -> list[Coverage]:
    """Sum the per-file figures of a ``coverage json`` report by package.

    Packages come out in alphabetical order. Files outside ``src/machine_failure``
    are ignored.
    """
    sums: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for path, data in report["files"].items():
        if not path.replace("\\", "/").startswith(PACKAGE_ROOT):
            continue
        summary = data["summary"]
        totals = sums[package_of(path)]
        totals["statements"] += summary["num_statements"]
        totals["missed"] += summary["missing_lines"]
        totals["branches"] += summary["num_branches"]
        totals["partial"] += summary["num_partial_branches"]
        totals["covered_lines"] += summary["covered_lines"]
        totals["covered_branches"] += summary["covered_branches"]
    return [Coverage(name, **sums[name]) for name in sorted(sums)]


def total_of(packages: list[Coverage]) -> Coverage:
    """The figures of all packages together."""
    return Coverage(
        "Total",
        sum(p.statements for p in packages),
        sum(p.missed for p in packages),
        sum(p.branches for p in packages),
        sum(p.partial for p in packages),
        sum(p.covered_lines for p in packages),
        sum(p.covered_branches for p in packages),
    )


def display(percent: float) -> str:
    """One decimal, rounded down so that 99.99 never shows as 100.0."""
    return f"{math.floor(percent * 10) / 10:.1f}%"


def below(packages: list[Coverage], minimum: float) -> list[Coverage]:
    """The packages whose coverage is under ``minimum`` percent."""
    return [p for p in packages if p.percent < minimum]


def format_table(packages: list[Coverage], minimum: float) -> str:
    """A Markdown table with a row per package and a total, and a verdict line.

    Packages with nothing to cover (such as the empty root ``__init__.py``) are left
    out of the rows but still count in the total.
    """
    lines = [
        "| Package | Stmts | Miss | Branch | BrPart | Cover |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in packages:
        if p.statements or p.branches:
            lines.append(_row(p, bold=False))
    lines.append(_row(total_of(packages), bold=True))
    failing = below(packages, minimum)
    if failing:
        names = ", ".join(f"`{p.name}` ({display(p.percent)})" for p in failing)
        verdict = f"Below the {minimum:g}% minimum: {names}."
    else:
        verdict = f"Every package is at or above the {minimum:g}% minimum."
    return "\n".join([*lines, "", verdict])


def _row(p: Coverage, *, bold: bool) -> str:
    cells = [p.name, p.statements, p.missed, p.branches, p.partial, display(p.percent)]
    if bold:
        cells = [f"**{cell}**" for cell in cells]
    else:
        cells[0] = f"`{p.name}`"
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def read_minimum(pyproject: Path = PYPROJECT) -> float:
    """``fail_under`` from the ``[tool.package-coverage]`` section."""
    config = tomllib.loads(pyproject.read_text())
    try:
        return float(config["tool"]["package-coverage"]["fail_under"])
    except KeyError:
        raise SystemExit(
            f"{pyproject} has no [tool.package-coverage] fail_under"
        ) from None


def load_report(json_file: Path | None) -> dict[str, Any]:
    """The coverage report: from ``json_file``, or from the data of the last run."""
    if json_file is not None:
        report: dict[str, Any] = json.loads(json_file.read_text())
        return report
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "json", "-o", "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"could not read the coverage data; run pytest first.\n{result.stderr}"
        )
    report = json.loads(result.stdout)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fail-under",
        type=float,
        help="minimum coverage per package in percent (default: from pyproject.toml)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_file",
        help="a coverage.py JSON report to read instead of the last run's data",
    )
    args = parser.parse_args(argv)

    minimum = args.fail_under if args.fail_under is not None else read_minimum()
    packages = by_package(load_report(args.json_file))
    if not packages:
        print("No coverage data for src/machine_failure was found.", file=sys.stderr)
        return 1
    print(format_table(packages, minimum))
    return 1 if below(packages, minimum) else 0


if __name__ == "__main__":
    sys.exit(main())
