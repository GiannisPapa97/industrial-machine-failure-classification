"""Tests for scripts/coverage_by_package.py, which is not part of the package."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "coverage_by_package.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("coverage_by_package", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["coverage_by_package"] = module
    spec.loader.exec_module(module)
    return module


script = load_script()


def file_entry(
    statements: int, missed: int, branches: int = 0, partial: int = 0
) -> dict[str, Any]:
    """The part of a ``coverage json`` file entry that the script reads."""
    return {
        "summary": {
            "num_statements": statements,
            "missing_lines": missed,
            "covered_lines": statements - missed,
            "num_branches": branches,
            "num_partial_branches": partial,
            "covered_branches": branches - partial,
        }
    }


REPORT = {
    "files": {
        "src/machine_failure/__init__.py": file_entry(0, 0),
        "src/machine_failure/data/loading.py": file_entry(20, 0, 4),
        "src/machine_failure/data/schema.py": file_entry(30, 3, 6, 2),
        "src/machine_failure/models/final.py": file_entry(50, 10, 10, 0),
        "tests/test_x.py": file_entry(5, 5),
    }
}


@pytest.mark.parametrize(
    ("path", "package"),
    [
        ("src/machine_failure/data/loading.py", "data"),
        ("src/machine_failure/models/deep/er/x.py", "models"),
        ("src/machine_failure/__init__.py", "machine_failure"),
        ("src\\machine_failure\\config\\schemas.py", "config"),
    ],
)
def test_a_file_belongs_to_the_package_it_is_in(path: str, package: str) -> None:
    assert script.package_of(path) == package


def test_files_are_summed_by_package_and_other_paths_are_ignored() -> None:
    packages = script.by_package(REPORT)

    assert [p.name for p in packages] == ["data", "machine_failure", "models"]
    data = packages[0]
    assert (data.statements, data.missed, data.branches, data.partial) == (50, 3, 10, 2)
    assert (data.covered_lines, data.covered_branches) == (47, 8)


def test_percent_counts_statements_and_branches_like_coverage_py() -> None:
    data = script.by_package(REPORT)[0]

    assert data.percent == pytest.approx(100 * (47 + 8) / (50 + 10))


def test_a_package_with_nothing_to_cover_is_fully_covered() -> None:
    root = script.by_package(REPORT)[1]

    assert (root.statements, root.branches) == (0, 0)
    assert root.percent == 100.0


def test_total_sums_all_packages() -> None:
    total = script.total_of(script.by_package(REPORT))

    assert (total.name, total.statements, total.missed) == ("Total", 100, 13)
    assert (total.branches, total.partial) == (20, 2)


@pytest.mark.parametrize(
    ("percent", "shown"),
    [(100.0, "100.0%"), (99.99, "99.9%"), (91.25, "91.2%"), (0.0, "0.0%")],
)
def test_the_percentage_is_rounded_down(percent: float, shown: str) -> None:
    assert script.display(percent) == shown


def test_below_lists_the_packages_under_the_minimum() -> None:
    packages = script.by_package(REPORT)

    assert [p.name for p in script.below(packages, 90)] == ["models"]  # 83.3%
    assert script.below(packages, 80) == []


def test_the_table_has_a_row_per_package_a_total_and_a_verdict() -> None:
    table = script.format_table(script.by_package(REPORT), 90)

    lines = table.splitlines()
    assert lines[0] == "| Package | Stmts | Miss | Branch | BrPart | Cover |"
    assert lines[2] == "| `data` | 50 | 3 | 10 | 2 | 91.6% |"
    assert lines[3] == "| `models` | 50 | 10 | 10 | 0 | 83.3% |"
    assert lines[4] == "| **Total** | **100** | **13** | **20** | **2** | **87.5%** |"
    assert lines[-1] == "Below the 90% minimum: `models` (83.3%)."


def test_a_package_with_nothing_to_cover_has_no_row() -> None:
    table = script.format_table(script.by_package(REPORT), 90)

    assert "machine_failure" not in table


def test_the_verdict_is_positive_when_every_package_passes() -> None:
    table = script.format_table(script.by_package(REPORT), 80)

    assert table.splitlines()[-1] == "Every package is at or above the 80% minimum."


def test_the_minimum_is_read_from_pyproject(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.package-coverage]\nfail_under = 87.5\n")

    assert script.read_minimum(pyproject) == 87.5


def test_a_missing_minimum_is_an_error(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.other]\nx = 1\n")

    with pytest.raises(SystemExit, match="fail_under"):
        script.read_minimum(pyproject)


def test_the_project_minimum_is_ninety_percent() -> None:
    assert script.read_minimum(ROOT / "pyproject.toml") == 90


@pytest.fixture
def report_file(tmp_path: Path) -> Path:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(REPORT))
    return path


def test_main_prints_the_table_and_fails_below_the_minimum(
    report_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = script.main(["--json", str(report_file), "--fail-under", "90"])

    assert status == 1
    assert "Below the 90% minimum: `models`" in capsys.readouterr().out


def test_main_succeeds_when_every_package_passes(
    report_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = script.main(["--json", str(report_file), "--fail-under", "80"])

    assert status == 0
    assert "Every package is at or above" in capsys.readouterr().out


def test_main_uses_the_project_minimum_by_default(report_file: Path) -> None:
    assert script.main(["--json", str(report_file)]) == 1  # models is at 83.3%


def test_main_fails_when_there_is_nothing_to_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"files": {"tests/test_x.py": file_entry(1, 0)}}))

    assert script.main(["--json", str(empty), "--fail-under", "90"]) == 1
    assert "No coverage data" in capsys.readouterr().err


def test_without_a_json_file_the_data_of_the_last_run_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # coverage data does not exist in a fresh folder: the script says to run pytest
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="run pytest first"):
        script.load_report(None)


def test_the_script_reads_the_data_of_a_real_coverage_run(tmp_path: Path) -> None:
    (tmp_path / "program.py").write_text("print('hello')\n")
    run = [sys.executable, "-m", "coverage", "run", "program.py"]
    subprocess.run(run, cwd=tmp_path, check=True, capture_output=True)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--fail-under", "90"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # the run covered nothing of src/machine_failure, so there is nothing to report
    assert result.returncode == 1
    assert "No coverage data" in result.stderr
