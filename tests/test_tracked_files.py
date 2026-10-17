"""Source and test files must not be hidden by .gitignore.

A broad ignore rule (``models/`` once hid the whole ``machine_failure.models``
package) makes git skip files without a warning, and ruff skips them too. They keep
working on the machine that has them, and are missing for everyone else. In CI the
ignored files are not in the checkout, so this test is for your own machine: it
fails before you push.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKED = ["src", "tests", "scripts", "configs"]
GENERATED = ("__pycache__/", ".pyc", ".pyo", ".egg-info/")


def ignored_files(repo: Path, folders: list[str]) -> list[str]:
    """Files under ``folders`` that git ignores, apart from generated ones."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            *folders,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        path
        for path in result.stdout.splitlines()
        if not any(marker in path for marker in GENERATED)
    ]


def init_repo(path: Path, gitignore: str) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / ".gitignore").write_text(gitignore)


needs_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is not installed"
)


@needs_git
def test_no_source_or_test_file_is_git_ignored() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout (for example an extracted archive)")

    assert ignored_files(ROOT, CHECKED) == []


@needs_git
def test_a_file_hidden_by_a_broad_rule_is_found(tmp_path: Path) -> None:
    init_repo(tmp_path, "models/\n")
    hidden = tmp_path / "src" / "pkg" / "models"
    hidden.mkdir(parents=True)
    (hidden / "base.py").write_text("x = 1\n")
    (tmp_path / "src" / "pkg" / "visible.py").write_text("y = 2\n")

    assert ignored_files(tmp_path, ["src"]) == ["src/pkg/models/base.py"]


@needs_git
def test_caches_and_compiled_files_are_not_reported(tmp_path: Path) -> None:
    init_repo(tmp_path, "__pycache__/\n*.py[cod]\n")
    cache = tmp_path / "src" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "mod.cpython-314.pyc").write_text("")
    (tmp_path / "src" / "stale.pyc").write_text("")

    assert ignored_files(tmp_path, ["src"]) == []
