# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards for the package layout itself.

These pin the two things a module move breaks silently: repository-relative data
paths, which resolve to a wrong-but-plausible directory rather than raising, and
the import shape, where a re-exporting `__init__.py` would quietly pull heavy or
optional dependencies into modules that deliberately avoid them.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from synclaw import paths
from synclaw.core import libraries
from synclaw.paths import REPO_ROOT

SRC = REPO_ROOT / "src"


def test_repo_relative_paths_resolve_to_the_repository_root() -> None:
    assert paths.REPO_ROOT == REPO_ROOT
    assert libraries.bundled_reaction_template_dir() == REPO_ROOT / "data" / "reaction_templates"


def test_committed_package_data_is_actually_loaded() -> None:
    # A path equality check alone would not catch a missing committed data file.
    assert libraries.bundled_reaction_catalog()["sources"]
    assert libraries.bundled_reaction_library_versions()


def _import_shape(statement: str, probe: str) -> bool:
    """Return whether `probe` is in sys.modules after running `statement`."""

    code = f"import sys\n{statement}\nraise SystemExit({probe!r} in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode in (0, 1), completed.stderr
    return completed.returncode == 1


@pytest.mark.parametrize(
    ("statement", "probe"),
    [
        # Optional and heavy paths stay off the base import graph.
        ("import synclaw.cli", "synclaw.runners.mcp_server"),
        # The tool contracts and compaction helpers are pure wire format.
        ("import synclaw.tools.compaction", "rdkit"),
        ("import synclaw.tools.contracts", "rdkit"),
        ("import synclaw.rollout.budget", "rdkit"),
    ],
)
def test_module_does_not_pull_in(statement: str, probe: str) -> None:
    assert not _import_shape(statement, probe)


SCRIPT_DIRS = ("scripts",)


def _scripts() -> list[Path]:
    found: list[Path] = []
    for name in SCRIPT_DIRS:
        found.extend(sorted((REPO_ROOT / name).glob("*.py")))
    return found


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_script_imports_cleanly(script: Path) -> None:
    # py_compile only parses; these modules import synclaw at module scope and would
    # otherwise break silently until someone ran them.
    spec = importlib.util.spec_from_file_location(f"_layout_probe_{script.stem}", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves field types via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # optional third-party deps are not layout problems
        pytest.skip(f"{script.name}: {exc}")
    finally:
        sys.modules.pop(spec.name, None)
