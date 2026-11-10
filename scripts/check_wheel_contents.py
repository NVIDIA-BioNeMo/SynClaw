# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assert the built wheel contains every tracked package module.

pytest reads from `src/` via `pythonpath`, so a subpackage missing `__init__.py`
-- or omitted by `packages.find` -- passes CI and only surfaces in an
installed wheel. Run after `uv build`.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.version import InvalidVersion, Version

SRC = Path("src")


def main(dist_dir: Path = Path("dist")) -> int:
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        print(f"no wheel in {dist_dir}; run `uv build` first", file=sys.stderr)
        return 1
    if len(wheels) != 1:
        print(f"expected exactly one wheel in {dist_dir}; remove stale builds first", file=sys.stderr)
        return 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        shipped = set(wheel.namelist())
        metadata_paths = [name for name in shipped if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            print("expected one distribution METADATA file", file=sys.stderr)
            return 1
        metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))
        source_version = read_version((SRC / "synclaw" / "__init__.py").read_text())
        if "synclaw/__init__.py" not in shipped:
            print("missing from wheel:\n  src/synclaw/__init__.py", file=sys.stderr)
            return 1
        wheel_version = read_version(wheel.read("synclaw/__init__.py").decode())
    # METADATA carries the PEP 440 normalized form, so compare parsed versions.
    if source_version != wheel_version or Version(source_version) != Version(metadata["Version"]):
        print("source, wheel and distribution versions disagree", file=sys.stderr)
        return 1
    release_tag = os.environ.get("CI_COMMIT_TAG")
    if release_tag:
        try:
            tag_matches = Version(release_tag.removeprefix("v")) == Version(source_version)
        except InvalidVersion:
            tag_matches = False
        if not tag_matches:
            print(f"release tag {release_tag!r} does not match {source_version}", file=sys.stderr)
            return 1
    # Walk the tree rather than asking git: the CI image has no git binary, and
    # what matters is that every module present in src/ reached the wheel.
    # providers.json and reaction_libraries.json are package data the runtime requires.
    modules = sorted(
        p
        for p in (SRC / "synclaw").rglob("*")
        if p.suffix in {".py", ".json"} and "__pycache__" not in p.parts
    )
    missing = sorted(str(p) for p in modules if str(p.relative_to(SRC)) not in shipped)
    if missing:
        print("missing from wheel:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    print(f"wheel {source_version} contains all {len(modules)} package files")
    return 0


def read_version(source: str) -> str:
    """Read the literal version without importing the package being checked."""

    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError("Package __version__ must be a string literal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    raise SystemExit(main(parser.parse_args().dist_dir))
