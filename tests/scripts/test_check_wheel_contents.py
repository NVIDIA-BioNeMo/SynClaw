# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release checks must catch a stale wheel or a mismatched release tag."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

from synclaw.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "check_wheel_contents", REPO_ROOT / "scripts" / "check_wheel_contents.py"
)
check_wheel_contents = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_wheel_contents)


@pytest.fixture
def wheel_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI_COMMIT_TAG", raising=False)
    package = tmp_path / "src" / "synclaw"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.2.1"\n')
    (package / "providers.json").write_text("{}")
    dist = tmp_path / "dist"
    dist.mkdir()

    def build(*, metadata_version="0.2.1", code_version="0.2.1", include_data=True):
        path = dist / "synclaw-0.2.1-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("synclaw/__init__.py", f'__version__ = "{code_version}"\n')
            if include_data:
                archive.writestr("synclaw/providers.json", "{}")
            archive.writestr(
                "synclaw-0.2.1.dist-info/METADATA", f"Name: synclaw\nVersion: {metadata_version}\n"
            )
        return path

    return build


def test_matching_wheel_and_release_tag_pass(wheel_project, monkeypatch):
    wheel_project()
    monkeypatch.setenv("CI_COMMIT_TAG", "v0.2.1")
    assert check_wheel_contents.main() == 0


@pytest.mark.parametrize(
    "options",
    [{"metadata_version": "0.1.0"}, {"code_version": "0.1.0"}, {"include_data": False}],
)
def test_stale_metadata_or_missing_package_data_fails(wheel_project, options):
    wheel_project(**options)
    assert check_wheel_contents.main() == 1


def test_mismatched_release_tag_fails(wheel_project, monkeypatch):
    wheel_project()
    monkeypatch.setenv("CI_COMMIT_TAG", "v0.2.0")
    assert check_wheel_contents.main() == 1


def test_stale_wheel_cannot_be_selected_silently(wheel_project):
    path = wheel_project()
    (Path("dist") / "synclaw-0.1.0-py3-none-any.whl").write_bytes(path.read_bytes())
    assert check_wheel_contents.main() == 1
