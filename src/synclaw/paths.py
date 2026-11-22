# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single filesystem anchor for repository-relative package data.

Modules that resolved the repository root from their own `__file__` depth would
silently point at the wrong directory after a move into a subpackage rather than
raising. This module stays at the package root, so the depth assumption lives in
exactly one place and cannot drift.

Repository-relative data is only present in a source checkout; an installed wheel
ships no `data/` directory. Callers already treat these paths as optional and fall
back when the file is absent.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"

# The CLI's --db default, relative to the working directory rather than the repository.
DEFAULT_DB_PATH = Path("var/synclaw.sqlite")
