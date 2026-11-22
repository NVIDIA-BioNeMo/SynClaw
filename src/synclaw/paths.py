# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem locations: bundled repository data and the CLI's default paths.

``REPO_ROOT`` and ``DATA_DIR`` exist only in a source checkout; an installed wheel ships
no ``data/`` directory. Keep this module at the package root, because ``REPO_ROOT`` is
derived from its depth.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"

# Defaults below are relative to the working directory, not the repository.
DEFAULT_DB_PATH = Path("var/synclaw.sqlite")
