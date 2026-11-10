# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-library selections shared by import, search, and rollout configuration."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, TypeAlias

from synclaw import paths

DEFAULT_REACTION_LIBRARIES = ("synformer", "paroutes", "ord", "nvidia")
LibrarySelection: TypeAlias = str | list[str] | tuple[str, ...] | None


def normalize_libraries(value: LibrarySelection) -> tuple[str, ...]:
    """Normalize a source union; an empty selection means all imported libraries."""
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) for v in values):
        raise ValueError("Reaction libraries must be names or a list of names")
    return tuple(dict.fromkeys(name.strip() for v in values for name in v.split(",") if name.strip()))


BUNDLED_REACTION_SOURCE = "bundled"
# Package data, so installed wheels, which ship no template files, can still tell
# when a prepared database was built from different templates.
REACTION_CATALOG_RESOURCE = "reaction_libraries.json"


def bundled_reaction_template_dir() -> Path:
    """Directory the catalog's source ``file`` entries are relative to (source checkouts only)."""
    return paths.DATA_DIR / "reaction_templates"


@cache
def _bundled_reaction_catalog_text() -> str:
    return resources.files("synclaw").joinpath(REACTION_CATALOG_RESOURCE).read_text()


def bundled_reaction_catalog() -> dict[str, Any]:
    """Return the bundled manifest: each source's file, template count and SHA-256, and the family file's."""
    return json.loads(_bundled_reaction_catalog_text())


def bundled_reaction_library_versions() -> dict[str, str]:
    """Map each bundled library to a digest of its template file and the shared family file.

    A change to one source file marks only that library stale; a change to the
    family file marks them all stale.
    """
    catalog = bundled_reaction_catalog()
    families = catalog["families"]["sha256"]
    return {
        name: hashlib.sha256(f"{source['sha256']}:{families}".encode()).hexdigest()
        for name, source in catalog["sources"].items()
    }
