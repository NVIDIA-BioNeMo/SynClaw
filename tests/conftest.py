# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from synclaw.core import db

# Isolate tests from the user's screen config and real watchlist. Screening tests
# opt in explicitly; other tests exercise the default without requiring downloads.
os.environ["SYNCLAW_TOXICITY_WATCHLIST"] = str(
    Path(__file__).parent / "fixtures" / "toxicity_watchlist.txt"
)
os.environ.pop("SYNCLAW_TOXICITY_SCREEN", None)


@pytest.fixture(autouse=True)
def block_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Tests must not perform live HTTP calls; patch the transport.")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)


@pytest.fixture
def inchi_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every Standard InChI conversion raise, as RDKit does on an internal error."""

    from rdkit.Chem import rdinchi

    def failed_conversion(mol: Any) -> None:
        raise RuntimeError("InChI unavailable")

    monkeypatch.setattr(rdinchi, "MolToInchi", failed_conversion)


@pytest.fixture(scope="session")
def bundled_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A database holding every bundled reaction library; tests must not modify it."""

    from synclaw.core.reactions import ingest_reaction_libraries

    database = tmp_path_factory.mktemp("bundled") / "catalog.sqlite"
    ingest_reaction_libraries(database)
    return database


def numbered_ids(db_path: str | Path) -> dict[str, str]:
    """Map each template's source ID, as its import file named it, to its numbered ID."""

    with db.session(db_path) as conn:
        return dict(
            conn.execute(
                "SELECT json_extract(metadata_json, '$.source_id'), reaction_id FROM reaction_templates"
            ).fetchall()
        )


def numbered_id(db_path: str | Path, source_id: str) -> str:
    return numbered_ids(db_path)[source_id]


@pytest.fixture(scope="session")
def bundled_ids(bundled_db: Path) -> dict[str, str]:
    """Map bundled source IDs, as the data notes name them, to the database's numbered IDs."""

    return numbered_ids(bundled_db)
