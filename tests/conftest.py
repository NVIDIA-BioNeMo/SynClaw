# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import pytest

# Isolate tests from the user's screen config and real watchlist. Screening tests
# opt in explicitly; other tests exercise the default without requiring downloads.
os.environ["SYNCLAW_TOXICITY_WATCHLIST"] = str(Path(__file__).parent / "fixtures" / "toxicity_watchlist.txt")
os.environ.pop("SYNCLAW_TOXICITY_SCREEN", None)


@pytest.fixture(autouse=True)
def block_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Tests must not perform live HTTP calls; patch the transport.")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)


@pytest.fixture
def inchi_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every Standard InChI conversion fail, as on an RDKit build without InChI."""

    from rdkit.Chem import rdinchi

    def failed_conversion(mol: Any) -> None:
        raise RuntimeError("InChI unavailable")

    monkeypatch.setattr(rdinchi, "MolToInchi", failed_conversion)
