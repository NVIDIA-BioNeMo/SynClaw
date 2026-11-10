# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Downloadable public data assets for local examples."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PublicDataFile:
    name: str
    url: str
    path: Path
    description: str


PUBLIC_DATA_FILES: tuple[PublicDataFile, ...] = (
    PublicDataFile(
        name="synformer-reaction-templates",
        url="https://raw.githubusercontent.com/wenhao-gao/synformer/main/data/rxn_templates/comprehensive.txt",
        path=Path("data/reaction_templates/synformer_comprehensive.txt"),
        description="SynFormer comprehensive reaction SMARTS templates.",
    ),
    PublicDataFile(
        name="synformer-chembl-1k",
        url="https://raw.githubusercontent.com/wenhao-gao/synformer/main/data/chembl_filtered_1k.txt",
        path=Path("data/synformer_test_sets/chembl_filtered_1k.txt"),
        description="SynFormer public ChEMBL 1k target set.",
    ),
    PublicDataFile(
        name="synformer-enamine-real-1k",
        url="https://raw.githubusercontent.com/wenhao-gao/synformer/main/data/enamine_smiles_1k.txt",
        path=Path("data/synformer_test_sets/enamine_smiles_1k.txt"),
        description="SynFormer public Enamine REAL 1k target-molecule set.",
    ),
    PublicDataFile(
        name="reasyn-mols-to-filter",
        url="https://raw.githubusercontent.com/NVIDIA-BioNeMo/ReaSyn/reasyn_v2/data/mols_to_filter.txt",
        path=Path("var/safety/mols_to_filter.txt"),
        description="ReaSyn dangerous-molecule watchlist for the toxicity screen.",
    ),
)


def list_public_data_files() -> list[dict[str, str]]:
    return [
        {
            "name": item.name,
            "url": item.url,
            "path": str(item.path),
            "description": item.description,
        }
        for item in PUBLIC_DATA_FILES
    ]


def download_public_data(
    *,
    dest_root: str | Path = ".",
    names: list[str] | None = None,
    force: bool = False,
    timeout: float = 60.0,
) -> dict[str, Any]:
    selected = _select_files(names)
    root = Path(dest_root)
    files: list[dict[str, Any]] = []
    downloaded = 0
    skipped = 0

    for item in selected:
        destination = root / item.path
        if destination.exists() and not force:
            skipped += 1
            files.append(_file_result(item, destination, status="skipped"))
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(item.url, timeout=timeout) as response:
            data = response.read()
        destination.write_bytes(data)
        downloaded += 1
        files.append(_file_result(item, destination, status="downloaded"))

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "files": files,
    }


def _select_files(names: list[str] | None) -> list[PublicDataFile]:
    if not names:
        return list(PUBLIC_DATA_FILES)
    by_name = {item.name: item for item in PUBLIC_DATA_FILES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise RuntimeError(f"Unknown public data file(s): {', '.join(unknown)}")
    return [by_name[name] for name in names]


def _file_result(item: PublicDataFile, destination: Path, *, status: str) -> dict[str, Any]:
    size_bytes = destination.stat().st_size if destination.exists() else 0
    digest = hashlib.sha256(destination.read_bytes()).hexdigest() if destination.exists() else None
    line_count = _count_lines(destination) if destination.exists() else None
    return {
        "name": item.name,
        "status": status,
        "path": str(destination),
        "url": item.url,
        "size_bytes": size_bytes,
        "sha256": digest,
        "line_count": line_count,
    }


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)
