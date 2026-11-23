# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Downloadable public data assets for local examples."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synclaw.core.toxicity import DEFAULT_WATCHLIST


@dataclass(frozen=True)
class PublicDataFile:
    """One downloadable file, pinned to an upstream commit and its expected SHA-256."""

    name: str
    url: str
    path: Path
    description: str
    sha256: str


# Neither upstream repository publishes tags or releases, so each URL is pinned to the last
# commit that changed the file; the SHA-256 is checked on every download.
_SYNFORMER = "https://raw.githubusercontent.com/wenhao-gao/synformer"
_REASYN = "https://raw.githubusercontent.com/NVIDIA-BioNeMo/ReaSyn"

PUBLIC_DATA_FILES: tuple[PublicDataFile, ...] = (
    PublicDataFile(
        name="synformer-chembl-1k",
        url=f"{_SYNFORMER}/bef02bc4481adf3ed792090fc6169331d4d0553a/data/chembl_filtered_1k.txt",
        path=Path("data/synformer_test_sets/chembl_filtered_1k.txt"),
        description="SynFormer public ChEMBL 1k target set.",
        sha256="3319bebde9ca9fa0529c6b127a9d7c1b91388d4254cc52de98ec1cfa12d3138e",
    ),
    PublicDataFile(
        name="reasyn-mols-to-filter",
        url=f"{_REASYN}/7c2384798fc2385634331d06d47a6164065c65a4/data/mols_to_filter.txt",
        path=DEFAULT_WATCHLIST,
        description="ReaSyn dangerous-molecule watchlist for the toxicity screen.",
        sha256="f112aba85b7c6031f03de5d190bbc4cab13898044a2b33638f0531e97fc70da8",
    ),
)


def list_public_data_files() -> list[dict[str, str]]:
    return [
        {
            "name": item.name,
            "url": item.url,
            "path": str(item.path),
            "description": item.description,
            "sha256": item.sha256,
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
        # An existing file is kept only if it is the pinned one; a partial or older copy is fetched again.
        if not force and destination.exists() and _sha256(destination.read_bytes()) == item.sha256:
            skipped += 1
            files.append(_file_result(item, destination, status="skipped"))
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(item.url, timeout=timeout) as response:
            data = response.read()
        if (digest := _sha256(data)) != item.sha256:
            raise RuntimeError(
                f"{item.name} from {item.url} has SHA-256 {digest}, not the pinned {item.sha256}"
            )
        # Written aside and moved into place, so an interrupted download leaves no partial file
        # that a later run would skip as already downloaded.
        temporary = destination.with_name(f".{destination.name}.download")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
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
    return [by_name[name] for name in dict.fromkeys(names)]


def _file_result(item: PublicDataFile, destination: Path, *, status: str) -> dict[str, Any]:
    data = destination.read_bytes()
    return {
        "name": item.name,
        "status": status,
        "path": str(destination),
        "url": item.url,
        "size_bytes": len(data),
        "sha256": _sha256(data),
        "line_count": len(data.splitlines()),
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
