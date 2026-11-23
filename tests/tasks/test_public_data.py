# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Downloading the public example data."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

from synclaw.core.toxicity import DEFAULT_WATCHLIST
from synclaw.tasks import public_data


def test_the_watchlist_downloads_where_the_toxicity_screen_reads_it() -> None:
    by_name = {item.name: item for item in public_data.PUBLIC_DATA_FILES}

    assert by_name["reasyn-mols-to-filter"].path == DEFAULT_WATCHLIST


def test_a_repeated_name_downloads_once_and_leaves_no_partial_file(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(b"CCO\nCCN\n")

    with patch("synclaw.tasks.public_data.urllib.request.urlopen", side_effect=fake_urlopen):
        result = public_data.download_public_data(
            dest_root=tmp_path, names=["synformer-chembl-1k", "synformer-chembl-1k"]
        )

    assert len(calls) == 1
    assert result["downloaded"] == 1
    (entry,) = result["files"]
    assert (entry["size_bytes"], entry["line_count"]) == (8, 2)
    written = tmp_path / "data/synformer_test_sets/chembl_filtered_1k.txt"
    assert written.read_bytes() == b"CCO\nCCN\n"
    assert not list(written.parent.glob(".*.download"))
