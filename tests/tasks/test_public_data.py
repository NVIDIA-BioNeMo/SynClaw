# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Downloading the public example data."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import pytest

from synclaw.core.toxicity import DEFAULT_WATCHLIST
from synclaw.tasks import public_data


def test_the_watchlist_downloads_where_the_toxicity_screen_reads_it() -> None:
    by_name = {item.name: item for item in public_data.PUBLIC_DATA_FILES}

    assert by_name["reasyn-mols-to-filter"].path == DEFAULT_WATCHLIST


def _pin(monkeypatch: pytest.MonkeyPatch, name: str, content: bytes) -> None:
    pinned = tuple(
        public_data.PublicDataFile(
            **{**item.__dict__, "sha256": hashlib.sha256(content).hexdigest()}
        )
        if item.name == name
        else item
        for item in public_data.PUBLIC_DATA_FILES
    )
    monkeypatch.setattr(public_data, "PUBLIC_DATA_FILES", pinned)


def test_the_downloads_are_the_chembl_targets_and_the_watchlist() -> None:
    names = {item["name"] for item in public_data.list_public_data_files()}

    assert names == {"synformer-chembl-1k", "reasyn-mols-to-filter"}


def test_every_download_is_pinned_to_a_commit() -> None:
    for item in public_data.PUBLIC_DATA_FILES:
        commit = item.url.split("/")[5]
        assert len(commit) == 40 and int(commit, 16) >= 0, item.name
        assert len(item.sha256) == 64, item.name


def test_a_repeated_name_downloads_once_and_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _pin(monkeypatch, "synformer-chembl-1k", b"CCO\nCCN\n")

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


def test_a_download_that_does_not_match_its_pin_is_refused(tmp_path: Path) -> None:
    with (
        patch(
            "synclaw.tasks.public_data.urllib.request.urlopen",
            return_value=io.BytesIO(b"changed upstream\n"),
        ),
        pytest.raises(RuntimeError, match="not the pinned"),
    ):
        public_data.download_public_data(dest_root=tmp_path, names=["synformer-chembl-1k"])

    assert not (tmp_path / "data/synformer_test_sets/chembl_filtered_1k.txt").exists()


def test_an_existing_file_that_differs_from_the_pin_is_reported_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, "synformer-chembl-1k", b"CCO\n")
    target = tmp_path / "data/synformer_test_sets/chembl_filtered_1k.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"CCO\nlocal edit\n")

    kept = public_data.download_public_data(dest_root=tmp_path, names=["synformer-chembl-1k"])
    with patch(
        "synclaw.tasks.public_data.urllib.request.urlopen", return_value=io.BytesIO(b"CCO\n")
    ):
        forced = public_data.download_public_data(
            dest_root=tmp_path, names=["synformer-chembl-1k"], force=True
        )

    assert kept["files"][0]["status"] == "modified"
    assert forced["files"][0]["status"] == "downloaded"
    assert target.read_bytes() == b"CCO\n"
