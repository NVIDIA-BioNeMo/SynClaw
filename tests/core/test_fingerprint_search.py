# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import DataStructs

from synclaw.core import chem, db, fingerprint_index
from synclaw.tools.synthesis import SynthesisTools

CATALOG = [
    "CCO",
    "CCCO",
    "CCOC",
    "OCCO",
    "CCCCO",
    "COCCO",
    "CCN",
    "CCCl",
    "c1ccccc1O",
    "c1ccccc1N",
    "Cc1ccccc1O",
    "OC(=O)c1ccccc1",
    "CC(C)O",
    "CC(=O)O",
    "NCCO",
    "C1CCOC1",
]


def test_threshold_query_returns_every_match_without_top_k_or_tie_caps() -> None:
    count = fingerprint_index.MAX_TIED_RESULTS + 17
    packed = fingerprint_index.pack_fingerprint(chem.constitutional_morgan_fingerprint("CCO"))
    index = fingerprint_index.FingerprintIndex(
        ids=np.arange(1, count + 1),
        fingerprints=np.tile(packed, (count, 1)),
        popcounts=np.full(count, np.bitwise_count(packed).sum()),
    )
    query = chem.constitutional_morgan_fingerprint("CCO")
    assert len(index.top_k(query, 1, min_similarity=1.0)) == fingerprint_index.MAX_TIED_RESULTS
    hits = index.top_k(query, None, min_similarity=1.0, exclude_ids={1})
    assert {bb_id for bb_id, _ in hits} == set(range(2, count + 1))


def test_threshold_query_handles_empty_fingerprints() -> None:
    fingerprints = np.zeros((2, fingerprint_index.WORDS), dtype=np.uint64)
    fingerprints[1, 0] = 1
    index = fingerprint_index.FingerprintIndex(
        ids=np.array([1, 2]),
        fingerprints=fingerprints,
        popcounts=np.array([0, 1]),
    )
    empty = DataStructs.ExplicitBitVect(chem.MORGAN_N_BITS)
    assert index.top_k(empty, None, min_similarity=1) == [(1, 1.0)]
    assert index.top_k(empty, None, min_similarity=0) == [(1, 1.0), (2, 0.0)]


def test_similarity_filter_is_exhaustive_inclusive_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = "C[C@H](O)CC"
    catalog = [target, "C[C@@H](O)CC", target + ".[Na+]", "CO", "c1ccccc1"]
    db_path = _prepared(tmp_path, catalog)
    cutoff = chem.similarity(target, catalog[2])["constitutional_similarity"]
    expected = {
        i
        for i, smiles in enumerate(catalog, 1)
        if chem.similarity(target, smiles)["constitutional_similarity"] >= cutoff
    } | {5}
    original = fingerprint_index.FingerprintIndex.top_k
    scans = []

    def tracked(self, query, k, **kwargs):
        scans.append(k)
        return original(self, query, k, **kwargs)

    monkeypatch.setattr(fingerprint_index.FingerprintIndex, "top_k", tracked)
    tools = SynthesisTools(
        db_path,
        hidden_target_smiles=target,
        hidden_similarity=cutoff,
        hidden_building_block_ids=frozenset({5}),
    )
    assert tools.hidden_building_block_ids == expected
    assert {row["id"] for row in tools.search_building_blocks("*")["results"]} == set(
        range(1, 6)
    ) - expected
    for bb_id in expected:
        assert not tools.get_building_block(bb_id)["found"]
        assert not tools.submit_pathway(building_block=f"BB:{bb_id}")["accepted"]
        assert not tools.submit_pathway(building_block=catalog[bb_id - 1])["accepted"]
    assert scans == [None]
    assert SynthesisTools(
        db_path, hidden_target_smiles=target, hidden_similarity=1
    ).hidden_building_block_ids == {1, 2}


@pytest.mark.parametrize("cutoff", [-0.1, 1.1, float("nan"), float("inf")])
def test_similarity_filter_rejects_invalid_cutoffs(tmp_path: Path, cutoff: float) -> None:
    with pytest.raises(RuntimeError, match="cutoff"):
        SynthesisTools(
            tmp_path / "unused.sqlite", hidden_target_smiles="CO", hidden_similarity=cutoff
        )


def test_similarity_filter_requires_a_target_and_supports_zero(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="target"):
        SynthesisTools(tmp_path / "unused.sqlite", hidden_similarity=0.95)
    tools = SynthesisTools(
        _prepared(tmp_path, CATALOG), hidden_target_smiles="CO", hidden_similarity=0
    )
    assert tools.hidden_building_block_ids == set(range(1, len(CATALOG) + 1))


def _prepared(tmp_path: Path, smiles: list[str]) -> Path:
    db_path = tmp_path / "test.sqlite"
    db.ingest_building_blocks(_write(tmp_path / "bbs.txt", smiles), db_path)
    db.prepare_building_block_search(db_path)
    return db_path


def _write(path: Path, smiles: list[str]) -> Path:
    path.write_text("SMILES\n" + "\n".join(smiles) + "\n", encoding="utf-8")
    return path


def test_search_returns_the_exhaustive_tanimoto_top_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Several stored chunks, so the index is assembled across chunk boundaries.
    monkeypatch.setattr(fingerprint_index, "STORED_CHUNK_ROWS", 3)
    db_path = _prepared(tmp_path, CATALOG)
    query = "CCCCl"
    reference = chem.bulk_tanimoto(
        chem.constitutional_morgan_fingerprint(query),
        [chem.constitutional_morgan_fingerprint(smiles) for smiles in CATALOG],
    )
    expected = sorted(score for score in reference if score >= 0.15)[::-1][:5]

    # An achiral query scores the same with or without stereo-awareness.
    results = SynthesisTools(db_path).search_building_blocks(query, limit=5)["results"]

    assert [row["match_type"] for row in results] == ["near"] * len(expected)
    assert [row["similarity"] for row in results] == pytest.approx(expected)
    assert [row["constitutional_similarity"] for row in results] == pytest.approx(expected)


def test_exact_hit_comes_first_once_for_any_spelling(tmp_path: Path) -> None:
    tools = SynthesisTools(_prepared(tmp_path, CATALOG))

    forward = tools.search_building_blocks("CCO", limit=3)["results"]
    reversed_spelling = tools.search_building_blocks("OCC", limit=3)["results"]

    assert [row["id"] for row in forward] == [row["id"] for row in reversed_spelling]
    assert [row["match_type"] for row in forward] == ["exact", "near", "near"]
    assert forward[0]["id"] not in [row["id"] for row in forward[1:]]
    # A SMILES-substring search would fill these slots with hits such as CCOC.
    assert "CCOC" not in [row["canonical_smiles"] for row in forward]


def test_search_ranks_the_matching_stereoisomer_first(tmp_path: Path) -> None:
    db_path = _prepared(tmp_path, ["C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O", "CC(N)C(=O)O"])

    results = SynthesisTools(db_path).search_building_blocks("C[C@H](N)C(=O)OC", limit=3)["results"]

    # Constitutionally all three tie, so only the stereo-aware score separates them.
    assert len({row["constitutional_similarity"] for row in results}) == 1
    assert (
        results[0]["canonical_smiles"]
        == chem.validate_smiles("C[C@H](N)C(=O)O")["canonical_smiles"]
    )
    assert results[0]["similarity"] > results[-1]["similarity"]


def test_invalid_smiles_query_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="valid SMILES"):
        SynthesisTools(_prepared(tmp_path, CATALOG)).search_building_blocks("CC(", limit=5)


def test_a_running_process_sees_blocks_added_by_a_new_prepare(tmp_path: Path) -> None:
    db_path = _prepared(tmp_path, CATALOG[:4])
    tools = SynthesisTools(db_path)
    assert (
        tools.search_building_blocks("CCCCC", limit=1)["results"][0]["canonical_smiles"] != "CCCCCO"
    )

    db.ingest_building_blocks(_write(tmp_path / "more.txt", ["CCCCCO"]), db_path)
    db.prepare_building_block_search(db_path)

    assert (
        tools.search_building_blocks("CCCCC", limit=1)["results"][0]["canonical_smiles"] == "CCCCCO"
    )


def test_a_staged_copy_is_searchable_on_its_own(tmp_path: Path) -> None:
    staged = db.stage_prepared_search_database(
        _prepared(tmp_path, CATALOG), destination_directory=tmp_path / "stage"
    )

    assert (
        SynthesisTools(staged).search_building_blocks("CCCCl", limit=1)["results"][0]["match_type"]
        == "near"
    )


def test_tools_for_a_missing_database_do_not_create_it(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "catalog.sqlite"

    SynthesisTools(missing, _initialize_database=False)

    assert not missing.parent.exists()


def test_preloading_an_unprepared_catalog_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    db.ingest_building_blocks(_write(tmp_path / "bbs.txt", CATALOG), db_path)

    with db.session(db_path) as conn, pytest.raises(RuntimeError, match="prepare-search"):
        db.preload_building_block_search(conn)
