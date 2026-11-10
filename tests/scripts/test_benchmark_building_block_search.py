# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys

from synclaw.core import db
from synclaw.paths import REPO_ROOT


def _load_benchmark():
    script = REPO_ROOT / "scripts" / "benchmark_building_block_search.py"
    spec = importlib.util.spec_from_file_location("benchmark_building_block_search", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _database_with_ids(tmp_path, ids: list[int]):
    path = tmp_path / "bbs.sqlite"
    db.init_db(path)
    with db.session(path) as conn:
        for position, identifier in enumerate(ids):
            # Distinct because canonical_smiles is UNIQUE, and short: size must
            # track the row count, not the id, or a sparse id builds a huge molecule.
            smiles = "C" * (1 + position)
            conn.execute(
                "INSERT INTO building_blocks"
                " (id, smiles, canonical_smiles, heavy_atom_count, created_at)"
                " VALUES (?, ?, ?, ?, datetime('now'))",
                (identifier, smiles, smiles, len(smiles)),
            )
        conn.commit()
    db.prepare_building_block_search(path)
    return path


def test_sampling_terminates_on_a_sparse_id_space(tmp_path) -> None:
    # Two rows a million apart. Probing random ids and snapping upward would need
    # to land on exactly 1_000_000 to reach the second row.
    benchmark = _load_benchmark()
    path = _database_with_ids(tmp_path, [1, 1_000_000])

    queries = benchmark._sample_database(path, count=2, seed=0)

    assert len(queries) == 2
    assert all(isinstance(q, str) and q for q in queries)


def test_sampling_is_capped_by_row_count_and_deterministic(tmp_path) -> None:
    benchmark = _load_benchmark()
    path = _database_with_ids(tmp_path, list(range(1, 21)))

    assert len(benchmark._sample_database(path, count=100, seed=1)) == 20
    assert benchmark._sample_database(path, count=5, seed=7) == benchmark._sample_database(
        path, count=5, seed=7
    )
