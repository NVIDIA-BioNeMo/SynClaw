# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive constitutional Morgan/Tanimoto search over the building-block catalog.

``prepare-search`` stores the catalog's fingerprints in the database as packed
chunks. Each process reads them once into a read-only array and scores every
building block per query. A parent that loads the index before forking its tool
workers shares that one copy with all of them.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import DataStructs

from synclaw.core import chem

FINGERPRINT_CHUNK_TABLE = "building_block_search_fingerprint_chunks"
WORDS = chem.MORGAN_N_BITS // 64
# Building blocks per stored chunk (16 MiB of packed fingerprints).
STORED_CHUNK_ROWS = 1 << 16
# Rows per scoring chunk; keeps the AND/popcount temporaries small.
SCAN_CHUNK_ROWS = 1 << 16
# Ties at the k-th score are all kept so callers can break them; this caps that set.
MAX_TIED_RESULTS = 1000

_LOCK = threading.Lock()
# One index per database path, replaced when the catalog is prepared again.
_LOADED: dict[str, tuple[str, FingerprintIndex]] = {}


@dataclass(frozen=True, slots=True)
class FingerprintIndex:
    """Building-block ids, packed fingerprints and their popcounts, in id order."""

    ids: np.ndarray
    fingerprints: np.ndarray
    popcounts: np.ndarray

    def top_k(
        self,
        query_fingerprint: Any,
        k: int | None,
        *,
        min_similarity: float,
        exclude_ids: set[int] | frozenset[int] = frozenset(),
    ) -> list[tuple[int, float]]:
        """Return top-k matches; k=None returns every threshold match without the tie cap."""

        query = pack_fingerprint(query_fingerprint)
        query_count = int(np.bitwise_count(query).sum())
        if (k is not None and (k <= 0 or query_count == 0)) or not len(self.ids):
            return []
        intersections = np.empty(len(self.ids), dtype=np.int32)
        for start in range(0, len(self.ids), SCAN_CHUNK_ROWS):
            block = self.fingerprints[start:start + SCAN_CHUNK_ROWS]
            intersections[start:start + len(block)] = np.bitwise_count(block & query).sum(axis=1, dtype=np.int32)
        unions = query_count + self.popcounts - intersections
        similarities = np.divide(intersections, unions, out=np.ones(len(self.ids)), where=unions != 0)
        if exclude_ids:
            excluded = np.fromiter(exclude_ids, dtype=np.int64)
            positions = np.searchsorted(self.ids, excluded)
            positions = positions[positions < len(self.ids)]
            positions = positions[np.isin(self.ids[positions], excluded)]
            similarities[positions] = -1.0
        candidates = np.flatnonzero(similarities >= min_similarity)
        if k is not None and len(candidates) > k:
            kth = np.partition(similarities[candidates], -k)[-k]
            candidates = candidates[similarities[candidates] >= kth]
        if k is not None and len(candidates) > MAX_TIED_RESULTS:
            # Candidates are in id order, so a stable sort keeps the lowest ids among ties.
            order = np.argsort(-similarities[candidates], kind="stable")
            candidates = candidates[order[:MAX_TIED_RESULTS]]
        return [(int(self.ids[position]), float(similarities[position])) for position in candidates]


def pack_fingerprint(fingerprint: Any) -> np.ndarray:
    """Pack an RDKit bit vector into the index's uint64 word layout."""

    return np.frombuffer(bytes.fromhex(DataStructs.BitVectToFPSText(fingerprint)), dtype=np.uint64)


def write_fingerprint_chunks(conn: sqlite3.Connection, rows: Iterable[tuple[int, str]]) -> None:
    """Store constitutional fingerprints for ``(id, canonical_smiles)`` rows given in id order."""

    chunk = 0
    ids: list[int] = []
    hex_fingerprints: list[str] = []

    def flush() -> None:
        nonlocal chunk
        conn.execute(
            f"INSERT INTO {FINGERPRINT_CHUNK_TABLE}(chunk, row_count, ids, fingerprints) VALUES (?, ?, ?, ?)",
            (chunk, len(ids), np.asarray(ids, dtype="<i8").tobytes(), bytes.fromhex("".join(hex_fingerprints))),
        )
        chunk += 1
        ids.clear()
        hex_fingerprints.clear()

    for building_block_id, canonical_smiles in rows:
        ids.append(int(building_block_id))
        hex_fingerprints.append(
            DataStructs.BitVectToFPSText(chem.constitutional_morgan_fingerprint(str(canonical_smiles)))
        )
        if len(ids) == STORED_CHUNK_ROWS:
            flush()
    if ids:
        flush()


def stored_fingerprint_count(conn: sqlite3.Connection) -> int:
    (count,) = conn.execute(f"SELECT COALESCE(SUM(row_count), 0) FROM {FINGERPRINT_CHUNK_TABLE}").fetchone()
    return int(count)


def load_fingerprint_index(conn: sqlite3.Connection, db_path: str | Path) -> FingerprintIndex:
    """Return this process's index for a prepared DB, reading it on first use."""

    database = str(Path(db_path).resolve())
    key = _catalog_key(conn)
    with _LOCK:
        loaded = _LOADED.get(database)
        if loaded is None or loaded[0] != key:
            loaded = (key, _read_index(conn))
            _LOADED[database] = loaded
    return loaded[1]


def _catalog_key(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        """
        SELECT index_version, building_block_count, maximum_building_block_id,
               fingerprint_radius, fingerprint_bits, rdkit_version, prepared_at
        FROM building_block_search_metadata WHERE singleton = 1
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("Building-block fingerprint search needs a prepared search index")
    return json.dumps(tuple(row))


def _read_index(conn: sqlite3.Connection) -> FingerprintIndex:
    count = stored_fingerprint_count(conn)
    ids = np.empty(count, dtype=np.int64)
    fingerprints = np.empty((count, WORDS), dtype=np.uint64)
    position = 0
    for row_count, id_bytes, fingerprint_bytes in conn.execute(
        f"SELECT row_count, ids, fingerprints FROM {FINGERPRINT_CHUNK_TABLE} ORDER BY chunk"
    ):
        ids[position:position + row_count] = np.frombuffer(id_bytes, dtype="<i8")
        fingerprints[position:position + row_count] = np.frombuffer(fingerprint_bytes, dtype=np.uint64).reshape(
            row_count, WORDS
        )
        position += row_count
    popcounts = np.empty(count, dtype=np.int32)
    for start in range(0, count, SCAN_CHUNK_ROWS):
        block = fingerprints[start:start + SCAN_CHUNK_ROWS]
        popcounts[start:start + len(block)] = np.bitwise_count(block).sum(axis=1, dtype=np.int32)
    # Read-only, so forked workers that inherit the arrays keep sharing their pages.
    for array in (ids, fingerprints, popcounts):
        array.flags.writeable = False
    return FingerprintIndex(ids=ids, fingerprints=fingerprints, popcounts=popcounts)
