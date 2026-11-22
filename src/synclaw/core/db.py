# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite storage for building blocks, reaction templates and prepared building-block search."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from synclaw.core import chem, fingerprint_index
from synclaw.core.libraries import (
    BUNDLED_REACTION_SOURCE,
    LibrarySelection,
    bundled_reaction_library_versions,
    normalize_libraries,
)

# Search tuning.
# Constitutional near-match hits that the stereo-aware score re-ranks.
STEREO_RERANK_POOL = 200
# Score shift for a matched search term (+) or exclusion (-), as a fraction of the best text match.
REACTION_TEXT_SEARCH_CURATED_BONUS = 0.2
# English function words that appear in most template descriptions and would otherwise
# let any conversational query match nearly the whole catalog.
REACTION_TEXT_SEARCH_STOPWORDS = frozenset(
    {
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "not",
        "of",
        "on",
        "or",
        "than",
        "that",
        "the",
        "then",
        "this",
        "to",
        "via",
        "with",
        "without",
    }
)
BUILDING_BLOCK_FINGERPRINT_TABLE = fingerprint_index.FINGERPRINT_CHUNK_TABLE
BUILDING_BLOCK_SEARCH_METADATA_TABLE = "building_block_search_metadata"

BUILDING_BLOCK_SEARCH_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {BUILDING_BLOCK_SEARCH_METADATA_TABLE} (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    building_block_count INTEGER NOT NULL,
    maximum_building_block_id INTEGER NOT NULL,
    fingerprint_radius INTEGER NOT NULL,
    fingerprint_bits INTEGER NOT NULL,
    rdkit_version TEXT NOT NULL,
    prepared INTEGER NOT NULL CHECK (prepared IN (0, 1)),
    prepared_at TEXT NOT NULL
);

-- Packed constitutional fingerprints, STORED_CHUNK_ROWS building blocks per row in id order:
-- ids as little-endian int64, fingerprints as MORGAN_N_BITS-bit vectors of uint64 words.
CREATE TABLE IF NOT EXISTS {BUILDING_BLOCK_FINGERPRINT_TABLE} (
    chunk INTEGER PRIMARY KEY,
    row_count INTEGER NOT NULL,
    ids BLOB NOT NULL,
    fingerprints BLOB NOT NULL
);

CREATE TRIGGER IF NOT EXISTS building_block_search_stale_insert
AFTER INSERT ON building_blocks BEGIN
    UPDATE {BUILDING_BLOCK_SEARCH_METADATA_TABLE} SET prepared = 0 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS building_block_search_stale_update
AFTER UPDATE ON building_blocks BEGIN
    UPDATE {BUILDING_BLOCK_SEARCH_METADATA_TABLE} SET prepared = 0 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS building_block_search_stale_delete
AFTER DELETE ON building_blocks BEGIN
    UPDATE {BUILDING_BLOCK_SEARCH_METADATA_TABLE} SET prepared = 0 WHERE singleton = 1;
END;
"""


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS building_blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    smiles TEXT NOT NULL,
    canonical_smiles TEXT NOT NULL UNIQUE,
    inchi_key TEXT,
    heavy_atom_count INTEGER,
    source_file TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_building_blocks_inchi_key ON building_blocks(inchi_key);

CREATE TABLE IF NOT EXISTS import_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    raw_value TEXT NOT NULL,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reaction_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reaction_id TEXT NOT NULL,
    library TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    smarts TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(reaction_id)
);

CREATE INDEX IF NOT EXISTS idx_reaction_templates_library ON reaction_templates(library);

CREATE TABLE IF NOT EXISTS reaction_library_catalog (
    library TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    catalog_sha256 TEXT NOT NULL,
    template_count INTEGER NOT NULL,
    installed_at TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open an existing database; only ``init_db`` creates one."""

    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SQLite database does not exist: {path}. Create it with `synclaw db init --db {path}` "
            "or an ingest command."
        )
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    """Create the database file if it is missing and apply the schema."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA)


def prepare_building_block_search(
    db_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the fingerprint search index used by model runs.

    When ``output_path`` is supplied, atomically create a copy containing only the
    chemistry catalog tables. Otherwise the database is prepared in place and then
    vacuumed, so the pages of the dropped index tables are returned to the disk.
    """

    source = Path(db_path).resolve()
    if not source.exists():
        raise RuntimeError(f"Cannot prepare search index: database does not exist: {source}")
    destination = Path(output_path).resolve() if output_path is not None else source
    if destination == source:
        _prepare_building_block_search_in_place(destination)
        with closing(connect(destination)) as conn:
            conn.execute("VACUUM")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            init_db(temporary)
            with session(temporary) as conn:
                conn.execute("ATTACH DATABASE ? AS source", (str(source),))
                for table in (
                    "building_blocks",
                    "reaction_templates",
                    "reaction_library_catalog",
                    "import_errors",
                ):
                    columns = [
                        str(row["name"])
                        for row in conn.execute(f"PRAGMA source.table_info({table})")
                    ]
                    if not columns:
                        continue
                    names = ", ".join(columns)
                    conn.execute(
                        f"INSERT INTO main.{table} ({names}) SELECT {names} FROM source.{table}"
                    )
            _prepare_building_block_search_in_place(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    with session(destination) as conn:
        metadata = _building_block_search_metadata(conn)
    return {
        "database": str(destination),
        "source_database": str(source),
        "building_block_count": metadata["building_block_count"],
        "fingerprint_count": metadata["fingerprint_count"],
        "prepared_at": metadata["prepared_at"],
    }


def _prepare_building_block_search_in_place(db_path: Path) -> None:
    init_db(db_path)
    with session(db_path) as conn:
        conn.executescript(
            f"""
            DROP TRIGGER IF EXISTS building_block_search_stale_insert;
            DROP TRIGGER IF EXISTS building_block_search_stale_update;
            DROP TRIGGER IF EXISTS building_block_search_stale_delete;
            DROP TABLE IF EXISTS {BUILDING_BLOCK_FINGERPRINT_TABLE};
            DROP TABLE IF EXISTS {BUILDING_BLOCK_SEARCH_METADATA_TABLE};
            """
        )
        conn.executescript(BUILDING_BLOCK_SEARCH_SCHEMA)
        fingerprint_index.write_fingerprint_chunks(
            conn, conn.execute("SELECT id, canonical_smiles FROM building_blocks ORDER BY id")
        )
        count, maximum_id = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM building_blocks"
        ).fetchone()
        conn.execute(
            f"""
            INSERT INTO {BUILDING_BLOCK_SEARCH_METADATA_TABLE}(
                singleton, building_block_count,
                maximum_building_block_id, fingerprint_radius,
                fingerprint_bits, rdkit_version, prepared, prepared_at
            ) VALUES (1, ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(singleton) DO UPDATE SET
                building_block_count=excluded.building_block_count,
                maximum_building_block_id=excluded.maximum_building_block_id,
                fingerprint_radius=excluded.fingerprint_radius,
                fingerprint_bits=excluded.fingerprint_bits,
                rdkit_version=excluded.rdkit_version,
                prepared=1,
                prepared_at=excluded.prepared_at
            """,
            (
                int(count),
                int(maximum_id),
                chem.MORGAN_RADIUS,
                chem.MORGAN_N_BITS,
                chem.RDKIT_VERSION,
                now_iso(),
            ),
        )


def require_reaction_templates(conn: sqlite3.Connection) -> None:
    """Fail loudly unless the database's reaction templates are present, current and numbered.

    Without templates the family overview is empty and every template search
    returns zero results with ``valid: true``.
    """

    libraries = _available_reaction_libraries(conn)
    if not libraries:
        raise RuntimeError(
            "This database has no reaction templates. Import the bundled sources with "
            "`synclaw ingest-reactions --db <path>` from a source checkout, or a custom file with "
            "`synclaw ingest-reactions <templates> --library <name> --db <path>`."
        )
    _require_current_reaction_libraries(conn, libraries)
    if any(stored != numbered for stored, numbered, _ in _reaction_template_numbering(conn)):
        raise RuntimeError(
            "This database's reaction template IDs are not numbered R001 onward. "
            "Re-run `synclaw ingest-reactions --db <path>` to renumber them; it reinstalls every "
            "bundled source, so then drop any you do not use with `synclaw db remove-reactions`."
        )


def _reaction_template_numbering(conn: sqlite3.Connection) -> list[tuple[str, str, int]]:
    """Pair each template's stored ID with its R001…RN ID and row id.

    Templates are ordered by a hash of their SMARTS, so neither IDs nor ID-order
    tie-breaks group templates by library.
    """

    rows = conn.execute("SELECT id, reaction_id, smarts FROM reaction_templates").fetchall()
    rows.sort(key=lambda row: (hashlib.sha256(row["smarts"].encode()).hexdigest(), row["id"]))
    width = max(3, len(str(len(rows))))
    return [
        (row["reaction_id"], f"R{number:0{width}d}", row["id"])
        for number, row in enumerate(rows, start=1)
    ]


def number_reaction_templates(conn: sqlite3.Connection) -> None:
    """Renumber every template R001…RN across all libraries in the database."""

    numbering = _reaction_template_numbering(conn)
    # Park every row on a unique placeholder first so no new ID collides with an old one.
    conn.executemany(
        "UPDATE reaction_templates SET reaction_id = ? WHERE id = ?",
        [(f"#{row_id}", row_id) for *_, row_id in numbering],
    )
    conn.executemany(
        "UPDATE reaction_templates SET reaction_id = ? WHERE id = ?",
        [(numbered, row_id) for _, numbered, row_id in numbering],
    )


def _available_reaction_libraries(conn: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[0])
        for row in conn.execute("SELECT DISTINCT library FROM reaction_templates ORDER BY 1")
    )


def _require_current_reaction_libraries(
    conn: sqlite3.Connection, libraries: tuple[str, ...]
) -> None:
    catalog = reaction_library_catalog(conn)
    untracked = [name for name in libraries if name not in catalog]
    if untracked:
        raise RuntimeError(
            f"Reaction library {', '.join(repr(name) for name in untracked)} has no import record in this "
            "database. Re-import it with `synclaw ingest-reactions --db <path>` "
            "(bundled libraries) or `synclaw ingest-reactions <templates> --library <name> --db <path>`, "
            "or drop libraries you no longer use with `synclaw db remove-reactions <name> --db <path>`."
        )
    expected = bundled_reaction_library_versions()
    bundled = [name for name in libraries if catalog[name]["source"] == BUNDLED_REACTION_SOURCE]
    retired = [name for name in bundled if name not in expected]
    if retired:
        raise RuntimeError(
            f"Reaction library {', '.join(repr(name) for name in retired)} was imported as a bundled source "
            "that this version no longer ships. Remove it with "
            "`synclaw db remove-reactions <name> --db <path>`, or rebuild the prepared database."
        )
    stale = [name for name in bundled if catalog[name]["catalog_sha256"] != expected[name]]
    if stale:
        raise RuntimeError(
            f"Bundled reaction library {', '.join(repr(name) for name in stale)} is out of date with this "
            "installation's template catalog. Run `synclaw ingest-reactions --db <path>` from a source "
            "checkout of this version, or rebuild the prepared database, to replace it."
        )


def reaction_library_catalog(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Return import records by library."""

    rows = conn.execute("SELECT * FROM reaction_library_catalog").fetchall()
    return {str(row["library"]): dict(row) for row in rows}


def replace_reaction_library(
    conn: sqlite3.Connection,
    library: str,
    *,
    source: str,
    catalog_sha256: str,
    template_count: int,
) -> None:
    conn.execute(
        """
        INSERT INTO reaction_library_catalog(library, source, catalog_sha256, template_count, installed_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(library) DO UPDATE SET
            source = excluded.source,
            catalog_sha256 = excluded.catalog_sha256,
            template_count = excluded.template_count,
            installed_at = excluded.installed_at
        """,
        (library, source, catalog_sha256, template_count, now_iso()),
    )


def remove_reaction_libraries(conn: sqlite3.Connection, libraries: LibrarySelection) -> int:
    """Delete these libraries' templates and import records, then renumber the rest."""

    names = normalize_libraries(libraries)
    if not names:
        return 0
    placeholders = ",".join("?" for _ in names)
    removed = conn.execute(
        f"DELETE FROM reaction_templates WHERE library IN ({placeholders})", names
    ).rowcount
    conn.execute(f"DELETE FROM reaction_library_catalog WHERE library IN ({placeholders})", names)
    number_reaction_templates(conn)
    return removed


def require_prepared_building_block_search(
    db_path: str | Path,
) -> dict[str, Any]:
    """Fail clearly unless a current, complete prepared BB search index exists."""

    with session(db_path) as conn:
        return _require_prepared_building_block_search(
            conn,
            db_path=db_path,
            verify_catalog=True,
        )


def _require_prepared_building_block_search(
    conn: sqlite3.Connection,
    *,
    db_path: str | Path | None = None,
    verify_catalog: bool = False,
) -> dict[str, Any]:
    """Validate prepared-search state, optionally including full catalog scans."""

    display_path = db_path or _connection_database_path(conn)
    try:
        metadata = _building_block_search_metadata(
            conn,
            include_catalog_state=verify_catalog,
        )
    except sqlite3.OperationalError as exc:
        raise RuntimeError(_prepared_search_error(display_path)) from exc
    if (
        not metadata["prepared"]
        or metadata["fingerprint_radius"] != chem.MORGAN_RADIUS
        or metadata["fingerprint_bits"] != chem.MORGAN_N_BITS
        or metadata["rdkit_version"] != chem.RDKIT_VERSION
        or not _prepared_search_tables_exist(conn)
        or (
            verify_catalog
            and (
                metadata["building_block_count"] != metadata["current_building_block_count"]
                or metadata["maximum_building_block_id"]
                != metadata["current_maximum_building_block_id"]
                or metadata["fingerprint_count"] != metadata["current_building_block_count"]
            )
        )
    ):
        raise RuntimeError(_prepared_search_error(display_path))
    return metadata


def _connection_database_path(conn: sqlite3.Connection) -> str:
    for row in conn.execute("PRAGMA database_list"):
        if str(row[1]) == "main" and row[2]:
            return str(row[2])
    return "the active database"


def stage_prepared_search_database(
    db_path: str | Path,
    *,
    destination_directory: str | Path | None = None,
) -> Path:
    """Copy a validated prepared DB once to shared node-local storage."""

    import fcntl

    source = Path(db_path).resolve()
    require_prepared_building_block_search(source)
    directory = Path(
        destination_directory or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
    )
    directory.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    stat = source.stat()
    source_token = hashlib.sha256(
        f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:12]
    destination = directory / f"synclaw-search-{job_id}-{source_token}.sqlite"
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if destination.exists():
                try:
                    require_prepared_building_block_search(destination)
                    return destination
                except RuntimeError:
                    pass
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=directory,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                source_connection = connect(source)
                destination_connection = connect(temporary)
                try:
                    source_connection.backup(destination_connection)
                finally:
                    destination_connection.close()
                    source_connection.close()
                require_prepared_building_block_search(temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return destination


def _building_block_search_metadata(
    conn: sqlite3.Connection,
    *,
    include_catalog_state: bool = True,
) -> dict[str, Any]:
    row = conn.execute(
        f"SELECT * FROM {BUILDING_BLOCK_SEARCH_METADATA_TABLE} WHERE singleton = 1"
    ).fetchone()
    if row is None:
        raise sqlite3.OperationalError("prepared building-block search metadata is absent")
    metadata = dict(row)
    if not include_catalog_state:
        return metadata
    current_count, current_maximum_id = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM building_blocks"
    ).fetchone()
    return {
        **metadata,
        "current_building_block_count": int(current_count),
        "current_maximum_building_block_id": int(current_maximum_id),
        "fingerprint_count": fingerprint_index.stored_fingerprint_count(conn),
    }


def _prepared_search_error(db_path: str | Path) -> str:
    return (
        f"Building-block search index is missing, stale, or incompatible for {db_path}. "
        f"Run `synclaw db prepare-search --db {db_path}` after building the chemistry catalog."
    )


def ingest_building_blocks(
    source_path: str | Path,
    db_path: str | Path,
    batch_size: int = 1000,
) -> dict[str, int | str]:
    """Load a SMILES text file into SQLite.

    The expected file shape is one SMILES per line with an optional `SMILES`
    header. Invalid rows are recorded in `import_errors`.
    """

    init_db(db_path)
    source = Path(source_path)
    source_name = str(source)
    stats: dict[str, int | str] = {
        "source_file": source_name,
        "read": 0,
        "inserted": 0,
        "duplicates": 0,
        "invalid": 0,
    }
    rows: list[tuple[str, str, str | None, int | None, str, str]] = []
    errors: list[tuple[str, int, str, str, str]] = []

    with session(db_path) as conn, source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            value = raw_line.strip()
            if not value:
                continue
            if line_number == 1 and value.lower() == "smiles":
                continue
            stats["read"] = int(stats["read"]) + 1
            result = chem.validate_smiles(value)
            if not result["valid"]:
                stats["invalid"] = int(stats["invalid"]) + 1
                errors.append((source_name, line_number, value, str(result["error"]), now_iso()))
            else:
                props = result.get("properties", {})
                rows.append(
                    (
                        value,
                        str(result["canonical_smiles"]),
                        props.get("inchi_key"),
                        props.get("heavy_atom_count"),
                        source_name,
                        now_iso(),
                    )
                )

            if len(rows) >= batch_size:
                inserted, duplicates = _insert_building_block_rows(conn, rows)
                stats["inserted"] = int(stats["inserted"]) + inserted
                stats["duplicates"] = int(stats["duplicates"]) + duplicates
                rows.clear()
            if len(errors) >= batch_size:
                conn.executemany(
                    """
                    INSERT INTO import_errors(source_file, line_number, raw_value, error, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    errors,
                )
                errors.clear()

        if rows:
            inserted, duplicates = _insert_building_block_rows(conn, rows)
            stats["inserted"] = int(stats["inserted"]) + inserted
            stats["duplicates"] = int(stats["duplicates"]) + duplicates
        if errors:
            conn.executemany(
                """
                INSERT INTO import_errors(source_file, line_number, raw_value, error, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                errors,
            )
    return stats


def _insert_building_block_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, str | None, int | None, str, str]],
) -> tuple[int, int]:
    inserted = 0
    for row in rows:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO building_blocks(
                smiles, canonical_smiles, inchi_key, heavy_atom_count, source_file, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        inserted += cursor.rowcount
    return inserted, len(rows) - inserted


def search_building_blocks(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    include_near: bool = True,
    min_similarity: float = 0.15,
    *,
    hidden_ids: frozenset[int] = frozenset(),
) -> list[dict[str, Any]]:
    """Return the exact stock match for a SMILES query, then its nearest neighbours.

    Near matches are the constitutional Morgan/Tanimoto top hits over the whole
    catalog, re-ranked by the stereo-aware score, so a stocked stereoisomer of the
    query outranks its enantiomer. An empty query or ``*`` browses the catalog.
    Building blocks in ``hidden_ids`` are treated as out of stock.

    Raises:
        ValueError: The query is not a valid SMILES.
    """

    text = (query or "").strip()
    if not text or text == "*":
        cursor = conn.execute(
            """
            SELECT id, smiles, canonical_smiles, inchi_key, heavy_atom_count, source_file
            FROM building_blocks
            WHERE id NOT IN (SELECT value FROM json_each(?))
            ORDER BY id
            LIMIT ?
            """,
            (json.dumps(sorted(hidden_ids)), limit),
        )
        return [_with_match_metadata(dict(row), "browse", None) for row in cursor.fetchall()]

    validation = chem.validate_smiles(text)
    if not validation["valid"]:
        raise ValueError(
            f"Building-block search needs a valid SMILES query; {text!r} is not one ({validation['error']})."
        )
    canonical_smiles = str(validation["canonical_smiles"])
    _require_prepared_building_block_search(conn)
    exact = get_building_block_by_canonical_smiles(conn, canonical_smiles, hidden_ids=hidden_ids)
    results = [_with_match_metadata(exact, "exact", 1.0)] if exact else []
    if not include_near or len(results) >= limit:
        return results
    remaining = limit - len(results)
    index = fingerprint_index.load_fingerprint_index(conn, _main_database_path(conn))
    hits = index.top_k(
        chem.constitutional_morgan_fingerprint(canonical_smiles),
        max(remaining, STEREO_RERANK_POOL),
        min_similarity=min_similarity,
        exclude_ids=hidden_ids | ({int(exact["id"])} if exact else frozenset()),
    )
    if not hits:
        return results
    constitutional_scores = dict(hits)
    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT id, smiles, canonical_smiles, inchi_key, heavy_atom_count, source_file
            FROM building_blocks
            WHERE id IN (SELECT value FROM json_each(?))
            """,
            (json.dumps(list(constitutional_scores)),),
        )
    ]
    stereo_scores = chem.bulk_tanimoto(
        chem.stereo_aware_morgan_fingerprint(canonical_smiles),
        [chem.stereo_aware_morgan_fingerprint(row["canonical_smiles"]) for row in rows],
    )
    near = []
    for row, stereo_score in zip(rows, stereo_scores, strict=True):
        if stereo_score < min_similarity:
            continue
        match = _with_match_metadata(row, "near", stereo_score)
        match["constitutional_similarity"] = constitutional_scores[row["id"]]
        near.append(match)
    query_heavy_atom_count = validation.get("properties", {}).get("heavy_atom_count") or 0
    near.sort(
        key=lambda row: (
            -row["similarity"],
            -row["constitutional_similarity"],
            row["heavy_atom_count"] is None,
            abs((row["heavy_atom_count"] or 0) - query_heavy_atom_count),
            row["id"],
        )
    )
    return results + near[:remaining]


def preload_building_block_search(conn: sqlite3.Connection) -> None:
    """Load the fingerprint index now instead of at the first search.

    Call it in a host process before forking tool workers: they inherit the loaded
    index and share its memory instead of each reading their own copy.

    Raises:
        RuntimeError: The catalog's search index is missing or stale.
    """

    _require_prepared_building_block_search(conn)
    fingerprint_index.load_fingerprint_index(conn, _main_database_path(conn))


def _prepared_search_tables_exist(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (BUILDING_BLOCK_FINGERPRINT_TABLE,)
        ).fetchone()
        is not None
    )


def get_building_block(
    conn: sqlite3.Connection, bb_id: int, *, hidden_ids: frozenset[int] = frozenset()
) -> dict[str, Any] | None:
    # An ID outside SQLite's 64-bit INTEGER range cannot exist; binding it would raise OverflowError.
    if bb_id in hidden_ids or not -(2**63) <= bb_id < 2**63:
        return None
    cursor = conn.execute(
        """
        SELECT id, smiles, canonical_smiles, inchi_key, heavy_atom_count, source_file
        FROM building_blocks
        WHERE id = ?
        """,
        (bb_id,),
    )
    row = cursor.fetchone()
    return dict(row) if row is not None else None


def exact_building_block_matches(
    conn: sqlite3.Connection, smiles: str, *, hidden_ids: frozenset[int] = frozenset()
) -> list[dict[str, Any]]:
    """The stock entry with this molecule's canonical SMILES, shaped like a search hit."""

    validation = chem.validate_smiles(smiles)
    row = (
        get_building_block_by_canonical_smiles(
            conn, validation["canonical_smiles"], hidden_ids=hidden_ids
        )
        if validation["valid"]
        else None
    )
    return [_with_match_metadata(row, "exact", 1.0)] if row else []


def get_building_block_by_canonical_smiles(
    conn: sqlite3.Connection, canonical_smiles: str, *, hidden_ids: frozenset[int] = frozenset()
) -> dict[str, Any] | None:
    cursor = conn.execute(
        """
        SELECT id, smiles, canonical_smiles, inchi_key, heavy_atom_count, source_file
        FROM building_blocks
        WHERE canonical_smiles = ?
        """,
        (canonical_smiles,),
    )
    row = cursor.fetchone()
    return dict(row) if row is not None and row["id"] not in hidden_ids else None


def building_block_ids_matching_target(conn: sqlite3.Connection, smiles: str) -> frozenset[int]:
    """IDs of every stock entry that exact-match scoring would count as this target.

    That is the entry with the same canonical SMILES plus any entry that
    ``chem.matches_target`` accepts, such as a tautomer stored under different SMILES
    or one isomer of stereo the target leaves open. Those are found through the
    stereo-free first InChIKey block of each of the target's tautomers.

    Raises:
        ValueError: ``smiles`` is not a valid SMILES.
    """

    validation = chem.validate_smiles(smiles)
    if not validation["valid"]:
        raise ValueError(f"{smiles!r} is not a valid SMILES ({validation['error']}).")
    ids: set[int] = set()
    exact = get_building_block_by_canonical_smiles(conn, str(validation["canonical_smiles"]))
    if exact is not None:
        ids.add(int(exact["id"]))
    try:
        prefixes = sorted(chem.target_match_inchi_key_prefixes(smiles))
    except ValueError:
        # A wildcard target is valid SMILES but has no InChI; only its exact entry counts.
        prefixes = []
    for prefix in prefixes:
        # A range over the full key uses the inchi_key index, which LIKE would not.
        rows = conn.execute(
            "SELECT id, canonical_smiles FROM building_blocks "
            "WHERE inchi_key >= ? AND inchi_key < ?",
            (f"{prefix}-", f"{prefix}."),
        )
        for row in rows:
            if chem.matches_target(smiles, row["canonical_smiles"]):
                ids.add(int(row["id"]))
    return frozenset(ids)


def upsert_reaction_template(
    conn: sqlite3.Connection,
    reaction_id: str,
    library: str,
    name: str,
    description: str,
    smarts: str,
    tags_json: str,
    metadata_json: str,
) -> bool:
    existing = conn.execute(
        "SELECT id FROM reaction_templates WHERE reaction_id = ? AND library = ?",
        (reaction_id, library),
    ).fetchone()
    timestamp = now_iso()
    if existing is None:
        conn.execute(
            """
            INSERT INTO reaction_templates(
                reaction_id, library, name, description, smarts,
                tags_json, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reaction_id,
                library,
                name,
                description,
                smarts,
                tags_json,
                metadata_json,
                timestamp,
                timestamp,
            ),
        )
        return True
    conn.execute(
        """
        UPDATE reaction_templates
        SET name = ?, description = ?, smarts = ?,
            tags_json = ?, metadata_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (name, description, smarts, tags_json, metadata_json, timestamp, existing["id"]),
    )
    return False


_TEMPLATE_COLUMNS = "reaction_id, name, description, smarts, tags_json, metadata_json"


def get_reaction_template(conn: sqlite3.Connection, reaction_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        f"SELECT {_TEMPLATE_COLUMNS} FROM reaction_templates WHERE reaction_id = ?",
        (reaction_id,),
    ).fetchone()
    return _reaction_template_row(row) if row is not None else None


def get_reaction_template_by_smarts(conn: sqlite3.Connection, smarts: str) -> dict[str, Any] | None:
    """Return the one template with exactly this SMARTS, or None if none or several match."""

    rows = conn.execute(
        f"SELECT {_TEMPLATE_COLUMNS} FROM reaction_templates WHERE smarts = ? LIMIT 2",
        (str(smarts).strip(),),
    ).fetchall()
    return _reaction_template_row(rows[0]) if len(rows) == 1 else None


def search_reaction_templates(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Rank templates by TF-IDF similarity to a text query."""

    text = query.strip()
    if not text or text == "*":
        raise ValueError(f"Reaction template text search needs a query, not {query!r}")
    index = _reaction_text_index(conn)
    query_terms = set(_text_terms(text))
    query_smarts_terms = set(_smarts_terms(text))
    normalized_query = _normalize_reaction_search_phrase(text)
    scored = []
    for position, row in enumerate(index.rows):
        curated = any(
            _search_phrase_is_contained(normalized_query, hint)
            for hint in row.get("query_hints") or []
        ) - any(
            _search_phrase_is_contained(normalized_query, term)
            for term in row.get("search_exclusions") or []
        )
        scored.append(
            (
                row,
                index.text.cosine(query_terms, position)
                + index.smarts.cosine(query_smarts_terms, position),
                curated,
                row["reaction_id"].lower() == text.lower() or row["smarts"] == text,
                len(query_terms & index.text.documents[position].keys()) / max(1, len(query_terms)),
            )
        )
    # Cosine scores are scaled to the best match so curated hints and exclusions shift a
    # template by a fixed fraction of it.
    top = max((cosine for _, cosine, _, _, _ in scored), default=0.0) or 1.0
    matches = []
    for row, cosine, curated, exact_id, coverage in scored:
        score = cosine / top + REACTION_TEXT_SEARCH_CURATED_BONUS * curated
        if exact_id or score > 0:
            matches.append((row, score, exact_id, coverage))
    matches.sort(
        key=lambda match: (
            -match[2],
            -match[1],
            -match[3],
            match[0]["reaction_id"],
        )
    )
    return [copy.deepcopy(row) for row, *_ in matches[:limit]]


def _text_terms(text: str) -> list[str]:
    """Lowercase alphanumeric runs, so punctuation and hyphens separate terms.

    Single characters (SMARTS element letters, atom-map digits) and stopwords are dropped
    because they match nearly every template. Plurals fold to their singular form.
    """

    return [
        _singular(term)
        for term in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(term) >= 2 and term not in REACTION_TEXT_SEARCH_STOPWORDS
    ]


def _singular(term: str) -> str:
    """Strip plural endings only; full stemming would merge reaction names like amination into amine."""

    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "us", "is")):
        return term[:-1]
    return term


def _smarts_terms(text: str) -> list[str]:
    """Bracket atoms without atom-map numbers, so partial SMARTS queries match template atoms."""

    return [re.sub(r":\d+\]$", "]", atom) for atom in re.findall(r"\[[^\[\]]+\]", str(text))]


def _template_search_document(template: dict[str, Any]) -> Counter[str]:
    return Counter(
        _text_terms(
            " ".join(
                [
                    str(template.get("reaction_id", "")),
                    str(template.get("name", "")),
                    str(template.get("description", "")),
                    " ".join(template.get("tags") or []),
                    str(template.get("reaction_family", "")),
                    " ".join(template.get("query_hints") or []),
                    " ".join(
                        str(role.get("role", "")) for role in template.get("reactant_roles") or []
                    ),
                ]
            )
        )
    )


class _TfIdfVectors:
    """Sublinear-TF, smoothed-IDF document vectors."""

    def __init__(self, documents: list[Counter[str]]):
        self.documents = documents
        document_frequency = Counter(term for document in documents for term in document)
        self.idf = {
            term: math.log((1 + len(documents)) / (1 + count)) + 1
            for term, count in document_frequency.items()
        }
        self.norms = [
            math.sqrt(
                sum(
                    ((1 + math.log(count)) * self.idf[term]) ** 2
                    for term, count in document.items()
                )
            )
            for document in documents
        ]

    def cosine(self, terms: set[str], position: int) -> float:
        document, norm = self.documents[position], self.norms[position]
        query_norm = math.sqrt(sum(self.idf.get(term, 0.0) ** 2 for term in terms))
        dot = sum(
            (1 + math.log(document[term])) * self.idf[term] ** 2 for term in terms & document.keys()
        )
        return dot / (norm * query_norm) if norm and query_norm else 0.0


class _ReactionTextIndex:
    """Text and SMARTS vectors for the database's templates, kept apart so SMARTS never shifts text scores."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.text = _TfIdfVectors([_template_search_document(row) for row in rows])
        self.smarts = _TfIdfVectors([Counter(_smarts_terms(row.get("smarts", ""))) for row in rows])


def _reaction_text_index(conn: sqlite3.Connection) -> _ReactionTextIndex:
    database_path = _main_database_path(conn)
    if not database_path or conn.in_transaction:
        return _ReactionTextIndex(list_reaction_templates(conn))
    return _cached_reaction_text_index(database_path, *_reaction_template_revision(conn))


@lru_cache(maxsize=16)
def _cached_reaction_text_index(
    database_path: str,
    row_count: int,
    newest_update: str,
) -> _ReactionTextIndex:
    conn = connect(database_path)
    try:
        return _ReactionTextIndex(list_reaction_templates(conn))
    finally:
        conn.close()


def list_reaction_templates(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return every template in ID order."""

    cursor = conn.execute(
        f"SELECT {_TEMPLATE_COLUMNS} FROM reaction_templates ORDER BY reaction_id"
    )
    return [_reaction_template_row(row) for row in cursor.fetchall()]


def table_counts(db_path: str | Path) -> dict[str, int]:
    """Row counts of the tables this database has; older databases may lack some."""

    tables = [
        "building_blocks",
        "import_errors",
        "reaction_templates",
        "reaction_library_catalog",
    ]
    with session(db_path) as conn:
        present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        return {
            table: int(conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"])
            for table in tables
            if table in present
        }


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL records must be objects")
            yield value


def _metadata_text_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _metadata_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _reaction_template_row(row: sqlite3.Row) -> dict[str, Any]:
    # Provenance and source IDs stay in the database, out of every tool result.
    value = dict(row)
    decoded = {}
    for key, default in (("tags_json", []), ("metadata_json", {})):
        try:
            decoded[key] = json.loads(value.pop(key))
        except (TypeError, json.JSONDecodeError):
            decoded[key] = default
    value["tags"] = decoded["tags_json"]
    reactants, products = _split_reaction_smarts(str(value.get("smarts") or ""))
    reactant_count, product_count = reaction_template_arity(str(value.get("smarts") or ""))
    value["reactant_count"] = reactant_count
    value["product_count"] = product_count
    introduces_unmapped_product_atoms = _introduces_unmapped_product_atoms(products)
    value["introduces_unmapped_product_atoms"] = introduces_unmapped_product_atoms
    metadata = decoded["metadata_json"] if isinstance(decoded["metadata_json"], dict) else {}
    annotation = metadata.get("annotation") if isinstance(metadata.get("annotation"), dict) else {}
    if annotation:
        value["reaction_family"] = str(
            annotation.get("family") or annotation.get("reaction_family") or ""
        )
        value["query_hints"] = _metadata_text_list(annotation.get("search_terms"))
        value["search_exclusions"] = _metadata_text_list(annotation.get("search_exclusions"))
        value["reactant_roles"] = _metadata_dict_list(annotation.get("reactant_roles"))
        value["family_metadata"] = (
            dict(annotation["family_metadata"])
            if isinstance(annotation.get("family_metadata"), dict)
            else {}
        )
        value["overview"] = (
            dict(annotation["overview"]) if isinstance(annotation.get("overview"), dict) else {}
        )
        value["execution_hints"] = _metadata_text_list(
            annotation.get("retro_planning_notes")
        ) + _metadata_text_list(annotation.get("execution_caveats"))
        return value

    value["reaction_family"] = ""
    value["query_hints"] = list(value.get("tags") or [])
    value["search_exclusions"] = []
    value["reactant_roles"] = []
    value["family_metadata"] = {}
    value["overview"] = {}
    value["execution_hints"] = []
    return value


def _reaction_template_revision(conn: sqlite3.Connection) -> tuple[int, str]:
    row = conn.execute(
        "SELECT COUNT(*) AS row_count, COALESCE(MAX(updated_at), '') AS newest_update "
        "FROM reaction_templates"
    ).fetchone()
    return int(row["row_count"]), str(row["newest_update"])


def _main_database_path(conn: sqlite3.Connection) -> str:
    for row in conn.execute("PRAGMA database_list").fetchall():
        if str(row["name"]) == "main":
            return str(row["file"] or "")
    return ""


def _split_reaction_smarts(smarts: str) -> tuple[str, str]:
    if ">>" not in smarts:
        return smarts, ""
    reactants, products = smarts.split(">>", 1)
    return reactants, products


def reaction_template_arity(smarts: str) -> tuple[int, int]:
    """Return RDKit's reactant and product template counts for a reaction SMARTS."""

    reaction = chem.rdChemReactions.ReactionFromSmarts(smarts)
    if reaction is None:
        raise ValueError(f"RDKit could not parse reaction SMARTS {smarts!r}")
    reaction.Initialize()
    return reaction.GetNumReactantTemplates(), reaction.GetNumProductTemplates()


def _introduces_unmapped_product_atoms(products: str) -> bool:
    return bool(chem.smarts_products_have_unmapped_atoms(products))


def reaction_template_library_overview(
    conn: sqlite3.Connection,
    *,
    max_families: int = 8,
    max_notable_templates: int = 4,
) -> dict[str, Any]:
    templates = list_reaction_templates(conn)
    grouped: dict[str, dict[str, Any]] = {}
    for template in templates:
        family = str(template.get("reaction_family") or "unclassified")
        family_metadata = (
            template.get("family_metadata")
            if isinstance(template.get("family_metadata"), dict)
            else {}
        )
        family_query_hints = _metadata_text_list(family_metadata.get("search_terms"))
        group = grouped.setdefault(
            family,
            {
                "family": family,
                "description": str(family_metadata.get("description") or "").strip(),
                "query_hints": family_query_hints,
                "count": 0,
            },
        )
        if not group["description"]:
            group["description"] = str(family_metadata.get("description") or "").strip()
        if family_query_hints and not group["query_hints"]:
            group["query_hints"] = family_query_hints
        group["count"] += 1
    for group in grouped.values():
        if not group["description"] or not group["query_hints"]:
            raise ValueError(
                f"Reaction family {group['family']!r}: "
                "missing family_metadata.description or family_metadata.search_terms. "
                "Import the bundled catalog with `synclaw ingest-reactions --db <path>` and drop unused "
                "libraries with `synclaw db remove-reactions <name> --db <path>`, "
                "or add family descriptions and search terms to custom templates "
                "and reimport them with ingest-reactions --annotations."
            )
    families = sorted(
        grouped.values(),
        key=lambda item: (-int(item["count"]), str(item["family"])),
    )
    return {
        "template_count": len(templates),
        "families": families[: max(1, int(max_families))],
        "family_count": len(grouped),
        "notable_templates": _notable_reaction_templates(
            templates,
            limit=max_notable_templates,
        ),
    }


def _notable_reaction_templates(
    templates: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    scored = []
    for template in templates:
        overview = template.get("overview") if isinstance(template.get("overview"), dict) else {}
        priority = overview.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int) or priority <= 0:
            continue
        category = str(overview.get("category") or template.get("reaction_id") or "")
        scored.append((priority, category, str(template.get("reaction_id") or ""), template))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    notable = []
    seen_categories = set()
    for _, category, _, template in scored:
        if len(notable) >= limit:
            break
        if category in seen_categories:
            continue
        seen_categories.add(category)
        notable.append(
            {
                "name": template.get("name"),
                "reaction_family": template.get("reaction_family"),
                "description": template.get("description"),
                "query_hints": list(template.get("query_hints") or [])[:4],
            }
        )
    return notable


def _reaction_search_tokens(text: Any) -> list[str]:
    return [
        term
        for term in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_:+#=\\/\-]*",
            str(text).lower(),
        )
        if len(term) >= 2
    ]


def _normalize_reaction_search_phrase(text: Any) -> str:
    return " ".join(_reaction_search_tokens(text))


def _search_phrase_is_contained(normalized_query: str, candidate: Any) -> bool:
    normalized_candidate = _normalize_reaction_search_phrase(candidate)
    if not normalized_candidate:
        return False
    return f" {normalized_candidate} " in f" {normalized_query} "


def _with_match_metadata(
    row: dict[str, Any], match_type: str, similarity: float | None
) -> dict[str, Any]:
    row = dict(row)
    row["match_type"] = match_type
    row["similarity"] = similarity
    submit_reference = f"BB:{row['id']}"
    row["submit_reference"] = submit_reference
    row["submit_as"] = {
        "preferred": submit_reference,
        "raw_smiles": row.get("smiles"),
        "canonical_smiles": row.get("canonical_smiles"),
    }
    row["raw_smiles_submission_allowed"] = True
    row["query_is_exact_building_block"] = match_type == "exact"
    if match_type == "exact":
        row["submission_note"] = (
            f"Exact purchasable building-block match. Prefer {submit_reference}; "
            "the returned raw or canonical SMILES is also accepted by submit_pathway."
        )
    elif match_type == "near":
        row["submission_note"] = (
            f"Near purchasable building-block match. Prefer {submit_reference}; "
            "do not submit the original query SMILES unless it appears as an exact building-block match."
        )
    else:
        row["submission_note"] = (
            f"Purchasable building-block result. Prefer {submit_reference}; "
            "the returned raw or canonical SMILES is accepted by submit_pathway."
        )
    return row
