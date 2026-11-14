# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the reactant-template matrix directory that `tasks.targets` generates from.

`generate_targets` needs to know *which* building blocks fit reactant position 2 of a given
template -- a question `search_building_blocks` cannot answer, since exact and Tanimoto search
cannot express a substructure query. The answer is precomputed here, once, as a matrix whose
rows are building blocks and whose columns are reaction templates.

Templates come from this package's own `data/reaction_templates/<library>/templates.jsonl`, so a
matrix built here cannot cite a template the installed package does not have. Blocks come from a
building-block catalog: a `SMILES`- or `isosmiles`-headed file, optionally gzipped, whose first
whitespace-separated column is the SMILES.

The build has two phases, both of which `build_matrix_directory` runs:

1. *Catalog*: read the selected template libraries and the building-block catalog, canonicalize and
   deduplicate the blocks, and write everything except `matrix.npy`.
2. *Fill*: compile each template's reactant queries once per worker, then test every block
   against every query. This is the expensive phase: the matrix has one byte per block per template,
   and the work is spread across worker processes.

Workers write disjoint row ranges of one memory-mapped `matrix.npy`, so the matrix itself is never
collected in the parent. The parent does hold every block's SMILES, through both phases -- about
0.5 GB for 5M blocks -- so peak memory scales with the catalog, just far more gently than the
matrix does.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from synclaw import paths
from synclaw.core.libraries import DEFAULT_REACTION_LIBRARIES, normalize_libraries
from synclaw.core.reactions import _rdkit_reaction_modules
from synclaw.tasks.targets import (
    BUILDING_BLOCK_FILE,
    MATRIX_FILE,
    MAX_REACTANTS,
    OFFSET_FILE,
    REACTION_FILE,
    TEMPLATE_ID_FILE,
    line_offsets,
    read_smiles_lines,
)

COLUMN_FILE = "columns.npz"
META_FILE = "meta.json"
DROPPED_TEMPLATE_FILE = "dropped_templates.tsv"

COMMON_REAGENT_FILE = paths.DATA_DIR / "building_blocks" / "common_reagent_building_blocks.txt"
DEFAULT_CHUNK_SIZE = 20_000

MULTICOMPONENT_POLICIES = ("drop", "major", "keep")
ATOM_MAP_POLICIES = ("drop", "strip", "keep")


def _progress(message: str) -> None:
    """Progress goes to stderr; stdout stays a single JSON document for callers to parse."""
    print(message, file=sys.stderr, flush=True)


def _pool_context():
    """`spawn`, matching `tasks.targets`, so a worker never inherits parent state it did not ask for.

    `fork` also works here and starts faster. It is not worth the difference: the fill phase runs
    for minutes, and forking a parent that has already imported RDKit, run a pool and mapped a
    multi-gigabyte array is the kind of thing that fails far from its cause.
    """
    return mp.get_context("spawn")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- templates


def load_template_records(libraries: Sequence[str]) -> list[dict[str, Any]]:
    """Read the selected libraries' template records, deduplicated by template id.

    A template is stored once in its owning library's file and records cross-library membership
    in `libraries`, so a union reads whole files and drops any id already seen.
    """
    root = paths.DATA_DIR / "reaction_templates"
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for library in libraries:
        path = root / library / "templates.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"no template file for library {library!r} at {path}")
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["template_id"] in seen:
                    continue
                seen.add(record["template_id"])
                records.append(record)
    return records


def write_templates(records: Sequence[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """Write reactions.txt / template_ids.txt, dropping what the engine cannot execute."""
    _, AllChem, rdChemReactions = _rdkit_reaction_modules()
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[str, str]] = []
    for record in records:
        smarts = record["smarts"].strip()
        if "\t" in smarts or "\n" in smarts:
            dropped.append((record["template_id"], "SMARTS contains a tab or newline"))
            continue
        try:
            reaction = AllChem.ReactionFromSmarts(smarts)
            rdChemReactions.ChemicalReaction.Initialize(reaction)
            num_reactants = reaction.GetNumReactantTemplates()
        except Exception as exc:  # noqa: BLE001 - the reason is reported, not handled
            dropped.append((record["template_id"], f"RDKit could not initialize it: {exc}"))
            continue
        if num_reactants == 0:
            dropped.append((record["template_id"], "no reactant templates"))
        elif num_reactants > MAX_REACTANTS:
            dropped.append(
                (record["template_id"], f"{num_reactants} reactants; the engine supports <= {MAX_REACTANTS}")
            )
        else:
            kept.append(record)

    with open(out_dir / REACTION_FILE, "w", newline="\n") as handle:
        handle.writelines(record["smarts"].strip() + "\n" for record in kept)
    with open(out_dir / TEMPLATE_ID_FILE, "w", newline="\n") as handle:
        handle.writelines(record["template_id"] + "\n" for record in kept)
    with open(out_dir / DROPPED_TEMPLATE_FILE, "w", newline="\n") as handle:
        handle.write("template_id\treason\n")
        handle.writelines(f"{tid}\t{reason}\n" for tid, reason in dropped)

    by_library: dict[str, int] = {}
    for record in kept:
        by_library[record["library"]] = by_library.get(record["library"], 0) + 1
    return {"kept": len(kept), "dropped": len(dropped), "by_library": by_library}


# --------------------------------------------------------------------------- building blocks


def _largest_component(smiles: str, Chem: Any) -> str:
    """The component with the most atoms, matching how a major-component policy picks one."""
    parts = [Chem.MolFromSmiles(part) for part in smiles.split(".")]
    best = max((p for p in parts if p is not None), key=lambda m: m.GetNumAtoms(), default=None)
    return "" if best is None else Chem.MolToSmiles(best, canonical=True, isomericSmiles=True)


def _canonicalize(task: tuple[list[str], str, str]) -> list[tuple[str, str]]:
    """Canonicalize a chunk, returning (smiles, status) per row.

    Status is "ok", "invalid" (RDKit rejected it), "multicomponent" (a salt or mixture the `drop`
    policy excludes) or "atom_mapped". `major` rewrites a mixture to its largest component and
    `strip` clears atom maps; both then report "ok".
    """
    Chem, _, _ = _rdkit_reaction_modules()
    block, multicomponent, atom_maps = task
    out: list[tuple[str, str]] = []
    for smiles in block:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            out.append(("", "invalid"))
            continue
        if any(atom.GetAtomMapNum() for atom in mol.GetAtoms()):
            if atom_maps == "drop":
                out.append(("", "atom_mapped"))
                continue
            if atom_maps == "strip":
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if "." in canonical:
            if multicomponent == "drop":
                out.append(("", "multicomponent"))
                continue
            if multicomponent == "major":
                canonical = _largest_component(canonical, Chem)
                if not canonical:
                    out.append(("", "invalid"))
                    continue
        out.append((canonical, "ok"))
    return out


def _read_building_blocks(path: Path) -> Iterator[str]:
    """Yield the SMILES column of a building-block catalog, gzipped or plain."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        if not header or header[0].lower() not in {"smiles", "isosmiles"}:
            raise ValueError(f"unexpected building-block header in {path}: {header!r}")
        for line in handle:
            fields = line.split()
            if fields:
                yield fields[0]


def _chunks(iterable: Iterable[str], size: int) -> Iterator[list[str]]:
    block: list[str] = []
    for item in iterable:
        block.append(item)
        if len(block) >= size:
            yield block
            block = []
    if block:
        yield block


def write_building_blocks(
    building_blocks: Path,
    reagents: Path | None,
    out_dir: Path,
    workers: int,
    multicomponent: str = "drop",
    atom_maps: str = "drop",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Canonicalize, deduplicate and write the combined building-block file.

    Chunks are canonicalized in parallel but consumed in source order, so the surviving
    representative of a duplicate group is its first occurrence in the catalog.
    """
    seen: set[str] = set()
    read = invalid = duplicates = dropped_mixtures = dropped_mapped = 0
    out_path = out_dir / BUILDING_BLOCK_FILE
    offsets: list[int] = []
    position = 0

    def record(canonical: str, status: str) -> None:
        nonlocal invalid, duplicates, dropped_mixtures, dropped_mapped, position
        if status == "invalid":
            invalid += 1
        elif status == "multicomponent":
            dropped_mixtures += 1
        elif status == "atom_mapped":
            dropped_mapped += 1
        elif canonical in seen:
            duplicates += 1
        else:
            seen.add(canonical)
            offsets.append(position)
            position += out.write(canonical + "\n")

    with open(out_path, "w", newline="\n") as out:
        position += out.write("SMILES\n")
        with _pool_context().Pool(processes=workers) as pool:
            tasks = (
                (block, multicomponent, atom_maps)
                for block in _chunks(_read_building_blocks(building_blocks), chunk_size)
            )
            for block in pool.imap(_canonicalize, tasks, chunksize=1):
                for canonical, status in block:
                    read += 1
                    record(canonical, status)
        source_rows = read
        source_written = len(seen)

        if reagents is not None:
            lines = [line.strip() for line in open(reagents)]
            if lines and lines[0].upper() == "SMILES":
                lines = lines[1:]
            reagent_smiles = [line for line in lines if line]
            read += len(reagent_smiles)
            for canonical, status in _canonicalize((reagent_smiles, multicomponent, atom_maps)):
                record(canonical, status)

    offsets.append(position)
    np.save(out_dir / OFFSET_FILE, np.array(offsets, dtype=np.int64))
    return {
        "rows_read": read,
        "source_rows": source_rows,
        "invalid": invalid,
        "duplicates": duplicates,
        "multicomponent_dropped": dropped_mixtures,
        "atom_mapped_dropped": dropped_mapped,
        "written": len(seen),
        "reagents_added": len(seen) - source_written,
    }


# --------------------------------------------------------------------------- matrix fill

# Set once per worker by _init_worker; compiling the reactant queries is the expensive part.
_QUERIES: list[list[Any]] | None = None
_MATRIX_PATH: Path | None = None


def reactant_queries(smarts: str) -> list[Any]:
    """The reactant-side substructure queries of one template, in reactant order.

    `GetReactantTemplate` hands back a view into the reaction rather than a molecule of its own,
    so each one is copied: otherwise the reaction is collected when this returns and every later
    match reads freed memory, which segfaults the worker and leaves the pool waiting forever for
    a result that will never arrive.
    """
    Chem, AllChem, rdChemReactions = _rdkit_reaction_modules()
    reaction = AllChem.ReactionFromSmarts(smarts)
    rdChemReactions.ChemicalReaction.Initialize(reaction)
    return [Chem.Mol(reaction.GetReactantTemplate(i)) for i in range(reaction.GetNumReactantTemplates())]


def _init_worker(reaction_smarts: list[str], matrix_path: str) -> None:
    global _QUERIES, _MATRIX_PATH
    _QUERIES = [reactant_queries(smarts) for smarts in reaction_smarts]
    _MATRIX_PATH = Path(matrix_path)


def _fill_block(task: tuple[int, list[str]]) -> tuple[int, int]:
    """Fill rows [start, start + len(block)) of the matrix; return (start, rows written)."""
    Chem, _, _ = _rdkit_reaction_modules()
    start, smiles_block = task
    assert _QUERIES is not None and _MATRIX_PATH is not None
    block = np.zeros((len(smiles_block), len(_QUERIES)), dtype=np.uint8)
    for i, smiles in enumerate(smiles_block):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for j, queries in enumerate(_QUERIES):
            flag = 0
            for position, query in enumerate(queries):
                if mol.HasSubstructMatch(query):
                    flag |= 1 << position
            if flag:
                block[i, j] = flag
    # Reopened per block: an F-ordered row slice assignment writes one contiguous run per column,
    # and disjoint row ranges in a shared mapping never collide.
    matrix = np.load(_MATRIX_PATH, mmap_mode="r+")
    matrix[start : start + len(smiles_block), :] = block
    matrix.flush()
    del matrix
    return start, len(smiles_block)


def compute_columns(matrix: np.ndarray, num_reactants: Sequence[int], chunk: int = 1 << 19) -> dict[str, np.ndarray]:
    """Per-column reactant counts and the seed-reaction indices, computed in row chunks.

    A seed reaction is one whose every reactant position is fillable from the catalog, so a route
    can start there. Rows are streamed, so no intermediate is materialized at full size.
    """
    num_rows, num_cols = matrix.shape
    counts = np.zeros(num_cols, dtype=np.int32)
    or_reduce = np.zeros(num_cols, dtype=np.uint8)
    for start in range(0, num_rows, chunk):
        block = np.asarray(matrix[start : start + chunk])
        counts += (block != 0).sum(axis=0, dtype=np.int32)
        or_reduce |= np.bitwise_or.reduce(block, axis=0)
    full_flag = np.array([(1 << n) - 1 for n in num_reactants], dtype=np.uint8)
    seeds = np.nonzero(full_flag == or_reduce)[0].tolist()
    return {
        "reactant_count": counts,
        "or_reduce": or_reduce,
        "seed_reaction_indices": np.array(seeds, dtype=np.int64),
    }


def fill_matrix(
    directory: Path,
    workers: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fill `matrix.npy` from the catalog files already written into `directory`."""
    smiles = read_smiles_lines(directory / BUILDING_BLOCK_FILE)
    if limit is not None:
        smiles = smiles[:limit]
        # The matrix rows must line up with the block file, so shorten it to what was used.
        block_path = directory / BUILDING_BLOCK_FILE
        with open(block_path, "w", newline="\n") as handle:
            handle.write("SMILES\n")
            handle.writelines(s + "\n" for s in smiles)
        np.save(directory / OFFSET_FILE, line_offsets(block_path, skip_header=True))

    reaction_smarts = [line.strip() for line in open(directory / REACTION_FILE) if line.strip()]
    shape = (len(smiles), len(reaction_smarts))
    matrix_path = directory / MATRIX_FILE
    _progress(
        f"filling a {shape[0]:,} x {shape[1]} matrix ({shape[0] * shape[1] / 1e9:.2f} GB) "
        f"with {workers} workers"
    )

    # Create the .npy up front so workers can map it read/write; Fortran order for column reads.
    handle = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.uint8, shape=shape, fortran_order=True)
    del handle

    tasks = [(start, smiles[start : start + chunk_size]) for start in range(0, len(smiles), chunk_size)]
    started = time.time()
    rows_done = 0
    with _pool_context().Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(reaction_smarts, str(matrix_path)),
    ) as pool:
        for _, rows in pool.imap_unordered(_fill_block, tasks, chunksize=1):
            rows_done += rows
            if rows_done % (chunk_size * 50) < chunk_size:
                _progress(f"  {rows_done:,} / {shape[0]:,} blocks")
    fill_seconds = time.time() - started

    matrix = np.load(matrix_path, mmap_mode="r")
    columns = compute_columns(matrix, [len(queries) for queries in map(reactant_queries, reaction_smarts)])
    np.savez(directory / COLUMN_FILE, **columns)
    del matrix

    return {
        "built_at": datetime.now(UTC).isoformat(),
        "shape": list(shape),
        "fortran_order": True,
        "fill_seconds": round(fill_seconds, 1),
        "workers": workers,
        "limit": limit,
        "seed_reactions": len(columns["seed_reaction_indices"]),
        "reactions_with_a_matching_block": int((columns["reactant_count"] > 0).sum()),
        "mean_reactants_per_reaction": float(columns["reactant_count"].mean()),
    }


# --------------------------------------------------------------------------- orchestration


def build_matrix_directory(
    building_blocks: str | Path,
    output: str | Path,
    libraries: Sequence[str] = DEFAULT_REACTION_LIBRARIES,
    workers: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
    multicomponent: str = "drop",
    atom_maps: str = "drop",
    common_reagents: bool = True,
) -> dict[str, Any]:
    """Write a complete reactant-template matrix directory and return its manifest."""
    if multicomponent not in MULTICOMPONENT_POLICIES:
        raise ValueError(f"multicomponent must be one of {MULTICOMPONENT_POLICIES}")
    if atom_maps not in ATOM_MAP_POLICIES:
        raise ValueError(f"atom_maps must be one of {ATOM_MAP_POLICIES}")

    selected = normalize_libraries(list(libraries)) or DEFAULT_REACTION_LIBRARIES
    building_blocks = Path(building_blocks)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    workers = workers or os.cpu_count() or 8

    _progress(f"reading templates [{', '.join(selected)}]")
    records = load_template_records(selected)
    template_stats = write_templates(records, directory)
    _progress(
        f"  {len(records)} records -> {template_stats['kept']} usable, "
        f"{template_stats['dropped']} dropped {template_stats['by_library']}"
    )

    reagents = COMMON_REAGENT_FILE if common_reagents else None
    if reagents is not None and not reagents.exists():
        raise FileNotFoundError(f"common reagents not found at {reagents}")

    _progress(f"canonicalizing building blocks from {building_blocks} with {workers} workers")
    block_stats = write_building_blocks(
        building_blocks, reagents, directory, workers,
        multicomponent=multicomponent, atom_maps=atom_maps, chunk_size=chunk_size,
    )
    _progress(
        f"  {block_stats['written']:,} unique blocks ({block_stats['duplicates']:,} duplicates, "
        f"{block_stats['invalid']:,} invalid, {block_stats['multicomponent_dropped']:,} multi-component, "
        f"{block_stats['atom_mapped_dropped']:,} atom-mapped, {block_stats['reagents_added']} reagents added)"
    )

    matrix_stats = fill_matrix(directory, workers, chunk_size=chunk_size, limit=limit)
    _progress(f"  filled in {matrix_stats['fill_seconds'] / 60:.1f} min")
    if limit is not None:
        # `fill_matrix` truncated the block file to the rows it filled, so the counts must describe
        # that file. Reagents follow the catalog rows, so a limit drops them first.
        kept = matrix_stats["shape"][0]
        from_source = block_stats["written"] - block_stats["reagents_added"]
        block_stats["written_before_limit"] = block_stats["written"]
        block_stats["written"] = kept
        block_stats["reagents_added"] = max(0, kept - from_source)

    from rdkit import (
        rdBase,  # imported late, as core.reactions does, to keep import cost off callers
    )

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rdkit_version": str(rdBase.rdkitVersion),
        "libraries": list(selected),
        "building_blocks_source": str(building_blocks),
        "building_blocks_sha256": _sha256(building_blocks),
        "common_reagents": None if reagents is None else str(reagents),
        "multicomponent_policy": multicomponent,
        "atom_map_policy": atom_maps,
        "templates": template_stats,
        "building_blocks": block_stats,
        "matrix": matrix_stats,
        "output": str(directory),
    }
    (directory / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    return meta
