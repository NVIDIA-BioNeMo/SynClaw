#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact and near building-block searches against prepared BioNeMo-SynClaw DBs."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from synclaw.core import db
from synclaw.tools.contracts import TOOL_ARGUMENT_DEFAULTS
from synclaw.tools.synthesis import SynthesisTools

SEARCH_DEFAULTS = TOOL_ARGUMENT_DEFAULTS["search_building_blocks"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Prepared search DB; repeat to compare databases",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        help="Optional one-SMILES-per-line query file; defaults to sampling the first DB",
    )
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=SEARCH_DEFAULTS["limit"])
    parser.add_argument("--min-similarity", type=float, default=SEARCH_DEFAULTS["min_similarity"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    databases = dict(_parse_database(value) for value in args.db)
    query_count = max(1, int(args.queries))
    if args.query_file is None:
        first_database = next(iter(databases.values()))
        queries = _sample_database(first_database, count=query_count, seed=args.seed)
        query_source = str(first_database)
    else:
        queries = _sample_query_file(args.query_file, count=query_count, seed=args.seed)
        query_source = str(args.query_file)
    if not queries:
        raise SystemExit("No query SMILES were available")

    results = {
        "query_source": query_source,
        "query_count": len(queries),
        "seed": args.seed,
        "concurrency": max(1, int(args.concurrency)),
        "limit": max(1, int(args.limit)),
        "min_similarity": float(args.min_similarity),
        "databases": {
            label: _benchmark_database(
                path,
                queries=queries,
                concurrency=max(1, int(args.concurrency)),
                limit=max(1, int(args.limit)),
                min_similarity=float(args.min_similarity),
                warmup=max(0, int(args.warmup)),
            )
            for label, path in databases.items()
        },
    }
    rendered = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def _parse_database(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise SystemExit(f"Invalid --db value {value!r}; expected LABEL=PATH")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Database does not exist: {path}")
    return label.strip(), path


def _sample_database(path: Path, *, count: int, seed: int) -> list[str]:
    with db.session(path) as conn:
        db.require_prepared_building_block_search(path)
        (row_count,) = conn.execute("SELECT COUNT(*) FROM building_blocks").fetchone()
        if row_count == 0:
            return []
        # Sample row offsets, not id values: probing random ids and snapping to the
        # next row up spins forever on a sparse id space, because every candidate
        # maps onto rows already taken.
        rng = random.Random(seed)
        offsets = rng.sample(range(int(row_count)), k=min(count, int(row_count)))
        # One JSON parameter keeps large --queries under SQLite's bound-variable limit.
        rows = conn.execute(
            """
            SELECT canonical_smiles
            FROM (
                SELECT canonical_smiles, ROW_NUMBER() OVER (ORDER BY id) - 1 AS position
                FROM building_blocks
            )
            WHERE position IN (SELECT value FROM json_each(?))
            """,
            (json.dumps(offsets),),
        ).fetchall()
        return [str(row["canonical_smiles"]) for row in rows]


def _sample_query_file(path: Path, *, count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            value = raw_line.strip()
            if not value or (line_number == 1 and value.lower() == "smiles"):
                continue
            seen_rows += 1
            if len(reservoir) < count:
                reservoir.append(value)
                continue
            replacement = rng.randrange(seen_rows)
            if replacement < count:
                reservoir[replacement] = value
    rng.shuffle(reservoir)
    return reservoir


def _benchmark_database(
    path: Path,
    *,
    queries: list[str],
    concurrency: int,
    limit: int,
    min_similarity: float,
    warmup: int,
) -> dict[str, Any]:
    db.require_prepared_building_block_search(path)
    tools = SynthesisTools(path)
    # Warm both modes so neither timing includes a one-time load (the fingerprint index for near).
    for query in queries[:warmup]:
        for include_near in (False, True):
            tools.search_building_blocks(
                query, limit=limit, include_near=include_near, min_similarity=min_similarity
            )
    return {
        "path": str(path),
        "exact_only": _benchmark_mode(
            tools,
            queries=queries,
            concurrency=concurrency,
            limit=limit,
            include_near=False,
            min_similarity=min_similarity,
        ),
        "with_near": _benchmark_mode(
            tools,
            queries=queries,
            concurrency=concurrency,
            limit=limit,
            include_near=True,
            min_similarity=min_similarity,
        ),
    }


def _benchmark_mode(
    tools: SynthesisTools,
    *,
    queries: list[str],
    concurrency: int,
    limit: int,
    include_near: bool,
    min_similarity: float,
) -> dict[str, Any]:
    def run(query: str) -> tuple[float, int]:
        start = time.perf_counter()
        result = tools.search_building_blocks(
            query,
            limit=limit,
            include_near=include_near,
            min_similarity=min_similarity,
        )
        return (time.perf_counter() - start) * 1000.0, int(result["count"])

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        records = list(executor.map(run, queries))
    wall_seconds = time.perf_counter() - wall_start
    latencies = [record[0] for record in records]
    counts = [record[1] for record in records]
    return {
        "requests": len(records),
        "wall_seconds": wall_seconds,
        "requests_per_second": len(records) / wall_seconds if wall_seconds else None,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies),
        },
        "mean_results": sum(counts) / len(counts),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


if __name__ == "__main__":
    main()
