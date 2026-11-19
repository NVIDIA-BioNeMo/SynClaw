# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for collecting model-run result summaries."""

from __future__ import annotations

import csv
import glob
from pathlib import Path
from typing import Any


def combine_summary_csvs(inputs: list[str | Path], output: str | Path) -> dict[str, Any]:
    output_path = Path(output)
    input_paths = _expand_inputs(inputs, output_path)
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = ["source_summary_file"]

    for path in input_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for name in reader.fieldnames:
                if name and name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                rows.append(
                    {
                        "source_summary_file": str(path),
                        **{str(key): value or "" for key, value in row.items() if key},
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "input_files": [str(path) for path in input_paths],
        "output": str(output_path),
        "rows": len(rows),
        "columns": fieldnames,
    }


def _expand_inputs(inputs: list[str | Path], output_path: Path) -> list[Path]:
    paths: list[Path] = []
    resolved_output = output_path.resolve()
    for value in inputs:
        text = str(value)
        matches = sorted(Path(match) for match in glob.glob(text))
        candidates = matches if matches else [Path(text)]
        for candidate in candidates:
            if candidate.resolve() == resolved_output:
                continue
            if candidate.is_file() and candidate not in paths:
                paths.append(candidate)
    if not paths:
        raise RuntimeError("No result CSV files matched the provided inputs")
    return paths
