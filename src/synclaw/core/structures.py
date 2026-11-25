# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readers for building-block files, and the salt policy every importer shares."""

from __future__ import annotations

import gzip
import itertools
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

STRUCTURE_FORMATS = ("smiles", "sdf")
MULTICOMPONENT_POLICIES = ("drop", "major", "keep")

# One input row: its line number, the supplier's ID, the molecule (None if unreadable), and its
# text (the SMILES as written; for an SDF record, empty unless it could not be read).
Row = tuple[int, str, "Chem.Mol | None", str]
# The published eMolecules files write the tag as ">  <EMOL_VERSION_ID>", with two spaces.
_SDF_ID_TAG = re.compile(r">\s+<EMOL_(VERSION|PARENT)_ID>")


def check_multicomponent_policy(policy: str) -> None:
    """Raise ``ValueError`` unless ``policy`` is one of ``MULTICOMPONENT_POLICIES``."""
    if policy not in MULTICOMPONENT_POLICIES:
        raise ValueError(f"multicomponent must be one of {MULTICOMPONENT_POLICIES}")


def major_component(mol: Any) -> Any:
    """The component the ``major`` policy keeps: most heavy atoms, ties broken by canonical SMILES.

    Ties are common (about a quarter of the multi-component rows in eMolecules), so they are broken
    by canonical SMILES rather than by component order.
    """
    components = Chem.GetMolFrags(mol, asMols=True)
    return min(components, key=lambda part: (-part.GetNumHeavyAtoms(), Chem.MolToSmiles(part)))


def apply_multicomponent(mol: Any, policy: str) -> tuple[Any, bool]:
    """The molecule after the policy (None if dropped), and whether it had several components."""
    if len(Chem.GetMolFrags(mol)) < 2:
        return mol, False
    if policy == "drop":
        return None, True
    return (major_component(mol) if policy == "major" else mol), True


def _open(path: str | Path) -> Any:
    return (
        gzip.open(path, "rt", encoding="utf-8")
        if str(path).endswith(".gz")
        else open(path, encoding="utf-8")
    )


def detect_format(path: str | Path) -> str:
    """``sdf`` if the name says so or the first lines hold a mol-block marker, else ``smiles``."""
    if re.search(r"\.sdf(\.gz)?$", str(path), re.IGNORECASE):
        return "sdf"
    with _open(path) as handle:
        head = list(itertools.islice(handle, 40))
    marked = any(
        "V2000" in line or "V3000" in line or line.startswith(("M  END", "$$$$")) for line in head
    )
    return "sdf" if marked else "smiles"


@contextmanager
def open_structures(
    path: str | Path, file_format: str | None = None
) -> Iterator[tuple[str, str, Iterator[Row]]]:
    """Open a ``smiles`` or ``sdf`` file, read compressed if it ends in ``.gz``.

    ``file_format`` is detected from the file if unset. Yields the ID column's name, the
    structure column's name and the rows. A ``smiles`` row is a SMILES then an optional ID, and
    the file may start with a ``SMILES`` or ``isosmiles`` header. An ``sdf`` record is a mol block
    whose ``EMOL_VERSION_ID`` or ``EMOL_PARENT_ID`` field is the ID.
    """
    file_format = file_format or detect_format(path)
    if file_format not in STRUCTURE_FORMATS:
        raise ValueError(f"file format must be one of {STRUCTURE_FORMATS}")
    with _open(path) as handle:
        yield _sdf_rows(handle) if file_format == "sdf" else _smiles_rows(handle)


def _smiles_rows(handle: Any) -> tuple[str, str, Iterator[Row]]:
    first = handle.readline()
    head = first.split()
    has_header = bool(head) and head[0].lower() in {"smiles", "isosmiles"}
    lines = handle if has_header else itertools.chain([first], handle)

    def rows() -> Iterator[Row]:
        for number, line in enumerate(lines, start=2 if has_header else 1):
            fields = line.split()
            if fields:
                source_id = fields[1] if len(fields) > 1 else ""
                yield number, source_id, Chem.MolFromSmiles(fields[0]), fields[0]

    return (
        head[1] if has_header and len(head) > 1 else "id",
        head[0] if has_header else "smiles",
        rows(),
    )


def _sdf_rows(handle: Any) -> tuple[str, str, Iterator[Row]]:
    def rows() -> Iterator[Row]:
        lines: list[str] = []
        first_line = 1
        for number, line in enumerate(handle, start=1):
            if not line.startswith("$$$$"):
                lines.append(line)
                continue
            end = next((i for i, text in enumerate(lines) if text.startswith("M  END")), None)
            mol = Chem.MolFromMolBlock("".join(lines[: end + 1])) if end is not None else None
            ids = {
                m[1]: lines[i + 1].strip()
                for i, t in enumerate(lines)
                if (m := _SDF_ID_TAG.match(t))
            }
            record_id = ids.get("VERSION") or ids.get("PARENT", "")
            yield first_line, record_id, mol, "" if mol else "<unreadable mol block>"
            lines = []
            first_line = number + 1

    return "id", "structure", rows()
