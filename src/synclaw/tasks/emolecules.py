# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download and normalize the eMolecules orderable building-block catalog."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import urllib.request
import zlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# RDKit directly rather than synclaw.core.chem: validate_smiles also computes an
# InChIKey and formula per molecule, which this catalog-sized stream discards.
from rdkit import Chem, RDLogger, rdBase

from synclaw.tasks.matrix import MULTICOMPONENT_POLICIES

EMOLECULES_DOWNLOAD_ROOT = "https://downloads.emolecules.com/orderbb"
_RELEASE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

ProgressCallback = Callable[[dict[str, Any]], None]
CONVERT_PROGRESS_INTERVAL = 100_000
RDLogger.DisableLog("rdApp.*")


def prepare_emolecules_building_blocks(
    *,
    dest_root: str | Path = ".",
    release: str,
    source_url: str | None = None,
    multicomponent: str = "major",
    force: bool = False,
    timeout: float = 60.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Download and convert an eMolecules parent archive for ``synclaw ingest-bbs``.

    The derived file contains a ``SMILES`` header followed by one canonical
    isomeric SMILES per line. Parent IDs are intentionally omitted because the
    BioNeMo-SynClaw importer expects the complete line to be a SMILES value.
    ``multicomponent`` sets how salts and mixtures are handled: ``keep`` them as
    shipped, reduce each to its ``major`` (largest by heavy atoms) component, or
    ``drop`` them.
    """

    _check_policy(multicomponent)
    if _RELEASE_PATTERN.fullmatch(release) is None:
        raise RuntimeError("eMolecules release must use YYYY-MM-DD format")

    url = source_url or f"{EMOLECULES_DOWNLOAD_ROOT}/{release}/parent.smi.gz"
    destination_dir = Path(dest_root) / "data" / "building_blocks" / "emolecules"
    archive_path = destination_dir / f"parent_{release}.smi.gz"
    output_path = destination_dir / f"emolecules_parent_{release}.txt"
    invalid_path = destination_dir / f"emolecules_parent_{release}_invalid.tsv"
    manifest_path = destination_dir / f"emolecules_parent_{release}_manifest.json"
    destination_dir.mkdir(parents=True, exist_ok=True)

    downloaded = False
    if force or not archive_path.exists():
        _download_archive(url, archive_path, timeout=timeout)
        downloaded = True
        _emit(progress, phase="downloaded", path=str(archive_path), url=url)
    else:
        _validate_gzip(archive_path)
        _emit(progress, phase="reused_archive", path=str(archive_path), url=url)

    source_sha256 = _sha256(archive_path)
    if (
        not force
        and output_path.exists()
        and invalid_path.exists()
        and _manifest_matches(
            manifest_path,
            source_sha256=source_sha256,
            output_path=output_path,
            multicomponent=multicomponent,
        )
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            **manifest,
            "archive_status": "downloaded" if downloaded else "reused",
            "conversion_status": "reused",
        }

    stats = convert_emolecules_parent_archive(
        archive_path,
        output_path,
        invalid_path=invalid_path,
        multicomponent=multicomponent,
        progress=progress,
    )
    manifest = {
        "release": release,
        "source_url": url,
        "source_archive": str(archive_path),
        "source_sha256": source_sha256,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "invalid_output": str(invalid_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "rdkit_version": str(rdBase.rdkitVersion),
        "format": "SMILES header followed by one canonical isomeric SMILES per line",
        "deduplication": "first canonical occurrence in source order, after the multicomponent policy",
        "multicomponent": multicomponent,
        **stats,
    }
    _write_json_atomic(manifest_path, manifest)
    return {
        **manifest,
        "archive_status": "downloaded" if downloaded else "reused",
        "conversion_status": "converted",
    }


def convert_emolecules_parent_archive(
    archive_path: str | Path,
    output_path: str | Path,
    *,
    invalid_path: str | Path,
    multicomponent: str = "major",
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """Convert ``parent.smi.gz`` to one validated, de-duplicated SMILES per line.

    The multicomponent policy applies before canonical de-duplication, so a salt reduced
    to a component that is already a row counts as a duplicate.
    """

    _check_policy(multicomponent)
    source = Path(archive_path)
    destination = Path(output_path)
    invalid_destination = Path(invalid_path)
    # No pre-scan here: callers validate the archive once on download or reuse, and
    # the streaming read below raises on a bad CRC of its own accord.
    destination.parent.mkdir(parents=True, exist_ok=True)
    invalid_destination.parent.mkdir(parents=True, exist_ok=True)

    output_temporary = destination.with_name(f".{destination.name}.tmp")
    invalid_temporary = invalid_destination.with_name(f".{invalid_destination.name}.tmp")
    seen: set[str] = set()
    source_rows = 0
    valid_rows = 0
    invalid_rows = 0
    duplicate_rows = 0
    multicomponent_rows = 0
    multicomponent_dropped = 0

    try:
        with (
            gzip.open(source, "rt", encoding="utf-8") as source_handle,
            output_temporary.open("w", encoding="utf-8", newline="\n") as output_handle,
            invalid_temporary.open("w", encoding="utf-8", newline="\n") as invalid_handle,
        ):
            header = source_handle.readline().strip().split()
            if not header or header[0].lower() not in {"smiles", "isosmiles"}:
                raise RuntimeError(f"Unexpected eMolecules parent header: {header!r}")

            output_handle.write("SMILES\n")
            invalid_handle.write("source_line\tparent_id\tisosmiles\n")

            def report() -> None:
                _emit(
                    progress,
                    phase="convert",
                    source_rows=source_rows,
                    output_rows=len(seen),
                    invalid_rows=invalid_rows,
                    duplicate_rows=duplicate_rows,
                )

            for source_line, raw_line in enumerate(source_handle, start=2):
                # Report before the row is classified so invalid and duplicate rows,
                # which `continue` below, cannot skip a boundary.
                if source_rows and source_rows % CONVERT_PROGRESS_INTERVAL == 0:
                    report()
                source_rows += 1
                fields = raw_line.strip().split()
                if len(fields) < 2:
                    invalid_rows += 1
                    invalid_handle.write(f"{source_line}\t\t{raw_line.strip()}\n")
                    continue

                input_smiles, parent_id = fields[0], fields[1]
                mol = Chem.MolFromSmiles(input_smiles)
                if mol is None:
                    invalid_rows += 1
                    invalid_handle.write(f"{source_line}\t{parent_id}\t{input_smiles}\n")
                    continue

                valid_rows += 1
                if "." in input_smiles:
                    components = Chem.GetMolFrags(mol, asMols=True)
                    if len(components) > 1:
                        multicomponent_rows += 1
                        if multicomponent == "drop":
                            multicomponent_dropped += 1
                            continue
                        if multicomponent == "major":
                            # Ties are common (about a quarter of multi-component rows), so
                            # break them by canonical SMILES rather than by source order.
                            mol = min(
                                components,
                                key=lambda part: (-part.GetNumHeavyAtoms(), Chem.MolToSmiles(part)),
                            )
                canonical_smiles = Chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=True,
                )
                if canonical_smiles in seen:
                    duplicate_rows += 1
                    continue

                seen.add(canonical_smiles)
                output_handle.write(f"{canonical_smiles}\n")
            # A boundary that falls on the last row has no next iteration to report it.
            if source_rows and source_rows % CONVERT_PROGRESS_INTERVAL == 0:
                report()

        os.replace(output_temporary, destination)
        os.replace(invalid_temporary, invalid_destination)
    finally:
        output_temporary.unlink(missing_ok=True)
        invalid_temporary.unlink(missing_ok=True)

    return {
        "source_rows": source_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "duplicate_rows": duplicate_rows,
        "multicomponent_rows": multicomponent_rows,
        "multicomponent_dropped": multicomponent_dropped,
        "output_rows_excluding_header": len(seen),
    }


def _download_archive(url: str, destination: Path, *, timeout: float) -> None:
    temporary = destination.with_name(f".{destination.name}.download")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "SynClaw eMolecules importer"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        _validate_gzip(temporary)
        os.replace(temporary, destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download eMolecules archive from {url}: {exc}") from exc


def _validate_gzip(path: Path) -> None:
    try:
        with gzip.open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                pass
    except (OSError, EOFError, zlib.error) as exc:
        # zlib.error is what a CRC mismatch raises, and it subclasses neither of
        # the others, so a corrupt archive escaped as an unhandled traceback.
        raise RuntimeError(f"Invalid or incomplete gzip archive at {path}: {exc}") from exc


def _check_policy(multicomponent: str) -> None:
    if multicomponent not in MULTICOMPONENT_POLICIES:
        raise RuntimeError(f"multicomponent must be one of {MULTICOMPONENT_POLICIES}")


def _manifest_matches(manifest_path: Path, *, source_sha256: str, output_path: Path, multicomponent: str) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        manifest.get("source_sha256") == source_sha256
        and manifest.get("multicomponent", "keep") == multicomponent
        and manifest.get("output_sha256") == _sha256(output_path)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _emit(progress: ProgressCallback | None, **event: Any) -> None:
    if progress is not None:
        progress(event)
