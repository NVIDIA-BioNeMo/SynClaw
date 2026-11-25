# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download and normalize the eMolecules orderable building-block catalog."""

from __future__ import annotations

import gzip
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
# InChIKey per molecule, which this catalog-sized stream discards.
from rdkit import Chem, RDLogger, rdBase

from synclaw.core.structures import (
    apply_multicomponent,
    check_multicomponent_policy,
    open_structures,
)
from synclaw.tasks.matrix import sha256_file

EMOLECULES_DOWNLOAD_ROOT = "https://downloads.emolecules.com/orderbb"
# "parent" is the supplier's compound with salts and solvates stripped; "version" keeps them. The
# SMILES files are small and the SDF files about twenty times larger. Unless told otherwise, the
# first of parent.smi, version.smi, parent.sdf and version.sdf that exists and holds compounds is
# used.
ARCHIVE_KINDS = ("parent", "version")
ARCHIVE_FORMATS = ("smi", "sdf")
_RELEASE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

ProgressCallback = Callable[[dict[str, Any]], None]


class ArchiveNotFound(RuntimeError):
    """The release has no such archive (HTTP 404)."""


class EmptyArchive(RuntimeError):
    """The archive holds no compounds."""


CONVERT_PROGRESS_INTERVAL = 100_000
RDLogger.DisableLog("rdApp.*")


def prepare_emolecules_building_blocks(
    *,
    dest_root: str | Path = ".",
    release: str,
    archive: str | None = None,
    file_format: str | None = None,
    source_url: str | None = None,
    multicomponent: str = "major",
    force: bool = False,
    timeout: float = 60.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Download and convert an eMolecules archive for ``synclaw ingest-bbs``.

    ``archive`` picks ``parent`` or ``version`` (the supplier's compounds with salts and solvates
    kept) and ``file_format`` picks ``smi`` or ``sdf``. Whatever is left unset is tried in turn:
    parent before version, and the SMILES files before the much larger SDF files. A candidate
    that the release does not have (HTTP 404) or that holds no compounds is skipped for the next;
    the last one's failure is raised. ``source_url`` replaces the download URL and tries nothing
    else.

    The derived file contains a ``SMILES`` header followed by one canonical
    isomeric SMILES per line. The IDs are intentionally omitted because the
    BioNeMo-SynClaw importer expects the complete line to be a SMILES value.
    ``multicomponent`` sets how salts and mixtures are handled: ``keep`` them as
    shipped, reduce each to its ``major`` (largest by heavy atoms) component, or
    ``drop`` them.

    Raises:
        RuntimeError: A bad release, archive or policy; a download failure; or no archive with
            compounds.
    """

    check_multicomponent_policy(multicomponent)
    if _RELEASE_PATTERN.fullmatch(release) is None:
        raise RuntimeError("eMolecules release must use YYYY-MM-DD format")
    if archive is not None and archive not in ARCHIVE_KINDS:
        raise RuntimeError(f"eMolecules archive must be one of {', '.join(ARCHIVE_KINDS)}")
    if file_format is not None and file_format not in ARCHIVE_FORMATS:
        raise RuntimeError(f"eMolecules file format must be one of {', '.join(ARCHIVE_FORMATS)}")

    if source_url:
        kinds = (archive or "parent",)
        formats = (file_format or ("sdf" if source_url.endswith(".sdf.gz") else "smi"),)
    else:
        kinds = (archive,) if archive else ARCHIVE_KINDS
        formats = (file_format,) if file_format else ARCHIVE_FORMATS
    candidates = [(kind, fmt) for fmt in formats for kind in kinds]
    destination_dir = Path(dest_root) / "data" / "building_blocks" / "emolecules"
    destination_dir.mkdir(parents=True, exist_ok=True)

    for kind, fmt in candidates:
        try:
            return _prepare_archive(
                destination_dir,
                release=release,
                kind=kind,
                fmt=fmt,
                url=source_url or f"{EMOLECULES_DOWNLOAD_ROOT}/{release}/{kind}.{fmt}.gz",
                multicomponent=multicomponent,
                force=force,
                timeout=timeout,
                progress=progress,
            )
        except (ArchiveNotFound, EmptyArchive) as exc:
            if (kind, fmt) == candidates[-1]:
                raise
            _emit(progress, phase="skipped", archive=f"{kind}.{fmt}.gz", reason=str(exc))
    raise AssertionError("unreachable")  # candidates is never empty


def _prepare_archive(
    destination_dir: Path,
    *,
    release: str,
    kind: str,
    fmt: str,
    url: str,
    multicomponent: str,
    force: bool,
    timeout: float,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    archive_path = destination_dir / f"{kind}_{release}.{fmt}.gz"
    output_path = destination_dir / f"emolecules_{release}.txt"
    invalid_path = destination_dir / f"emolecules_{release}_invalid.tsv"
    manifest_path = destination_dir / f"emolecules_{release}_manifest.json"

    downloaded = False
    if force or not archive_path.exists():
        _download_archive(url, archive_path, timeout=timeout)
        downloaded = True
        _emit(progress, phase="downloaded", path=str(archive_path), url=url)
    else:
        _emit(progress, phase="reused_archive", path=str(archive_path), url=url)

    # A manifest that matches this archive's hash means it was already read in full and converted.
    source_sha256 = sha256_file(archive_path)
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

    if not downloaded:
        _validate_gzip(archive_path)  # A download was validated before it replaced the archive.
    stats = convert_emolecules_archive(
        archive_path,
        output_path,
        invalid_path=invalid_path,
        multicomponent=multicomponent,
        progress=progress,
    )
    manifest = {
        "release": release,
        "archive": kind,
        "file_format": fmt,
        "source_url": url,
        "source_archive": str(archive_path),
        "source_sha256": source_sha256,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
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


def convert_emolecules_archive(
    archive_path: str | Path,
    output_path: str | Path,
    *,
    invalid_path: str | Path,
    multicomponent: str = "major",
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """Convert an eMolecules ``.smi.gz`` or ``.sdf.gz`` archive to one validated, de-duplicated SMILES per line.

    A SMILES row is a SMILES followed by its IDs (a parent ID, or a version ID and a parent ID).
    An SDF record is a mol block with ``EMOL_VERSION_ID`` and ``EMOL_PARENT_ID`` tags. The
    multicomponent policy applies before canonical de-duplication, so a salt reduced
    to a component that is already a row counts as a duplicate.

    Raises:
        EmptyArchive: The archive has no compounds.
        RuntimeError: The archive is malformed.
    """

    check_multicomponent_policy(multicomponent)
    source = Path(archive_path)
    destination = Path(output_path)
    invalid_destination = Path(invalid_path)
    # No pre-scan here: callers validate the archive before converting it, and the streaming
    # read below raises on a bad CRC of its own accord.
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
            open_structures(source) as (
                id_column,
                structure_column,
                rows,
            ),
            output_temporary.open("w", encoding="utf-8", newline="\n") as output_handle,
            invalid_temporary.open("w", encoding="utf-8", newline="\n") as invalid_handle,
        ):
            output_handle.write("SMILES\n")
            invalid_handle.write(f"source_line\t{id_column}\t{structure_column}\n")

            def report() -> None:
                _emit(
                    progress,
                    phase="convert",
                    source_rows=source_rows,
                    output_rows=len(seen),
                    invalid_rows=invalid_rows,
                    duplicate_rows=duplicate_rows,
                )

            for source_line, source_id, mol, text in rows:
                # Report before the row is classified so invalid and duplicate rows,
                # which `continue` below, cannot skip a boundary.
                if source_rows and source_rows % CONVERT_PROGRESS_INTERVAL == 0:
                    report()
                source_rows += 1
                if mol is None:
                    invalid_rows += 1
                    invalid_handle.write(f"{source_line}\t{source_id}\t{text}\n")
                    continue

                valid_rows += 1
                mol, multicomponent_row = apply_multicomponent(mol, multicomponent)
                multicomponent_rows += multicomponent_row
                if mol is None:
                    multicomponent_dropped += 1
                    continue
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
            if valid_rows == 0:
                raise EmptyArchive(f"{source} has no readable compounds")

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
        failure = ArchiveNotFound if getattr(exc, "code", None) == 404 else RuntimeError
        raise failure(f"Could not download eMolecules archive from {url}: {exc}") from exc


def _validate_gzip(path: Path) -> None:
    try:
        with gzip.open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                pass
    except (OSError, EOFError, zlib.error) as exc:
        # zlib.error is what a CRC mismatch raises, and it subclasses neither of
        # the others, so a corrupt archive escaped as an unhandled traceback.
        raise RuntimeError(f"Invalid or incomplete gzip archive at {path}: {exc}") from exc


def _manifest_matches(
    manifest_path: Path, *, source_sha256: str, output_path: Path, multicomponent: str
) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        manifest.get("source_sha256") == source_sha256
        and manifest.get("multicomponent") == multicomponent
        and manifest.get("output_sha256") == sha256_file(output_path)
    )


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
