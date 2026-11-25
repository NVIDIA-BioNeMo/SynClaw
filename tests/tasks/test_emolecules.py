# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem

from synclaw.core import db
from synclaw.tasks.emolecules import (
    convert_emolecules_archive,
    prepare_emolecules_building_blocks,
)

URLOPEN = "synclaw.tasks.emolecules.urllib.request.urlopen"


def _write_gz(path, text):
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def _release(files, requested):
    """A ``urlopen`` stand-in serving the named files, and a 404 for any other name."""

    def serve(request, **_kwargs):
        name = request.full_url.rsplit("/", 1)[1]
        requested.append(name)
        if name not in files:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        return files[name].open("rb")

    return serve


class EmoleculesTests(unittest.TestCase):
    def test_convert_progress_reports_every_boundary_including_the_last_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "parent.smi.gz"
            with gzip.open(source, "wt", encoding="utf-8", newline="\n") as handle:
                handle.write("isosmiles parent_id\nCCO 1\nOCC 2\nnot-a-smiles 3\nCCN 4\n")
            events = []

            with patch("synclaw.tasks.emolecules.CONVERT_PROGRESS_INTERVAL", 2):
                convert_emolecules_archive(
                    source,
                    root / "out.txt",
                    invalid_path=root / "invalid.tsv",
                    progress=events.append,
                )

            # Row 2 is a duplicate and row 4 ends the archive; neither may skip its boundary.
            self.assertEqual([event["source_rows"] for event in events], [2, 4])
            self.assertEqual(
                events[-1],
                {
                    "phase": "convert",
                    "source_rows": 4,
                    "output_rows": 2,
                    "invalid_rows": 1,
                    "duplicate_rows": 1,
                },
            )

    def test_convert_parent_archive_validates_canonicalizes_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "parent.smi.gz"
            output = root / "building_blocks.txt"
            invalid = root / "invalid.tsv"
            with gzip.open(source, "wt", encoding="utf-8", newline="\n") as handle:
                handle.write("isosmiles parent_id\n")
                handle.write("C[C@H](O)F 1\n")
                handle.write("F[C@@H](O)C 2\n")
                handle.write("not-a-smiles 3\n")
                handle.write("OCC 4\n")

            stats = convert_emolecules_archive(
                source,
                output,
                invalid_path=invalid,
            )

            expected_chiral = Chem.MolToSmiles(
                Chem.MolFromSmiles("C[C@H](O)F"),
                canonical=True,
                isomericSmiles=True,
            )
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                ["SMILES", expected_chiral, "CCO"],
            )
            self.assertEqual(
                stats,
                {
                    "source_rows": 4,
                    "valid_rows": 3,
                    "invalid_rows": 1,
                    "duplicate_rows": 1,
                    "multicomponent_rows": 0,
                    "multicomponent_dropped": 0,
                    "output_rows_excluding_header": 2,
                },
            )
            self.assertIn("4\t3\tnot-a-smiles", invalid.read_text(encoding="utf-8"))

            database = root / "smoke.sqlite"
            ingest_stats = db.ingest_building_blocks(output, database)
            search_stats = db.prepare_building_block_search(database)
            self.assertEqual(ingest_stats["inserted"], 2)
            self.assertEqual(ingest_stats["invalid"], 0)
            self.assertEqual(search_stats["building_block_count"], 2)

    def test_multicomponent_policy_applies_before_deduplication(self):
        rows = "isosmiles parent_id\nCCN.Cl 1\nCCN 2\nCl.CCCCO 3\nCCO.CCO 4\n"
        expected = {
            "keep": ["SMILES", "CCN.Cl", "CCN", "CCCCO.Cl", "CCO.CCO"],
            "major": ["SMILES", "CCN", "CCCCO", "CCO"],
            "drop": ["SMILES", "CCN"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "parent.smi.gz"
            with gzip.open(source, "wt", encoding="utf-8", newline="\n") as handle:
                handle.write(rows)
            for policy, lines in expected.items():
                output = root / f"{policy}.txt"
                stats = convert_emolecules_archive(
                    source, output, invalid_path=root / f"{policy}.tsv", multicomponent=policy
                )
                self.assertEqual(output.read_text(encoding="utf-8").splitlines(), lines, policy)
                self.assertEqual(stats["multicomponent_rows"], 3)
                self.assertEqual(stats["multicomponent_dropped"], 3 if policy == "drop" else 0)
                self.assertEqual(
                    stats["valid_rows"],
                    stats["output_rows_excluding_header"]
                    + stats["duplicate_rows"]
                    + stats["multicomponent_dropped"],
                )
            with self.assertRaises(ValueError):
                convert_emolecules_archive(
                    source, root / "x.txt", invalid_path=root / "x.tsv", multicomponent="bogus"
                )

    def test_major_breaks_size_ties_independently_of_component_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = []
            for name, rows in {"a": "CCO.CCN 1\n", "b": "CCN.CCO 1\n"}.items():
                source = root / f"{name}.smi.gz"
                with gzip.open(source, "wt", encoding="utf-8", newline="\n") as handle:
                    handle.write("isosmiles parent_id\n" + rows)
                output = root / f"{name}.txt"
                convert_emolecules_archive(
                    source, output, invalid_path=root / f"{name}.tsv", multicomponent="major"
                )
                outputs.append(output.read_text(encoding="utf-8"))
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[0], "SMILES\nCCN\n")

    def test_prepare_reconverts_when_the_multicomponent_policy_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture.smi.gz"
            with gzip.open(fixture, "wt", encoding="utf-8", newline="\n") as handle:
                handle.write("isosmiles parent_id\nCCN.Cl 1\n")
            options = {
                "dest_root": root / "destination",
                "release": "2026-07-01",
                "source_url": fixture.as_uri(),
            }
            with patch(
                "synclaw.tasks.emolecules.urllib.request.urlopen",
                side_effect=lambda *_args, **_kwargs: fixture.open("rb"),
            ):
                kept = prepare_emolecules_building_blocks(**options, multicomponent="keep")
            self.assertEqual(kept["conversion_status"], "converted")
            self.assertEqual(
                prepare_emolecules_building_blocks(**options, multicomponent="keep")[
                    "conversion_status"
                ],
                "reused",
            )
            major = prepare_emolecules_building_blocks(**options)
            self.assertEqual(major["conversion_status"], "converted")
            self.assertEqual(Path(major["output"]).read_text(encoding="utf-8"), "SMILES\nCCN\n")

    def test_prepare_downloads_file_url_and_reuses_verified_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture.smi.gz"
            with gzip.open(fixture, "wt", encoding="utf-8", newline="\n") as handle:
                handle.write("isosmiles parent_id\nCCO 10\n")

            with patch(
                "synclaw.tasks.emolecules.urllib.request.urlopen",
                side_effect=lambda *_args, **_kwargs: fixture.open("rb"),
            ):
                first = prepare_emolecules_building_blocks(
                    dest_root=root / "destination",
                    release="2026-07-01",
                    source_url=fixture.as_uri(),
                )
            # A matching manifest proves the archive was read in full, so it is not read again.
            with patch(
                "synclaw.tasks.emolecules._validate_gzip",
                side_effect=AssertionError("reused archive was decompressed again"),
            ):
                second = prepare_emolecules_building_blocks(
                    dest_root=root / "destination",
                    release="2026-07-01",
                    source_url=fixture.as_uri(),
                )

            output = Path(first["output"])
            manifest = output.with_name("emolecules_2026-07-01_manifest.json")
            self.assertEqual(output.read_text(encoding="utf-8"), "SMILES\nCCO\n")
            self.assertEqual(first["archive_status"], "downloaded")
            self.assertEqual(first["conversion_status"], "converted")
            self.assertEqual(second["archive_status"], "reused")
            self.assertEqual(second["conversion_status"], "reused")
            self.assertEqual(json.loads(manifest.read_text())["output_rows_excluding_header"], 1)

    def test_a_release_without_a_parent_archive_falls_back_to_the_version_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version = _write_gz(
                root / "version.smi.gz",
                "isosmiles version_id parent_id\nCCN.Cl 1 1\nCCN 2 1\nnot-a-smiles 3 2\n",
            )
            requested = []
            options = {"dest_root": root / "destination", "release": "2026-10-01"}
            with patch(URLOPEN, side_effect=_release({"version.smi.gz": version}, requested)):
                first = prepare_emolecules_building_blocks(**options)
                second = prepare_emolecules_building_blocks(**options)

            self.assertEqual(requested, ["parent.smi.gz", "version.smi.gz", "parent.smi.gz"])
            self.assertEqual(first["archive"], "version")
            self.assertEqual(Path(first["output"]).read_text(encoding="utf-8"), "SMILES\nCCN\n")
            # The second call finds the version archive on disk and converts nothing again.
            self.assertEqual(second["archive_status"], "reused")
            self.assertEqual(second["conversion_status"], "reused")
            invalid = Path(first["invalid_output"]).read_text(encoding="utf-8")
            self.assertTrue(invalid.startswith("source_line\tversion_id\tisosmiles\n"))
            self.assertIn("4\t3\tnot-a-smiles", invalid)

    def test_an_explicit_choice_is_never_replaced_by_another_archive(self):
        requested = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(URLOPEN, side_effect=_release({}, requested)),
        ):
            for archive in ("parent", "version"):
                with self.assertRaisesRegex(RuntimeError, "Could not download"):
                    prepare_emolecules_building_blocks(
                        dest_root=tmp, release="2026-10-01", archive=archive
                    )
            self.assertEqual(
                requested, ["parent.smi.gz", "parent.sdf.gz", "version.smi.gz", "version.sdf.gz"]
            )
            for bad in ({"archive": "bogus"}, {"file_format": "bogus"}):
                with self.assertRaisesRegex(RuntimeError, "must be one of"):
                    prepare_emolecules_building_blocks(dest_root=tmp, release="2026-10-01", **bad)

    def test_a_release_with_only_an_sdf_archive_is_read_from_it(self):
        records = [
            f"{Chem.MolToMolBlock(Chem.MolFromSmiles(smiles))}>  <EMOL_VERSION_ID>\n{i}\n\n$$$$\n"
            for i, smiles in [(1, "CCN.Cl"), (2, "CCN"), (3, "C[C@H](O)F")]
        ]
        # A record whose mol block cannot be read is logged, not fatal.
        records.append("broken\nM  END\n>  <EMOL_VERSION_ID>\n4\n\n$$$$\n")
        requested = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "version.sdf.gz": _write_gz(root / "version.sdf.gz", "".join(records)),
                # Published, but empty.
                "version.smi.gz": _write_gz(root / "stub.gz", "isosmiles version_id parent_id\n"),
            }
            with patch(URLOPEN, side_effect=_release(files, requested)):
                result = prepare_emolecules_building_blocks(
                    dest_root=root / "destination", release="2026-10-01"
                )

            self.assertEqual(
                requested, ["parent.smi.gz", "version.smi.gz", "parent.sdf.gz", "version.sdf.gz"]
            )
            self.assertEqual((result["archive"], result["file_format"]), ("version", "sdf"))
            expected = Chem.MolToSmiles(Chem.MolFromSmiles("C[C@H](O)F"))
            self.assertEqual(
                Path(result["output"]).read_text(encoding="utf-8").splitlines(),
                ["SMILES", "CCN", expected],
            )
            self.assertEqual(
                (result["source_rows"], result["invalid_rows"], result["duplicate_rows"]), (4, 1, 1)
            )
            invalid = Path(result["invalid_output"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(invalid[0], "source_line\tid\tstructure")
            self.assertTrue(invalid[1].endswith("\t4\t<unreadable mol block>"))

    def test_an_archive_with_a_header_but_no_compounds_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_gz(root / "version.smi.gz", "isosmiles version_id parent_id\n")

            with self.assertRaisesRegex(RuntimeError, "no readable compounds"):
                convert_emolecules_archive(source, root / "out.txt", invalid_path=root / "i.tsv")
            self.assertFalse((root / "out.txt").exists())


if __name__ == "__main__":
    unittest.main()
