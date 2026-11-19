# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem

from synclaw.core import db
from synclaw.tasks.emolecules import (
    convert_emolecules_parent_archive,
    prepare_emolecules_building_blocks,
)


class EmoleculesTests(unittest.TestCase):
    def test_convert_progress_reports_every_boundary_including_the_last_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "parent.smi.gz"
            with gzip.open(source, "wt", encoding="utf-8", newline="\n") as handle:
                handle.write("isosmiles parent_id\nCCO 1\nOCC 2\nnot-a-smiles 3\nCCN 4\n")
            events = []

            with patch("synclaw.tasks.emolecules.CONVERT_PROGRESS_INTERVAL", 2):
                convert_emolecules_parent_archive(
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

            stats = convert_emolecules_parent_archive(
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
                stats = convert_emolecules_parent_archive(
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
            with self.assertRaises(RuntimeError):
                convert_emolecules_parent_archive(
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
                convert_emolecules_parent_archive(
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
            second = prepare_emolecules_building_blocks(
                dest_root=root / "destination",
                release="2026-07-01",
                source_url=fixture.as_uri(),
            )

            output = Path(first["output"])
            manifest = output.with_name("emolecules_parent_2026-07-01_manifest.json")
            self.assertEqual(output.read_text(encoding="utf-8"), "SMILES\nCCO\n")
            self.assertEqual(first["archive_status"], "downloaded")
            self.assertEqual(first["conversion_status"], "converted")
            self.assertEqual(second["archive_status"], "reused")
            self.assertEqual(second["conversion_status"], "reused")
            self.assertEqual(json.loads(manifest.read_text())["output_rows_excluding_header"], 1)


if __name__ == "__main__":
    unittest.main()
