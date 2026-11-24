# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import tempfile
import unittest
from pathlib import Path

from conftest import numbered_id, numbered_ids

from synclaw.core import db
from synclaw.core.reactions import ingest_reaction_templates, load_template_annotations
from synclaw.paths import REPO_ROOT
from synclaw.tools.synthesis import SynthesisTools

QUERY_ANNOTATIONS = REPO_ROOT / "tests" / "fixtures" / "reaction_template_query_cases.jsonl"
PRODUCTION_ANNOTATIONS = REPO_ROOT / "data" / "reaction_templates" / "synformer" / "templates.jsonl"


class ReactionTemplateTests(unittest.TestCase):
    def test_ingest_and_search_reaction_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.jsonl"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                json.dumps(
                    {
                        "reaction_id": "demo:R001",
                        "name": "Alcohol oxidation",
                        "description": "Convert an alcohol-like reactant pattern to a carbonyl-like product.",
                        "smarts": "[C:1][O:2]>>[C:1]=[O:2]",
                        "tags": ["oxidation", "alcohol"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stats = ingest_reaction_templates(source, db_path=db_path, library="demo")
            result = SynthesisTools(db_path).search_reaction_templates("oxidation", limit=5)

            self.assertEqual(stats["inserted"], 1)
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R001"))
            self.assertIn("alcohol", result["results"][0]["tags"])
            self.assertEqual(result["results"][0]["reactant_count"], 1)

    def test_reaction_template_search_handles_natural_language_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.jsonl"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                json.dumps(
                    {
                        "reaction_id": "demo:R001",
                        "name": "Alcohol oxidation",
                        "description": "Convert an alcohol-like reactant pattern to a carbonyl-like product.",
                        "smarts": "[C:1][O:2]>>[C:1]=[O:2]",
                        "tags": ["oxidation", "alcohol"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).search_reaction_templates(
                "alcohol carbonyl product", limit=5
            )

            self.assertEqual(result["count"], 1)
            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R001"))

    def test_libraries_may_reuse_a_source_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "first.jsonl"
            second = tmp_path / "second.jsonl"
            db_path = tmp_path / "test.sqlite"
            first.write_text(
                json.dumps(
                    {
                        "reaction_id": "R001",
                        "name": "First library oxidation",
                        "description": "First library template.",
                        "smarts": "[C:1][O:2]>>[C:1]=[O:2]",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "reaction_id": "R001",
                        "name": "Second library amination",
                        "description": "Second library template.",
                        "smarts": "[C:1][Cl:2].[N:3]>>[C:1][N:3]",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            ingest_reaction_templates(first, db_path=db_path, library="first")
            ingest_reaction_templates(second, db_path=db_path, library="second")

            ids = numbered_ids(db_path)
            self.assertEqual(sorted(ids), ["first:R001", "second:R001"])
            self.assertEqual(sorted(ids.values()), ["R001", "R002"])

    def test_raw_template_import_does_not_infer_reaction_query_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[Cl,Br,I][C$(C([CX4,c])([CX4,c])([CX4,c])),"
                "C$([CH]([CX4,c])([CX4,c])),C$([CH2]([CX4,c])),"
                "C$([CH3]),c$(c):1]>>[N]#[C][#6:1]\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).search_reaction_templates(
                "cyanide substitution nitrile", limit=5
            )
            template = SynthesisTools(db_path).get_reaction_template(
                numbered_id(db_path, "demo:R000")
            )["template"]

            self.assertEqual(result["count"], 0)
            self.assertEqual(template["tags"], [])
            self.assertEqual(template["reaction_family"], "")
            self.assertTrue(
                template["description"].startswith(
                    "Template with 1 reactant pattern(s) and 1 product pattern(s)."
                )
            )
            self.assertNotIn("nitrile", template["description"].lower())

    def test_explicit_template_annotations_replace_inferred_search_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            annotations = tmp_path / "annotations.jsonl"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[Cl,Br,I][C$(C([CX4,c])([CX4,c])([CX4,c])),"
                "C$([CH]([CX4,c])([CX4,c])),C$([CH2]([CX4,c])),"
                "C$([CH3]),c$(c):1]>>[N]#[C][#6:1]\n",
                encoding="utf-8",
            )
            annotations.write_text(
                json.dumps(
                    {
                        "library": "demo",
                        "reaction_id": "demo:R000",
                        "name": "Curated nitrile installation",
                        "family": "curated nitrile formation",
                        "description": "Curated description for model-facing template discovery.",
                        "search_terms": ["custom cyanation cue"],
                        "retro_planning_notes": "Disconnect the nitrile to a curated halide precursor.",
                        "execution_caveats": ["Curated caveat."],
                        "confidence": "high",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stats = ingest_reaction_templates(
                source,
                db_path=db_path,
                library="demo",
                annotations_path=annotations,
            )
            result = SynthesisTools(db_path).search_reaction_templates(
                "custom cyanation cue",
                limit=5,
            )

            self.assertEqual(stats["annotated"], 1)
            template = result["results"][0]
            self.assertEqual(template["name"], "Curated nitrile installation")
            self.assertEqual(
                template["description"],
                "Curated description for model-facing template discovery.",
            )
            self.assertEqual(template["reaction_family"], "curated nitrile formation")
            self.assertEqual(template["query_hints"], ["custom cyanation cue"])
            self.assertEqual(template["reactant_roles"], [])
            self.assertNotIn("cyanide substitution", template["tags"])
            self.assertIn("Curated caveat.", template["execution_hints"])

    def test_annotation_validation_rejects_duplicate_and_inconsistent_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotations = Path(tmp) / "annotations.jsonl"
            base = {
                "library": "demo",
                "reaction_id": "demo:R000",
                "family": "test family",
                "family_metadata": {
                    "description": "Shared description.",
                    "search_terms": ["shared term"],
                },
            }
            annotations.write_text(
                json.dumps(base) + "\n" + json.dumps(base) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate annotation"):
                load_template_annotations(annotations, library="demo")

            second = {
                **base,
                "reaction_id": "demo:R001",
                "family_metadata": {
                    "description": "Different description.",
                    "search_terms": ["shared term"],
                },
            }
            annotations.write_text(
                json.dumps(base) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inconsistent family_metadata"):
                load_template_annotations(annotations, library="demo")

    def test_annotation_validation_rejects_malformed_new_fields(self):
        invalid_values = (
            ({"search_exclusions": "not-a-list"}, "search_exclusions"),
            ({"reactant_roles": [{"position": 0, "role": "invalid"}]}, "position"),
            ({"overview": {"priority": 1}}, "require a category"),
            ({"replace_tags": "yes"}, "replace_tags must be a boolean"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            annotations = Path(tmp) / "annotations.jsonl"
            for extra, expected_error in invalid_values:
                with self.subTest(extra=extra):
                    annotations.write_text(
                        json.dumps({"reaction_id": "demo:R000", **extra}) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        load_template_annotations(annotations, library="demo")

    def test_production_annotations_cover_reviewed_search_misses(self):
        expected_ids = {
            "chloroformate": {"synformer:R009"},
            "aldol": {
                "synformer:R003",
                "synformer:R010",
            },
            "amidation": {"synformer:R008"},
            "quinazoline": {"synformer:R035"},
            "SNAr": {"synformer:R103"},
            "enone": {
                "synformer:R003",
                "synformer:R010",
            },
            "1,4-diketone": {"synformer:R090"},
            "transesterification": {"synformer:R072"},
            "buchwald-hartwig": {"synformer:R103"},
            "chan-lam": {"synformer:R099"},
            "paal-knorr": {"synformer:R090"},
            "hell-volhard-zelinsky": {
                "synformer:R111",
                "synformer:R112",
            },
            "Suzuki-Miyaura": {"synformer:R050"},
            "Negishi": {"synformer:R095"},
            "Fischer-Speier esterification": {"synformer:R072"},
            "Wittig olefination": {"synformer:R102"},
            "Williamson etherification": {"synformer:R056"},
        }
        annotations = load_template_annotations(
            PRODUCTION_ANNOTATIONS,
            library="synformer",
        )
        selected_ids = set().union(*expected_ids.values())

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            records = []
            for number, reaction_id in enumerate(sorted(selected_ids), start=1):
                record = dict(annotations[("synformer", reaction_id)])
                record.pop("reactant_roles", None)
                record["smarts"] = f"[*:{number}]>>[*:{number}]"
                records.append(record)
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="synformer")
            tools = SynthesisTools(db_path)
            ids = numbered_ids(db_path)

            for query, expected in expected_ids.items():
                with self.subTest(query=query):
                    result = tools.search_reaction_templates(query, limit=10)
                    result_ids = {item["reaction_id"] for item in result["results"]}
                    self.assertTrue({ids[source_id] for source_id in expected} <= result_ids)

    def test_production_annotations_classify_isocyanate_amine_as_urea_formation(self):
        annotations = load_template_annotations(
            PRODUCTION_ANNOTATIONS,
            library="synformer",
        )
        record = annotations[("synformer", "synformer:R106")]

        self.assertEqual(record["name"], "Urea formation from amine and isocyanate")
        self.assertEqual(record["family"], "urea formation")
        self.assertIn("isocyanate plus amine", record["search_terms"])
        self.assertNotIn("imine addition", record["search_terms"])

    def test_annotation_roles_must_cover_template_reactant_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.txt"
            annotations = Path(tmp) / "annotations.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            source.write_text("[C:1].[N:2]>>[C:1][N:2]\n", encoding="utf-8")
            annotations.write_text(
                json.dumps(
                    {
                        "reaction_id": "demo:R000",
                        "reactant_roles": [{"position": 1, "role": "first reactant"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "do not match template positions"):
                ingest_reaction_templates(
                    source,
                    db_path=db_path,
                    library="demo",
                    annotations_path=annotations,
                )

    def test_structured_import_uses_inline_annotation_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            source.write_text(
                json.dumps(
                    {
                        "reaction_id": "R001",
                        "reaction_smarts": "[C:1][O:2]>>[C:1]=[O:2]",
                        "name": "Curated inline template",
                        "description": "Curated inline description.",
                        "family": "curated family",
                        "family_metadata": {
                            "description": "Curated family description.",
                            "search_terms": ["curated family query"],
                        },
                        "search_terms": ["exclusive alias phrase"],
                        "reactant_roles": [{"position": 1, "role": "curated reactant"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            ingest_reaction_templates(source, db_path=db_path, library="demo")
            result = SynthesisTools(db_path).search_reaction_templates("exclusive alias phrase")

            self.assertEqual(result["results"][0]["reaction_family"], "curated family")
            self.assertEqual(result["results"][0]["reactant_roles"][0]["role"], "curated reactant")

    def test_inline_annotation_preserves_structured_source_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            source.write_text(
                json.dumps(
                    {
                        "reaction_id": "R001",
                        "reaction_smarts": "[C:1]>>[C:1]",
                        "tags": ["suzuki"],
                        "confidence": "high",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            ingest_reaction_templates(source, db_path=db_path, library="demo")
            result = SynthesisTools(db_path).search_reaction_templates("suzuki")

            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R001"))
            self.assertIn("suzuki", result["results"][0]["tags"])

    def test_external_annotation_replaces_structured_source_tags_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.jsonl"
            annotations = Path(tmp) / "annotations.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            source.write_text(
                json.dumps(
                    {
                        "reaction_id": "R001",
                        "reaction_smarts": "[C:1].[N:2]>>[C:1][N:2]",
                        "family": "suzuki coupling",
                        "tags": ["suzuki"],
                        "search_terms": ["suzuki coupling"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            annotations.write_text(
                json.dumps(
                    {
                        "reaction_id": "demo:R001",
                        "family": "amide formation",
                        "search_terms": ["amide coupling"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            ingest_reaction_templates(
                source,
                db_path=db_path,
                library="demo",
                annotations_path=annotations,
            )
            tools = SynthesisTools(db_path)

            self.assertEqual(tools.search_reaction_templates("suzuki")["count"], 0)
            result = tools.search_reaction_templates("amide coupling")
            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R001"))
            self.assertNotIn("suzuki", result["results"][0]["tags"])

    def test_csv_inline_annotation_fields_accept_json_and_blank_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.csv"
            db_path = Path(tmp) / "test.sqlite"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("reaction_id", "reaction_smarts", "search_terms", "confidence"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "reaction_id": "R001",
                        "reaction_smarts": "[C:1]>>[C:1]",
                        "search_terms": json.dumps(["curated oxidation"]),
                        "confidence": "high",
                    }
                )
                writer.writerow(
                    {
                        "reaction_id": "R002",
                        "reaction_smarts": "[N:1]>>[N:1]",
                        "search_terms": "",
                        "confidence": "",
                    }
                )

            stats = ingest_reaction_templates(source, db_path=db_path, library="demo")
            result = SynthesisTools(db_path).search_reaction_templates("curated oxidation")

            self.assertEqual(stats["inserted"], 2)
            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R001"))

    def test_bracket_led_inline_annotation_text_falls_back_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            note = "[3+2] cycloaddition disconnection works here"
            source.write_text(
                json.dumps(
                    {
                        "reaction_id": "R001",
                        "reaction_smarts": "[C:1]>>[C:1]",
                        "search_terms": "[3+2] cycloaddition, dipolar",
                        "retro_planning_notes": note,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            ingest_reaction_templates(source, db_path=db_path, library="demo")
            tools = SynthesisTools(db_path)
            result = tools.search_reaction_templates("dipolar")
            template = tools.get_reaction_template(numbered_id(db_path, "demo:R001"))["template"]

            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R001"))
            self.assertIn("[3+2] cycloaddition", template["query_hints"])
            self.assertIn(note, template["execution_hints"])

    def test_invalid_smarts_skip_inline_reactant_role_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            source.write_text(
                json.dumps(
                    {
                        "reaction_id": "R001",
                        "reaction_smarts": "nosmarts",
                        "reactant_roles": [
                            {"position": 1, "role": "first"},
                            {"position": 2, "role": "second"},
                        ],
                    }
                )
                + "\n"
                + json.dumps({"reaction_id": "R002", "smarts": "[C:1>>[C:1]"})
                + "\n"
                + json.dumps({"reaction_id": "R003", "smarts": ">>"})
                + "\n",
                encoding="utf-8",
            )

            stats = ingest_reaction_templates(source, db_path=db_path, library="demo")

            self.assertEqual(stats["records"], 3)
            self.assertEqual(stats["invalid"], 3)
            self.assertEqual(stats["inserted"], 0)

    def test_text_search_weights_rare_terms_above_common_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            names = ["coupling coupling coupling", "coupling", "coupling", "suzuki", "coupling"]
            source.write_text(
                "".join(
                    json.dumps(
                        {
                            "reaction_id": f"R{index:03d}",
                            "smarts": f"[C:{index + 1}]>>[C:{index + 1}]",
                            "name": name,
                        }
                    )
                    + "\n"
                    for index, name in enumerate(names)
                ),
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).search_reaction_templates("suzuki coupling")

            # "suzuki" occurs in one template and "coupling" in four, so the rare term wins.
            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R003"))
            self.assertEqual(result["results"][1]["reaction_id"], numbered_id(db_path, "demo:R000"))

    def test_text_search_caches_its_index_and_ignores_noise_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            source.write_text(
                json.dumps(
                    {
                        "reaction_id": "R001",
                        "reaction_smarts": "[C:1]-[N:2]>>[C:1]-[N:2]",
                        "name": "Curated imine-like transformation",
                        "search_terms": ["imine conversion"],
                        "confidence": "medium",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "reaction_id": "R002",
                        "reaction_smarts": "[C:3]-[N:4]>>[C:3]-[N:4]",
                        "name": "Plain transformation",
                        "search_terms": ["imine conversion"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            ingest_reaction_templates(source, db_path=db_path, library="demo")
            tools = SynthesisTools(db_path)
            tools.search_reaction_templates("imine conversion")
            before = db._cached_reaction_text_index.cache_info()
            tools.search_reaction_templates("imine conversion")
            self.assertEqual(db._cached_reaction_text_index.cache_info().hits, before.hits + 1)
            self.assertEqual(tools.search_reaction_templates("medium")["count"], 0)
            # Atom-map digits, element letters and stopwords must not match every template.
            self.assertEqual(tools.search_reaction_templates("zzz 1")["count"], 0)
            self.assertEqual(tools.search_reaction_templates("make a thing with the N")["count"], 0)
            self.assertEqual(
                tools.search_reaction_templates("imines")["results"][0]["reaction_id"],
                numbered_id(db_path, "demo:R001"),
            )

    def test_smarts_queries_match_exact_and_partial_template_smarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            smarts = [
                "[C:1](=[O:2])[OH].[NH2:3]>>[C:1](=[O:2])[NH:3]",
                "[c:1][Br].[NH2:2]>>[c:1][NH:2]",
                "[C:1](=[O:2])[OH].[OH:3]>>[C:1](=[O:2])[O:3]",
            ]
            source.write_text(
                "".join(
                    json.dumps(
                        {"reaction_id": f"R{index:03d}", "smarts": value, "name": "coupling"}
                    )
                    + "\n"
                    for index, value in enumerate(smarts)
                ),
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            tools = SynthesisTools(db_path)

            exact = tools.search_reaction_templates(smarts[2])["results"]
            partial = tools.search_reaction_templates("[c:7][Br]")["results"]

            self.assertEqual(exact[0]["reaction_id"], numbered_id(db_path, "demo:R002"))
            # Atom-map numbers are ignored, so the aryl bromide atoms find R001 alone.
            self.assertEqual(
                [item["reaction_id"] for item in partial], [numbered_id(db_path, "demo:R001")]
            )

    def test_search_phrase_matching_preserves_repeats_and_terms_after_twelve(self):
        repeated = db._normalize_reaction_search_phrase("amine to amine coupling")
        long_query = db._normalize_reaction_search_phrase(
            "one two three four five six seven eight nine ten eleven twelve thirteen amine coupling"
        )

        self.assertTrue(db._search_phrase_is_contained(repeated, "amine coupling"))
        self.assertTrue(db._search_phrase_is_contained(long_query, "amine coupling"))

    def test_raw_template_import_uses_zero_based_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "".join(f"[C:{n}][O:99]>>[C:{n}]=[O:99]\n" for n in range(1, 45))
                + "[N:1].[C:2]>>[N:1][C:2]\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).get_reaction_template(
                numbered_id(db_path, "demo:R044")
            )

            self.assertEqual(result["template"]["smarts"], "[N:1].[C:2]>>[N:1][C:2]")

    def test_structured_template_import_keeps_one_based_generated_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.jsonl"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                json.dumps({"reaction_smarts": "[C:1][O:2]>>[C:1]=[O:2]"}) + "\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            self.assertEqual(numbered_ids(db_path), {"demo:R001": "R001"})

    def test_annotation_supplies_query_language_for_raw_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[NH2;$([N][#6]);!$([N]C=[O,S,N]):1]."
                "[NX3;!$(N[C,S]=[S,O,N]);$(N[#6]);!$(N[#7]);H1,H2:2]"
                ">>[N:1][C](=[O])[N:2]\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(
                source,
                db_path=db_path,
                library="demo",
                annotations_path=QUERY_ANNOTATIONS,
            )

            result = SynthesisTools(db_path).search_reaction_templates(
                "urea amine coupling", limit=5
            )

            self.assertEqual(result["count"], 1)
            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R000"))
            self.assertIn("urea formation", result["results"][0]["tags"])
            self.assertIn("amine coupling", result["results"][0]["tags"])
            self.assertIn("urea", result["results"][0]["description"])
            self.assertEqual(
                result["results"][0]["reactant_roles"][0]["role"],
                "primary amine-like reactant",
            )

    def test_amide_search_prefers_carboxylic_acid_amine_template_over_urea(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "\n".join(
                    [
                        "[NH2;$([N][#6]);!$([N]C=[O,S,N]):1]."
                        "[NX3;!$(N[C,S]=[S,O,N]);$(N[#6]);!$(N[#7]);H1,H2:2]"
                        ">>[N:1][C](=[O])[N:2]",
                        "[NX3;!$(N[C]=[S,O,N]);$(N[#6,S]);!$(N[#7]);H1,H2:1]."
                        "[#6:4][C:5](=[O:6])[OH,O-]>>[N:1][C:5](=[O:6])[#6:4]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(
                source,
                db_path=db_path,
                library="demo",
                annotations_path=QUERY_ANNOTATIONS,
            )

            result = SynthesisTools(db_path).search_reaction_templates(
                "amide coupling carboxylic acid amine",
                limit=2,
            )

            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R001"))
            self.assertEqual(result["results"][0]["reaction_family"], "amide formation")
            self.assertIn("carboxylic acid plus amine", result["results"][0]["query_hints"])
            self.assertEqual(result["results"][1]["reaction_family"], "urea formation")
            for record in db.iter_jsonl(QUERY_ANNOTATIONS):
                for query in record.get("evaluation_queries", []):
                    ranked = SynthesisTools(db_path).search_reaction_templates(query, limit=3)
                    self.assertIn(
                        numbered_id(db_path, record["reaction_id"]),
                        [item["reaction_id"] for item in ranked["results"]],
                    )

    def test_reaction_library_overview_groups_templates_by_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[NH2;$([N][#6]);!$([N]C=[O,S,N]):1]."
                "[NX3;!$(N[C,S]=[S,O,N]);$(N[#6]);!$(N[#7]);H1,H2:2]"
                ">>[N:1][C](=[O])[N:2]\n"
                "[NX3;!$(N[C]=[S,O,N]);$(N[#6,S]);!$(N[#7]);H1,H2:1]."
                "[#6:4][C:5](=[O:6])[OH,O-]>>[N:1][C:5](=[O:6])[#6:4]\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(
                source,
                db_path=db_path,
                library="demo",
                annotations_path=QUERY_ANNOTATIONS,
            )

            with db.session(db_path) as conn:
                overview = db.reaction_template_library_overview(conn)

            self.assertEqual(overview["template_count"], 2)
            self.assertEqual(overview["families"][0]["family"], "amide formation")
            self.assertIn("amide coupling", overview["families"][0]["query_hints"])

    def test_reaction_library_overview_prefers_later_family_metadata_to_personal_hints(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.jsonl"
            db_path = Path(tmp) / "test.sqlite"
            records = [
                {
                    "reaction_id": "R001",
                    "reaction_smarts": "[C:1]>>[C:1]",
                    "family": "shared family",
                    "search_terms": ["personal first-template hint"],
                },
                {
                    "reaction_id": "R002",
                    "reaction_smarts": "[N:1]>>[N:1]",
                    "family": "shared family",
                    "search_terms": ["personal second-template hint"],
                    "family_metadata": {
                        "description": "Curated family description.",
                        "search_terms": ["curated family hint"],
                    },
                },
            ]
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            ingest_reaction_templates(source, db_path=db_path, library="demo")
            with db.session(db_path) as conn:
                overview = db.reaction_template_library_overview(conn)

            self.assertEqual(overview["families"][0]["query_hints"], ["curated family hint"])

    def test_reaction_library_overview_surfaces_notable_rare_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            annotations = tmp_path / "annotations.jsonl"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[#6:1][C:2](=O)[#6:3].[#6:4][NH:5]>>[#6:4][N:5][C:2]([#6:1])([#6:3])C(=O)O\n",
                encoding="utf-8",
            )
            annotations.write_text(
                json.dumps(
                    {
                        "library": "demo",
                        "reaction_id": "demo:R000",
                        "name": "Ketone addition with quaternary carboxylation",
                        "family": "abstract carbonyl transformation",
                        "description": (
                            "Forms a heteroatom-substituted quaternary carboxylic acid "
                            "using an abstract carboxylation step."
                        ),
                        "search_terms": [
                            "quaternary carboxylic acid",
                            "ketone nucleophile addition",
                            "abstract carbonyl transformation",
                        ],
                        "execution_caveats": [
                            "Abstract template introduces carboxylic acid atoms not supplied by reactants."
                        ],
                        "family_metadata": {
                            "description": "Abstract carbonyl additions with implicit product atoms.",
                            "search_terms": ["abstract carbonyl transformation"],
                        },
                        "overview": {
                            "priority": 70,
                            "category": "quaternary carboxylation",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(
                source, db_path=db_path, library="demo", annotations_path=annotations
            )

            with db.session(db_path) as conn:
                overview = db.reaction_template_library_overview(conn)

            self.assertEqual(
                overview["notable_templates"][0]["name"],
                "Ketone addition with quaternary carboxylation",
            )
            self.assertIn(
                "quaternary carboxylic acid", overview["notable_templates"][0]["query_hints"]
            )

    def test_simulate_reaction_checks_claimed_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).simulate_reaction(
                numbered_id(db_path, "demo:R000"),
                ["CO"],
                product_smiles="C=O",
            )

            self.assertTrue(result["valid"])
            self.assertTrue(result["product_matched"])
            self.assertIn("C=O", result["generated_products"])

    def test_simulate_reaction_canonicalizes_generated_products_before_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[#6:1][C:2](=O)[#6:3].[#6:4][OH,nH,NH:5]"
                ">>[#6:4][*:5][C:2]([#6:1])([#6:3])C(=O)O\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).simulate_reaction(
                numbered_id(db_path, "demo:R000"),
                ["CC(C)=O", "c1[nH]ccn1"],
                product_smiles="CC(C)(C(=O)O)n1ccnc1",
            )

            self.assertTrue(result["valid"])
            self.assertTrue(result["product_matched"])
            self.assertEqual(result["generated_products"], ["CC(C)(C(=O)O)n1ccnc1"])
            self.assertEqual(result["generated_product_sets"], [["CC(C)(C(=O)O)n1ccnc1"]])

    def test_scoring_needs_a_target_not_just_a_valid_looking_validation(self):
        # target_smiles and target_validation are separate parameters so a caller can
        # reuse a validation it already computed. A mismatched pair must not score
        # against the string "None".
        from synclaw.tools.synthesis import score_pathway_against_target

        scored = score_pathway_against_target(
            ["CCO"],
            target_smiles=None,
            target_validation={"valid": True, "canonical_smiles": "CCO"},
            preferred_smiles="CCO",
            accepted=True,
        )

        self.assertEqual(scored["final_smiles"], "CCO")
        self.assertIsNone(scored["similarity_to_target"])
        self.assertIsNone(scored["identical_to_target"])

    def test_claiming_a_product_makes_an_ungenerated_step_invalid(self):
        # submit_pathway always supplies product_smiles, and that is what makes step
        # validity imply a generated product: without a claim there is nothing to
        # contradict and the step validates with no products at all. The reduced
        # scoring gates in submit_pathway rely on this.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1]>>[C:1](F)(F)(F)(F)(F)\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            tools = SynthesisTools(db_path)

            unclaimed = tools.simulate_reaction(numbered_id(db_path, "demo:R000"), ["C"])
            claimed = tools.simulate_reaction(
                numbered_id(db_path, "demo:R000"), ["C"], product_smiles="CF"
            )

            self.assertEqual(unclaimed["generated_products"], [])
            self.assertTrue(unclaimed["valid"])
            self.assertEqual(claimed["generated_products"], [])
            self.assertFalse(claimed["valid"])

    def test_simulate_reaction_skips_unsanitizable_generated_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1]>>[C:1](F)(F)(F)(F)(F)\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).simulate_reaction(
                numbered_id(db_path, "demo:R000"),
                ["C"],
            )

            self.assertTrue(result["reaction_executed"])
            self.assertEqual(result["generated_products"], [])
            self.assertEqual(result["generated_product_sets"], [])
            self.assertEqual(result["skipped_invalid_product_set_count"], 1)

    def test_simulate_reaction_returns_generated_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).simulate_reaction(
                numbered_id(db_path, "demo:R000"),
                ["CO"],
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["expected_reactant_count"], 1)
            self.assertEqual(result["provided_reactant_count"], 1)
            self.assertEqual(result["generated_products"], ["C=O"])
            self.assertEqual(result["generated_product_sets"], [["C=O"]])

    def test_simulate_reaction_preserves_multi_product_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2].O\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).simulate_reaction(
                numbered_id(db_path, "demo:R000"),
                ["CO"],
                product_smiles="C=O",
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["generated_products"], ["C=O", "O"])
            self.assertEqual(result["generated_product_sets"], [["C=O", "O"]])
            self.assertEqual(result["generated_product_set_count"], 1)

    def test_simulate_reaction_tries_alternate_reactant_orders_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[NH2;$([N][#6]);!$([N]C=[O,S,N]):1]."
                "[NX3;!$(N[C,S]=[S,O,N]);$(N[#6]);!$(N[#7]);H1,H2:2]"
                ">>[N:1][C](=[O])[N:2]\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).simulate_reaction(
                numbered_id(db_path, "demo:R000"),
                ["N#CC1CC12CNC2", "Nc1nccc2occc12"],
            )

            self.assertTrue(result["valid"])
            self.assertTrue(result["reactant_order_permuted"])
            self.assertEqual(result["used_reactant_order"], [2, 1])
            self.assertIn("N#CC1CC12CN(C(=O)Nc1nccc3occc13)C2", result["generated_products"])

    def test_simulate_reaction_reports_reactant_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).simulate_reaction(
                numbered_id(db_path, "demo:R000"),
                ["CO", "CCO"],
            )

            self.assertFalse(result["valid"])
            self.assertEqual(result["expected_reactant_count"], 1)
            self.assertEqual(result["provided_reactant_count"], 2)
            self.assertIn("Reactant count mismatch", result["issues"][0])

    def test_simulate_reverse_reaction_returns_candidate_reactants(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.prepare_building_block_search(db_path)

            result = SynthesisTools(db_path).simulate_reverse_reaction(
                numbered_id(db_path, "demo:R000"),
                "C=O",
                include_near=False,
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["candidate_reactant_sets"][0]["reactants"], ["CO"])
            self.assertFalse(result["candidate_reactant_sets"][0]["all_precursors_grounded"])

    def test_simulate_reverse_reaction_preserves_reactant_tuples(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[N:1].[C:2](=[O:3])O>>[N:1][C:2](=[O:3])\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.prepare_building_block_search(db_path)

            result = SynthesisTools(db_path).simulate_reverse_reaction(
                numbered_id(db_path, "demo:R000"),
                "NC=O",
                include_near=False,
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["candidate_reactant_sets"][0]["reactants"], ["N", "O=CO"])
            self.assertEqual(result["candidate_reactant_sets"][0]["reactant_count"], 2)

    def test_simulate_reverse_reaction_accepts_multi_product_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2].O\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.prepare_building_block_search(db_path)

            result = SynthesisTools(db_path).simulate_reverse_reaction(
                numbered_id(db_path, "demo:R000"),
                ["C=O", "O"],
                include_near=False,
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["expected_product_count"], 2)
            self.assertEqual(result["candidate_reactant_sets"][0]["reactants"], ["CO"])
            self.assertEqual(result["product_canonical_smiles"], ["C=O", "O"])

    def test_parenthesized_disconnected_product_counts_as_one_rdkit_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                (
                    "[NX3;!$(N[C,S]=[S,O,N]);$(N[#6]);!$(N[#7]);H1,H2:1]."
                    "[#6:4][C:5](=[O:6])[O:7][C:8](=[O:9])[#6:10]"
                    ">>([N:1][C:5](=[O:6])[#6:4].[OH:7][C:8](=[O:9])[#6:10])\n"
                ),
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.prepare_building_block_search(db_path)
            tools = SynthesisTools(db_path)

            template = tools.get_reaction_template(numbered_id(db_path, "demo:R000"))["template"]
            result = tools.simulate_reverse_reaction(
                numbered_id(db_path, "demo:R000"),
                "[N-]=[N+]=NCC1C[N+](C(=O)C(CC(=O)O)C2CCCO2)=C(N)N1c1cccc(F)c1",
                include_near=False,
            )

            self.assertEqual(template["product_count"], 1)
            self.assertIn("1 product pattern(s)", template["description"])
            self.assertTrue(result["valid"])
            self.assertEqual(result["expected_product_count"], 1)
            self.assertIn(
                "O=C1CC(C2CCCO2)C(=O)O1",
                result["candidate_reactant_sets"][0]["reactants"],
            )

    def test_simulate_reverse_reaction_requires_all_products_for_multi_product_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2].O\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).simulate_reverse_reaction(
                numbered_id(db_path, "demo:R000"),
                "C=O",
                include_near=False,
            )

            self.assertFalse(result["valid"])
            self.assertIn("Product count mismatch", result["issues"][0])
            self.assertEqual(result["product_template_count"], 2)

    def test_simulate_reverse_reaction_filters_unsanitizable_candidate_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[NX3;!$(N[C,S]=[S,O,N]);$(N[#6]);!$(N[#7]);H1,H2:1].[#6:3][CX3;!$(C[!#6]);H0,H1:4]=[O]>>[N:1][C:4][#6:3]\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).simulate_reverse_reaction(
                numbered_id(db_path, "demo:R000"),
                "NC(C)=O",
                include_near=False,
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["candidate_reactant_sets"], [])
            self.assertEqual(result["skipped_invalid_candidate_set_count"], 1)
            self.assertIn(
                "RDKit did not produce any valid reverse precursor sets", result["issues"][0]
            )
            self.assertIn("does not prove the template cannot work forward", result["issues"][0])

    def test_search_reaction_templates_matches_molecule_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.jsonl"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "reaction_id": "demo:R001",
                                "name": "Alcohol oxidation",
                                "description": "Convert alcohol to carbonyl.",
                                "smarts": "[C:1][O:2]>>[C:1]=[O:2]",
                            }
                        ),
                        json.dumps(
                            {
                                "reaction_id": "demo:R002",
                                "name": "Amide formation",
                                "description": "Couple amine and acid.",
                                "smarts": "[N:1].[C:2](=[O:3])O>>[N:1][C:2](=[O:3])",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.prepare_building_block_search(db_path)

            result = SynthesisTools(db_path).search_reaction_templates(
                reactant_smiles=["N", "O=CO"],
                product_smiles="NC=O",
                run_forward=True,
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R002"))
            structure = result["results"][0]["structure_match"]
            self.assertTrue(structure["reactant_match"])
            self.assertTrue(structure["product_match"])
            self.assertTrue(structure["forward_simulation"]["requested_products_generated"]["NC=O"])

    def test_search_reaction_templates_forward_check_uses_canonical_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[#6:1][C:2](=O)[#6:3].[#6:4][OH,nH,NH:5]"
                ">>[#6:4][*:5][C:2]([#6:1])([#6:3])C(=O)O\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.prepare_building_block_search(db_path)

            result = SynthesisTools(db_path).search_reaction_templates(
                reactant_smiles=["CC(C)=O", "c1[nH]ccn1"],
                product_smiles="CC(C)(C(=O)O)n1ccnc1",
                run_forward=True,
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["count"], 1)
            forward = result["results"][0]["structure_match"]["forward_simulation"]
            self.assertTrue(forward["requested_products_generated"]["CC(C)(C(=O)O)n1ccnc1"])
            self.assertTrue(forward["requested_product_set_generated"])

    def test_search_simulates_the_matched_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][OH:2]>>[C:1][O:2]C\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            with db.session(db_path) as conn:
                with self.assertRaisesRegex(ValueError, "needs a query"):
                    db.search_reaction_templates(conn, " ", limit=1)
            db.prepare_building_block_search(db_path)
            tools = SynthesisTools(db_path)

            forward = tools.search_reaction_templates(reactant_smiles=["CO"], run_forward=True)
            reverse = tools.search_reaction_templates(product_smiles="COC")

            match = forward["results"][0]["structure_match"]
            self.assertEqual(match["forward_simulation"]["generated_products"], ["COC"])
            usefulness = reverse["results"][0]["reverse_usefulness"]
            self.assertEqual(usefulness["best_candidate_reactants"], ["CO"])

    def test_search_reaction_templates_ranks_by_reverse_usefulness(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.jsonl"
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "reaction_id": "demo:R001",
                                "name": "Carbonyl identity",
                                "description": "Structurally matches carbonyl product but does not split to stock.",
                                "smarts": "[C:1]=[O:2]>>[C:1]=[O:2]",
                            }
                        ),
                        json.dumps(
                            {
                                "reaction_id": "demo:R002",
                                "name": "Alcohol oxidation",
                                "description": "Reverse split gives alcohol.",
                                "smarts": "[C:1][O:2]>>[C:1]=[O:2]",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            bb_source.write_text("SMILES\nCO\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.ingest_building_blocks(bb_source, db_path)
            db.prepare_building_block_search(db_path)

            result = SynthesisTools(db_path).search_reaction_templates(
                product_smiles="C=O",
            )

            self.assertTrue(result["valid"])
            self.assertTrue(result["reverse_ranking"]["applied"])
            self.assertEqual(result["results"][0]["reaction_id"], numbered_id(db_path, "demo:R002"))
            usefulness = result["results"][0]["reverse_usefulness"]
            self.assertGreater(usefulness["score"], 100)
            self.assertEqual(usefulness["fully_grounded_candidate_sets"], 1)
            self.assertEqual(usefulness["best_candidate_exact_submit_references"], ["BB:1"])

    def test_search_reaction_templates_returns_text_and_molecule_results_side_by_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.jsonl"
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "reaction_id": "demo:R001",
                                "name": "Carbonyl identity",
                                "description": "Structurally ranked first by template ID.",
                                "smarts": "[C:1]=[O:2]>>[C:1]=[O:2]",
                            }
                        ),
                        json.dumps(
                            {
                                "reaction_id": "demo:R002",
                                "name": "Alcohol oxidation",
                                "description": "Reverse split gives alcohol.",
                                "smarts": "[C:1][O:2]>>[C:1]=[O:2]",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            bb_source.write_text("SMILES\nCO\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.ingest_building_blocks(bb_source, db_path)
            db.prepare_building_block_search(db_path)

            result = SynthesisTools(db_path).search_reaction_templates(
                query="alcohol oxidation",
                product_smiles="C=O",
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["search_mode"], "text_and_molecule")
            # Both templates fit the product; the grounded alcohol split ranks first.
            self.assertNotIn("results", result)
            self.assertEqual(
                [item["reaction_id"] for item in result["structural_results"]],
                [numbered_id(db_path, "demo:R002"), numbered_id(db_path, "demo:R001")],
            )
            self.assertEqual(result["structural_count"], len(result["structural_results"]))
            self.assertEqual(
                result["text_results"][0]["reaction_id"], numbered_id(db_path, "demo:R002")
            )
            self.assertEqual(result["text_count"], len(result["text_results"]))

    def test_search_reaction_templates_text_list_is_independent_of_molecule_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.jsonl"
            db_path = tmp_path / "test.sqlite"
            records = []
            for index in range(121):
                records.append(
                    {
                        "reaction_id": f"demo:R{index:03d}",
                        "name": "Special carbonyl identity"
                        if index == 120
                        else "Carbonyl identity",
                        "description": "Structurally matches carbonyl product.",
                        "smarts": f"[C:{index + 1}]=[O:999]>>[C:{index + 1}]=[O:999]",
                    }
                )
            source.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.prepare_building_block_search(db_path)

            result = SynthesisTools(db_path).search_reaction_templates(
                query="special",
                product_smiles="C=O",
                limit=5,
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["search_mode"], "text_and_molecule")
            self.assertEqual(result["matched_template_count"], 121)
            self.assertEqual(result["structural_count"], 5)
            self.assertEqual(
                result["text_results"][0]["reaction_id"], numbered_id(db_path, "demo:R120")
            )

    def test_search_reaction_templates_records_reverse_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.jsonl"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                json.dumps(
                    {
                        "reaction_id": "demo:R001",
                        "name": "Two product carbonyl template",
                        "description": "Partial product-side match but reverse requires both products.",
                        "smarts": "[C:1][O:2]>>[C:1]=[O:2].[OH2]",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).search_reaction_templates(
                product_smiles="C=O",
            )

            self.assertTrue(result["valid"])
            self.assertTrue(result["reverse_ranking"]["applied"])
            self.assertEqual(result["count"], 1)
            usefulness = result["results"][0]["reverse_usefulness"]
            self.assertFalse(usefulness["reverse_executed"])
            self.assertLess(usefulness["score"], 0)
            self.assertIn("Product count mismatch", usefulness["issues"][0])

    def test_search_reaction_templates_does_not_reverse_rank_reactant_only_searches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).search_reaction_templates(
                reactant_smiles=["CO"],
            )

            self.assertTrue(result["valid"])
            self.assertFalse(result["reverse_ranking"]["applied"])
            self.assertEqual(result["reverse_ranking"]["reason"], "product_smiles was not supplied")
            self.assertNotIn("reverse_usefulness", result["results"][0])

    def test_simulate_reverse_reaction_reports_invalid_product_and_unknown_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            tools = SynthesisTools(db_path)

            invalid_product = tools.simulate_reverse_reaction(
                numbered_id(db_path, "demo:R000"), "not-smiles"
            )
            unknown_template = tools.simulate_reverse_reaction("R999", "C=O")

            self.assertFalse(invalid_product["valid"])
            self.assertIn("product invalid", invalid_product["issues"][0])
            self.assertFalse(unknown_template["valid"])
            self.assertIn("Unknown reaction template", unknown_template["issues"][0])

    def test_simulate_reverse_reaction_limits_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1].[C:2]>>[C:1][C:2]\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.prepare_building_block_search(db_path)

            result = SynthesisTools(db_path).simulate_reverse_reaction(
                numbered_id(db_path, "demo:R000"),
                "CCCC",
                max_outcomes=1,
                include_near=False,
            )

            self.assertTrue(result["valid"])
            self.assertEqual(len(result["candidate_reactant_sets"]), 1)
            self.assertEqual(result["returned_candidate_count"], 1)

    def test_submit_pathway_requires_reaction_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            bb_source.write_text("SMILES\nCO\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                target_smiles="C=O",
                steps=[
                    {
                        "reaction_id": numbered_id(db_path, "demo:R000"),
                        "reactants": ["BB:1"],
                        "product_smiles": "C=O",
                    }
                ],
            )

            self.assertIn("reaction_validation", result)
            self.assertEqual(result["reaction_validation"], "valid")
            self.assertTrue(result["accepted"])
            self.assertIsNone(result["error_message"])
            self.assertEqual(result["final_smiles"], "C=O")
            self.assertEqual(result["actual_final_smiles"], "C=O")

    def test_submit_pathway_accepts_canonicalized_generated_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[#6:1][C:2](=O)[#6:3].[#6:4][OH,nH,NH:5]"
                ">>[#6:4][*:5][C:2]([#6:1])([#6:3])C(=O)O\n",
                encoding="utf-8",
            )
            bb_source.write_text("SMILES\nCC(C)=O\nc1[nH]ccn1\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                target_smiles="CC(C)(C(=O)O)n1ccnc1",
                steps=[
                    {
                        "reaction_id": numbered_id(db_path, "demo:R000"),
                        "reactants": ["BB:1", "BB:2"],
                        "product_smiles": "CC(C)(C(=O)O)n1ccnc1",
                    }
                ],
                pathway_text="Add imidazole to acetone under the abstract carboxylating template.",
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(result["reaction_validation"], "valid")
            self.assertEqual(result["actual_final_smiles"], "CC(C)(C(=O)O)n1ccnc1")
            self.assertEqual(result["final_smiles"], "CC(C)(C(=O)O)n1ccnc1")
            self.assertEqual(result["similarity_to_target"], 1.0)

    def test_submit_pathway_can_validate_targetless_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            bb_source.write_text("SMILES\nCO\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                steps=[
                    {
                        "reaction_id": numbered_id(db_path, "demo:R000"),
                        "reactants": ["BB:1"],
                        "product_smiles": "C=O",
                    }
                ],
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(result["final_smiles"], "C=O")
            self.assertIsNone(result["target"])
            self.assertIsNone(result["similarity_to_target"])

    def test_submit_pathway_accepts_zero_step_building_block_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            db.init_db(db_path)
            bb_source.write_text("SMILES\nCO\n", encoding="utf-8")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                steps=[],
                building_block="BB:1",
                target_smiles="CO",
                pathway_text="The target is directly available as a building block.",
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(result["pathway_type"], "direct_purchase")
            self.assertEqual(result["reaction_validation"], "valid")
            self.assertEqual(result["final_smiles"], "CO")
            self.assertEqual(result["submitted_final_smiles"], "CO")
            self.assertEqual(result["similarity_to_target"], 1.0)
            self.assertTrue(result["building_block_grounded"])
            self.assertEqual(result["referenced_building_block_ids"], [1])
            self.assertEqual(result["steps"], [])
            self.assertEqual(result["direct_building_block"]["submit_reference"], "BB:1")

    def test_submit_pathway_accepts_zero_step_building_block_by_exact_smiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            db.init_db(db_path)
            bb_source.write_text("SMILES\nOC\n", encoding="utf-8")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                building_block="CO",
                target_smiles="CO",
                pathway_text="The target is directly available as a building block.",
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(result["final_smiles"], "CO")
            self.assertEqual(result["referenced_building_block_ids"], [1])

    def test_submit_pathway_rejects_zero_step_without_building_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite"
            db.init_db(db_path)

            result = SynthesisTools(db_path).submit_pathway(
                steps=[],
                target_smiles="CO",
                pathway_text="No reaction is needed.",
            )

            self.assertFalse(result["accepted"])
            self.assertEqual(result["pathway_type"], "direct_purchase")
            self.assertEqual(result["reaction_validation"], "invalid")
            self.assertIn("zero-step submission requires building_block", result["issues"])
            self.assertIn("direct-purchase submission was rejected", result["error_message"])

    def test_submit_pathway_rejects_zero_step_unavailable_smiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            db.init_db(db_path)
            bb_source.write_text("SMILES\nCC\n", encoding="utf-8")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                steps=[],
                building_block="CO",
                target_smiles="CO",
                pathway_text="Try a direct-purchase molecule that is not in stock.",
            )

            self.assertFalse(result["accepted"])
            self.assertFalse(result["building_block_grounded"])
            self.assertEqual(result["referenced_building_block_ids"], [])
            self.assertIn("not an exact building-block match", result["issues"][0])

    def test_submit_pathway_rejects_building_block_with_reaction_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            bb_source.write_text("SMILES\nCO\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                steps=[
                    {
                        "reaction_id": numbered_id(db_path, "demo:R000"),
                        "reactants": ["BB:1"],
                        "product_smiles": "C=O",
                    }
                ],
                building_block="BB:1",
                target_smiles="C=O",
                pathway_text="Ambiguous direct-purchase plus reaction route.",
            )

            self.assertFalse(result["accepted"])
            self.assertEqual(result["final_smiles"], "C=O")
            self.assertTrue(
                any(
                    "building_block is only valid for zero-step" in issue
                    for issue in result["issues"]
                )
            )

    def test_submit_pathway_accepts_product_generated_by_permuted_reactant_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[NH2;$([N][#6]);!$([N]C=[O,S,N]):1]."
                "[NX3;!$(N[C,S]=[S,O,N]);$(N[#6]);!$(N[#7]);H1,H2:2]"
                ">>[N:1][C](=[O])[N:2]\n",
                encoding="utf-8",
            )
            bb_source.write_text("SMILES\nN#CC1CC12CNC2\nNc1nccc2occc12\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                target_smiles="N#CC1CC12CN(C(=O)Nc1nccc3occc13)C2",
                steps=[
                    {
                        "reaction_id": numbered_id(db_path, "demo:R000"),
                        "reactants": ["BB:1", "BB:2"],
                        "product_smiles": "N#CC1CC12CN(C(=O)Nc1nccc3occc13)C2",
                    }
                ],
            )

            self.assertTrue(result["accepted"])
            self.assertTrue(result["steps"][0]["reactant_order_permuted"])
            self.assertEqual(result["steps"][0]["used_reactant_order"], [2, 1])

    def test_submit_pathway_does_not_score_rejected_rdkit_generated_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            bb_source.write_text("SMILES\nCO\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                target_smiles="C=O",
                steps=[
                    {
                        "reaction_id": numbered_id(db_path, "demo:R000"),
                        "reactants": ["CO"],
                        "product_smiles": "CO",
                    }
                ],
                pathway_text="Claim methanol remains unchanged.",
            )

            self.assertFalse(result["accepted"])
            self.assertEqual(result["reaction_validation"], "invalid")
            self.assertEqual(result["submitted_final_smiles"], "CO")
            self.assertEqual(result["final_smiles"], "C=O")
            self.assertIsNone(result["similarity_to_target"])
            self.assertIsNone(result["constitutional_similarity_to_target"])
            self.assertIn("rejected", result["error_message"])
            self.assertTrue(result["building_block_grounded"])
            self.assertEqual(result["referenced_building_block_ids"], [1])

    def test_submit_pathway_prefers_best_stereo_final_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1]>>F[C@H:1](Cl)Br.F[C@@H:1](Cl)Br\n", encoding="utf-8")
            bb_source.write_text("SMILES\nC\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                target_smiles="F[C@H](Cl)Br",
                steps=[
                    {
                        "reaction_id": numbered_id(db_path, "demo:R000"),
                        "reactants": ["BB:1"],
                        "product_smiles": "F[C@@H](Cl)Br",
                    }
                ],
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(result["submitted_final_smiles"], "F[C@@H](Cl)Br")
            self.assertEqual(result["actual_final_smiles"], "F[C@H](Cl)Br")
            self.assertEqual(result["actual_final_products"], ["F[C@@H](Cl)Br", "F[C@H](Cl)Br"])
            self.assertEqual(result["constitutional_similarity_to_target"], 1.0)
            self.assertEqual(result["similarity_to_target"], 1.0)

    def test_submit_pathway_has_no_similarity_without_rdkit_final_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            bb_source = tmp_path / "bbs.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            bb_source.write_text("SMILES\nCO\nCCO\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")
            db.ingest_building_blocks(bb_source, db_path)

            result = SynthesisTools(db_path).submit_pathway(
                target_smiles="C=O",
                steps=[
                    {
                        "reaction_id": numbered_id(db_path, "demo:R000"),
                        "reactants": ["CO", "CCO"],
                        "product_smiles": "C=O",
                    }
                ],
                pathway_text="Use the wrong reactant count.",
            )

            self.assertFalse(result["accepted"])
            self.assertEqual(result["reaction_validation"], "invalid")
            self.assertEqual(result["submitted_final_smiles"], "C=O")
            self.assertIsNone(result["final_smiles"])
            self.assertIsNone(result["similarity_to_target"])
            self.assertIn("RDKit did not produce", result["error_message"])

    def test_submit_pathway_rejects_raw_smiles_not_in_building_block_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            db_path = tmp_path / "test.sqlite"
            source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            ingest_reaction_templates(source, db_path=db_path, library="demo")

            result = SynthesisTools(db_path).submit_pathway(
                target_smiles="C=O",
                steps=[
                    {
                        "reaction_id": numbered_id(db_path, "demo:R000"),
                        "reactants": ["CO"],
                        "product_smiles": "C=O",
                    }
                ],
                pathway_text="Try an ungrounded raw reactant.",
            )

            self.assertFalse(result["accepted"])
            self.assertFalse(result["building_block_grounded"])
            self.assertIsNone(result["final_smiles"])
            self.assertIn("not an exact building-block match", result["issues"][0])
            self.assertIn("purchasable building block", result["error_message"])
            self.assertIn("not directly purchasable", result["error_message"])
            self.assertIn("INT:<step_number>", result["error_message"])
            self.assertIn("pivot", result["error_message"])


if __name__ == "__main__":
    unittest.main()
