# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from synclaw.core import db
from synclaw.core.reactions import ingest_reaction_templates
from synclaw.tools.compaction import compact_tool_result
from synclaw.tools.synthesis import SynthesisTools


class DbAndToolsTests(unittest.TestCase):
    def test_reverse_grounding_reuses_identical_precursor_searches(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "synclaw.sqlite"
            tools = SynthesisTools(database)
            result = {
                "candidate_reactant_sets": [
                    {"reactants": ["CO"]},
                    {"reactants": ["CO"]},
                ]
            }
            with db.session(database) as conn, mock.patch(
                "synclaw.tools.synthesis.db.search_building_blocks",
                return_value=[],
            ) as search:
                tools._add_reverse_building_block_matches(
                    conn,
                    result,
                    include_near=True,
                    min_similarity=0.15,
                    top_near_matches=3,
                )
            self.assertEqual(search.call_count, 1)
            self.assertEqual(
                result["candidate_reactant_sets"][0]["reactant_details"],
                result["candidate_reactant_sets"][1]["reactant_details"],
            )

    def test_prepared_search_can_create_lean_copy_and_detect_staleness(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            source.write_text("SMILES\nCCO\nCCN\n", encoding="utf-8")
            database = tmp_path / "source.sqlite"
            prepared = tmp_path / "prepared.sqlite"
            db.ingest_building_blocks(source, database)

            with self.assertRaisesRegex(RuntimeError, "prepare-search"):
                db.require_prepared_building_block_search(database)

            summary = db.prepare_building_block_search(
                database,
                output_path=prepared,
            )
            self.assertEqual(summary["building_block_count"], 2)
            metadata = db.require_prepared_building_block_search(prepared)
            self.assertEqual(metadata["fingerprint_count"], 2)
            with db.session(prepared) as conn:
                conn.execute(f"DELETE FROM {db.BUILDING_BLOCK_FINGERPRINT_TABLE}")
            with self.assertRaisesRegex(RuntimeError, "stale"):
                db.require_prepared_building_block_search(prepared)

            db.prepare_building_block_search(prepared)
            with db.session(prepared) as conn:
                conn.execute(
                    """
                    INSERT INTO building_blocks(
                        smiles, canonical_smiles, heavy_atom_count, created_at
                    ) VALUES ('CCC', 'CCC', 3, ?)
                    """,
                    (db.now_iso(),),
                )
            with self.assertRaisesRegex(RuntimeError, "stale"):
                db.require_prepared_building_block_search(prepared)

    def test_search_hot_path_uses_only_constant_time_index_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            source.write_text("SMILES\nCCO\nCCN\n", encoding="utf-8")
            database = tmp_path / "prepared.sqlite"
            db.ingest_building_blocks(source, database)
            db.prepare_building_block_search(database)

            statements: list[str] = []
            with db.session(database) as conn:
                conn.set_trace_callback(statements.append)
                results = db.search_building_blocks(conn, "CCO", include_near=False)
                conn.set_trace_callback(None)

            self.assertEqual(results[0]["match_type"], "exact")
            traced_sql = "\n".join(statements).upper()
            self.assertIn("BUILDING_BLOCK_SEARCH_METADATA", traced_sql)
            self.assertNotIn("COUNT(", traced_sql)
            self.assertNotIn("MAX(", traced_sql)

    def test_search_rejects_stale_prepared_index_after_catalog_mutations(self):
        mutations = {
            "insert": (
                """
                INSERT INTO building_blocks(
                    smiles, canonical_smiles, heavy_atom_count, created_at
                ) VALUES ('CCC', 'CCC', 3, ?)
                """,
                (db.now_iso(),),
            ),
            "update": (
                "UPDATE building_blocks SET smiles = 'CCC', canonical_smiles = 'CCC' WHERE id = 1",
                (),
            ),
            "delete": ("DELETE FROM building_blocks WHERE id = 1", ()),
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            source.write_text("SMILES\nCCO\nCCN\n", encoding="utf-8")
            for name, (statement, parameters) in mutations.items():
                with self.subTest(mutation=name):
                    database = tmp_path / f"{name}.sqlite"
                    db.ingest_building_blocks(source, database)
                    db.prepare_building_block_search(database)
                    with db.session(database) as conn:
                        conn.execute(statement, parameters)
                    with db.session(database) as conn, self.assertRaisesRegex(
                        RuntimeError,
                        re.escape(f"synclaw db prepare-search --db {database.resolve()}"),
                    ):
                        db.search_building_blocks(conn, "CCO", include_near=False)

    def test_unprepared_database_allows_only_browsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            source.write_text("SMILES\nCCO\n", encoding="utf-8")
            database = tmp_path / "unprepared.sqlite"
            db.ingest_building_blocks(source, database)

            with db.session(database) as conn:
                self.assertEqual(db.search_building_blocks(conn, "*", limit=1)[0]["smiles"], "CCO")
                with self.assertRaisesRegex(RuntimeError, "prepare-search"):
                    db.search_building_blocks(conn, "CCO", include_near=False)

    def test_ingest_search_and_tool_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            templates = tmp_path / "templates.txt"
            source.write_text("SMILES\nCO\nCCN\nCO\n\n", encoding="utf-8")
            templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"

            stats = db.ingest_building_blocks(source, db_path)
            ingest_reaction_templates(templates, db_path=db_path, library="demo")
            db.prepare_building_block_search(db_path)

            self.assertEqual(stats["read"], 3)
            self.assertEqual(stats["inserted"], 2)
            self.assertEqual(stats["duplicates"], 1)

            tools = SynthesisTools(db_path)
            search = tools.search_building_blocks("CO", limit=10)

            self.assertEqual(search["count"], 1)
            self.assertEqual(search["results"][0]["id"], 1)
            self.assertEqual(search["results"][0]["submit_reference"], "BB:1")
            self.assertEqual(search["results"][0]["submit_as"]["preferred"], "BB:1")
            self.assertTrue(search["results"][0]["query_is_exact_building_block"])
            self.assertIn("submit_reference", search["submission_guidance"])

            submission = tools.submit_pathway(
                target_smiles="C=O",
                steps=[
                    {
                        "reaction_id": "demo:R000",
                        "reactants": ["BB:1"],
                        "product_smiles": "C=O",
                    }
                ],
                pathway_text="Oxidize BB:1 to formaldehyde.",
            )

            self.assertIn("reaction_validation", submission)
            self.assertEqual(submission["final_smiles"], "C=O")
            self.assertEqual(submission["similarity_to_target"], 1.0)
            self.assertEqual(submission["steps"][0]["resolved_reactants"][0]["building_block"]["id"], 1)

    def test_reaction_smarts_is_grounded_to_a_library_template_with_the_same_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            templates = tmp_path / "templates.txt"
            source.write_text("SMILES\nCO\n", encoding="utf-8")
            templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"
            db.ingest_building_blocks(source, db_path)
            ingest_reaction_templates(templates, db_path=db_path, library="demo")
            tools = SynthesisTools(db_path)
            oxidation = "[CH3:1][OH:2]>>[CH2:1]=[O:2]"

            submission = tools.submit_pathway(
                target_smiles="C=O",
                steps=[{"reaction_id": oxidation, "reactants": ["BB:1"], "product_smiles": "C=O"}],
                pathway_text="Oxidize methanol.",
            )
            wrong_claim = tools.simulate_reaction(oxidation, ["CO"], product_smiles="COC")
            unknown = tools.simulate_reaction("[C:1][O:2]>>[C:1][O:2]C", ["CO"])
            too_many = tools.simulate_reaction(".".join(["[C:1]"] * 5) + ">>[C:1]", ["CO"] * 5)

            self.assertTrue(submission["accepted"])
            self.assertEqual(submission["steps"][0]["reaction_id"], "demo:R000")
            self.assertEqual(submission["steps"][0]["reaction_smarts"], oxidation)
            self.assertFalse(wrong_claim["valid"])
            self.assertIn("does not generate the claimed product", wrong_claim["issues"][0])
            self.assertFalse(unknown["valid"])
            self.assertIn("No library template generates", unknown["issues"][0])
            self.assertEqual(too_many["issues"], ["A reaction SMARTS may take at most 4 reactants."])

    def test_reaction_smarts_prefers_its_own_template_and_needs_every_unclaimed_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            templates = tmp_path / "templates.txt"
            source.write_text("SMILES\nCO\nCC(O)CO\n", encoding="utf-8")
            templates.write_text(
                "[C:1][O:2]>>[C:1]=[O:2]\n[CH3:1][OH:2]>>[CH2:1]=[O:2]\n", encoding="utf-8"
            )
            primary = tmp_path / "primary.txt"
            primary.write_text("[CH2:1][OH:2]>>[CH1:1]=[O:2]\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"
            db.ingest_building_blocks(source, db_path)
            ingest_reaction_templates(templates, db_path=db_path, library="demo")
            ingest_reaction_templates(primary, db_path=db_path, library="primary")
            tools = SynthesisTools(db_path)
            any_alcohol = "[C:1][OH:2]>>[C:1]=[O:2]"

            own = tools.simulate_reaction("[CH3:1][OH:2]>>[CH2:1]=[O:2]", ["CO"], product_smiles="C=O")
            bad_product = tools.simulate_reaction(any_alcohol, ["CO"], product_smiles="C1C(")
            primary_only = SynthesisTools(db_path, reaction_libraries="primary")
            unclaimed = primary_only.simulate_reaction(any_alcohol, ["CC(O)CO"])
            claimed = primary_only.simulate_reaction(any_alcohol, ["CC(O)CO"], product_smiles="CC(O)C=O")

            self.assertEqual(own["reaction_id"], "demo:R001")
            self.assertEqual(bad_product["issues"], ["product invalid: RDKit could not parse SMILES"])
            self.assertFalse(unclaimed["valid"])
            self.assertIn("No library template generates the same products", unclaimed["issues"][0])
            self.assertTrue(claimed["valid"])
            self.assertEqual(claimed["reaction_id"], "primary:R000")

    def test_missing_building_block_reference_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            templates = tmp_path / "templates.txt"
            source.write_text("SMILES\nCCO\n", encoding="utf-8")
            templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"
            db.ingest_building_blocks(source, db_path)
            ingest_reaction_templates(templates, db_path=db_path, library="demo")

            submission = SynthesisTools(db_path).submit_pathway(
                target_smiles="C=O",
                steps=[
                    {
                        "reaction_id": "demo:R000",
                        "reactants": ["BB:999"],
                        "product_smiles": "C=O",
                    }
                ],
            )

            self.assertFalse(submission["accepted"])
            self.assertIn("unknown building block reference: BB:999", submission["issues"][0])

    def test_a_step_must_name_its_reaction_and_its_reactants(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            templates = tmp_path / "templates.txt"
            source.write_text("SMILES\nCCO\n", encoding="utf-8")
            templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"
            db.ingest_building_blocks(source, db_path)
            ingest_reaction_templates(templates, db_path=db_path, library="demo")
            tools = SynthesisTools(db_path)

            no_reaction = tools.submit_pathway(steps=[{"reactants": ["BB:1"], "product_smiles": "CC=O"}])
            no_reactants = tools.submit_pathway(
                steps=[{"reaction_id": "demo:R000", "product_smiles": "CC=O"}]
            )

            self.assertFalse(no_reaction["accepted"])
            self.assertIn("step 1: reaction_id is required", no_reaction["steps"][0]["issues"])
            self.assertFalse(no_reactants["accepted"])
            self.assertIn("step 1: reactants must be a non-empty list", no_reactants["steps"][0]["issues"])

    def test_every_step_must_feed_the_final_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            templates = tmp_path / "templates.txt"
            source.write_text("SMILES\nCCO\n", encoding="utf-8")
            templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"
            db.ingest_building_blocks(source, db_path)
            ingest_reaction_templates(templates, db_path=db_path, library="demo")
            step = {"reaction_id": "demo:R000", "reactants": ["BB:1"], "product_smiles": "CC=O"}

            submission = SynthesisTools(db_path).submit_pathway(target_smiles="CC=O", steps=[step, step])

            self.assertFalse(submission["accepted"])
            self.assertEqual(submission["reaction_validation"], "valid")
            self.assertIn("INT:1 is not used by a later step", submission["issues"][0])

    def test_a_forward_reference_does_not_count_as_using_an_intermediate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            templates = tmp_path / "templates.txt"
            source.write_text("SMILES\nCCO\n", encoding="utf-8")
            templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"
            db.ingest_building_blocks(source, db_path)
            ingest_reaction_templates(templates, db_path=db_path, library="demo")
            step = {"reaction_id": "demo:R000", "reactants": ["BB:1"], "product_smiles": "CC=O"}
            forward = {**step, "reactants": ["INT:2"]}

            submission = SynthesisTools(db_path).submit_pathway(
                target_smiles="CC=O", steps=[forward, step, step]
            )

            self.assertFalse(submission["accepted"])
            self.assertIn("step 1: unknown intermediate reference: INT:2", submission["issues"])
            self.assertTrue(
                any("INT:2 is not used by a later step" in issue for issue in submission["issues"])
            )

    def test_a_failed_step_is_still_used_by_the_step_that_references_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            templates = tmp_path / "templates.txt"
            source.write_text("SMILES\nCCO\n", encoding="utf-8")
            templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"
            db.ingest_building_blocks(source, db_path)
            ingest_reaction_templates(templates, db_path=db_path, library="demo")
            failed = {"reaction_id": "demo:R999", "reactants": ["BB:1"], "product_smiles": "CC=O"}
            uses_it = {"reaction_id": "demo:R000", "reactants": ["INT:1"], "product_smiles": "CC=O"}

            submission = SynthesisTools(db_path).submit_pathway(
                target_smiles="CC=O", steps=[failed, uses_it]
            )

            self.assertFalse(submission["accepted"])
            self.assertIn("step 2: unknown intermediate reference: INT:1", submission["issues"])
            self.assertFalse(any("is not used by a later step" in issue for issue in submission["issues"]))

    def test_search_returns_near_matches_after_the_exact_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            source.write_text("SMILES\nCCO\nCCN\nCCCCO\nN#CCO\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"
            db.ingest_building_blocks(source, db_path)
            db.prepare_building_block_search(db_path)

            tools = SynthesisTools(db_path)
            exact = tools.search_building_blocks("CCO", limit=3)
            near = tools.search_building_blocks("CCCl", limit=3, include_near=True, min_similarity=0.05)
            no_near = tools.search_building_blocks("CCCl", limit=3, include_near=False)

            self.assertEqual(exact["results"][0]["match_type"], "exact")
            self.assertEqual(exact["results"][0]["similarity"], 1.0)
            self.assertEqual(exact["exact_match_count"], 1)
            self.assertGreaterEqual(len(near["results"]), 1)
            self.assertEqual(near["results"][0]["match_type"], "near")
            self.assertIsNotNone(near["results"][0]["similarity"])
            self.assertFalse(near["results"][0]["query_is_exact_building_block"])
            self.assertIn("Near purchasable", near["results"][0]["submission_note"])
            self.assertGreaterEqual(near["near_match_count"], 1)
            self.assertEqual(no_near["count"], 0)

    def test_search_near_matches_prioritize_shared_smiles_fragments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            source.write_text("SMILES\nO=C1CSC(=O)N1\nCCCCCCCC\nCCO\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"
            db.ingest_building_blocks(source, db_path)
            db.prepare_building_block_search(db_path)

            results = SynthesisTools(db_path).search_building_blocks(
                "C=C1SC(=O)NC1=S",
                limit=2,
                include_near=True,
                min_similarity=0.05,
            )

            self.assertEqual(results["results"][0]["canonical_smiles"], "O=C1CSC(=O)N1")
            self.assertEqual(results["results"][0]["match_type"], "near")

    def test_reverse_reaction_reports_exact_and_near_building_block_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "bbs.txt"
            templates = tmp_path / "templates.txt"
            source.write_text("SMILES\nCO\nCCO\n", encoding="utf-8")
            templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            db_path = tmp_path / "test.sqlite"
            db.ingest_building_blocks(source, db_path)
            ingest_reaction_templates(templates, db_path=db_path, library="demo")
            db.prepare_building_block_search(db_path)

            result = SynthesisTools(db_path).simulate_reverse_reaction(
                "demo:R000",
                "C=O",
                include_near=True,
                min_similarity=0.0,
                top_near_matches=1,
            )

            detail = result["candidate_reactant_sets"][0]["reactant_details"][0]
            self.assertTrue(result["valid"])
            self.assertEqual(result["template_summary"]["reaction_id"], "demo:R000")
            model_view = compact_tool_result("simulate_reverse_reaction", result)
            self.assertEqual(model_view["template_summary"]["reaction_id"], "demo:R000")
            self.assertTrue(result["reverse_planning_summary"]["has_fully_grounded_candidate"])
            self.assertTrue(detail["has_exact_building_block_match"])
            self.assertEqual(detail["exact_building_block_matches"][0]["submit_reference"], "BB:1")
            self.assertLessEqual(len(detail["top_near_building_block_matches"]), 1)
            self.assertNotIn("source_file", detail["exact_building_block_matches"][0])
            self.assertGreaterEqual(result["building_block_match_summary"]["exact_match_count"], 1)
            self.assertIn("planning evidence", result["building_block_match_summary"]["note"])


if __name__ == "__main__":
    unittest.main()
