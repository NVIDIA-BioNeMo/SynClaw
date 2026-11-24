# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import json
import tempfile
import unittest
from pathlib import Path

from conftest import prepare_minimal_model_db

from synclaw.core import db, reactions
from synclaw.evaluation.evaluate import evaluate_trace_file, summarize_events
from synclaw.evaluation.results import combine_summary_csvs
from synclaw.paths import REPO_ROOT
from synclaw.rollout import policy
from synclaw.rollout.loop import call_local_tool
from synclaw.rollout.policy import (
    coerce_tool_arguments,
    parse_arguments,
    submit_tool_choice,
)
from synclaw.rollout.prompts import build_initial_messages
from synclaw.tasks.datasets import (
    TOOL_NAMES,
    build_llm_task,
    generate_llm_taskset,
)
from synclaw.tasks.public_data import list_public_data_files
from synclaw.tools.contracts import (
    TOOL_SCHEMAS,
    filter_chat_tool_schemas,
    filter_messages_tool_schemas,
    normalize_excluded_tools,
)
from synclaw.tools.synthesis import SynthesisTools


def _call_event(arguments, target):
    """A submit_pathway call event as the rollout loop records it."""

    call = policy.prepare_tool_call(
        raw_name="submit_pathway",
        raw_arguments=json.dumps(arguments),
        raw_tool_call={},
        event_index=0,
    )
    return {
        "event_type": "tool_call",
        "tool_name": "submit_pathway",
        "target_smiles": target,
        "payload": call.event_payload,
    }


def _result_event(result, target):
    """A submit_pathway result event as the rollout loop records it."""

    return {
        "event_type": "tool_result",
        "tool_name": "submit_pathway",
        "target_smiles": target,
        "payload": {"result": result, "outcome": "ok"},
    }


class ImportAndEvalTests(unittest.TestCase):
    def test_generate_llm_taskset_from_smiles_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "targets.txt"
            output = tmp_path / "tasks.jsonl"
            source.write_text("SMILES\nCCO\nCCN\n", encoding="utf-8")

            stats = generate_llm_taskset(source, output, dataset="fixture")

            self.assertEqual(stats["written"], 2)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["task_id"], "fixture-0001")
            self.assertEqual(records[0]["target_smiles"], "CCO")
            self.assertEqual(records[0]["question"], "Target molecule: CCO")
            self.assertEqual(
                records[0]["available_tools"],
                [
                    "search_building_blocks",
                    "similarity",
                    "mces",
                    "search_reaction_templates",
                    "simulate_reaction",
                    "simulate_reverse_reaction",
                    "submit_pathway",
                ],
            )

    def test_render_model_context_help_uses_model_runner_tool_call_default(self):
        import subprocess
        import sys

        script = REPO_ROOT / "scripts" / "render_model_context.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-tool-calls", result.stdout)
        self.assertIn("--max-turns", result.stdout)

    def test_evaluate_trace_does_not_score_submitted_final_without_actual_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "eval.sqlite"
            trace = tmp_path / "trace.jsonl"
            prepare_minimal_model_db(db_path)
            # Oxidizing methanol cannot make ethanol, so the run rejects this route.
            arguments = {
                "steps": [{"reaction_id": "R001", "reactants": ["BB:1"], "product_smiles": "CCO"}],
                "pathway_text": "Claim an invalid route to ethanol.",
            }
            result = SynthesisTools(db_path).submit_pathway(**arguments, target_smiles="CCO")
            events = [
                {
                    "event_type": "message",
                    "role": "user",
                    "target_smiles": "CCO",
                    "payload": {"text": "Plan CCO"},
                },
                _call_event(arguments, "CCO"),
                _result_event(result, "CCO"),
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
            )

            summary = evaluate_trace_file(trace, db_path, model="mock-model")

            self.assertTrue(summary["has_final_submission"])
            self.assertIsNone(summary["final_smiles"])
            self.assertEqual(summary["submitted_final_smiles"], "CCO")
            self.assertFalse(summary["accepted"])
            self.assertEqual(summary["reaction_validation"], "invalid")
            self.assertIsNone(summary["similarity_to_target"])

    def test_evaluate_trace_does_not_score_rejected_actual_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "eval.sqlite"
            trace = tmp_path / "trace.jsonl"
            events = [
                {
                    "event_type": "message",
                    "role": "user",
                    "target_smiles": "CCO",
                    "payload": {"text": "Plan CCO"},
                },
                {
                    "event_type": "selected_submission",
                    "tool_name": "submit_pathway",
                    "payload": {
                        "reason": "no_accepted_submission",
                        "result": {
                            "accepted": False,
                            "reaction_validation": "invalid",
                            "pathway_text": "A rejected route that produced the target.",
                            "actual_final_smiles": "CCO",
                            "final_smiles": "CCO",
                            "submitted_final_smiles": "CCO",
                            "constitutional_similarity_to_target": 1.0,
                            "similarity_to_target": 1.0,
                            "steps": [{"valid": True, "actual_product_smiles": "CCO"}],
                        },
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            summary = evaluate_trace_file(trace, db_path, model="mock-model")

            self.assertFalse(summary["accepted"])
            self.assertIsNone(summary["final_smiles"])
            self.assertIsNone(summary["similarity_to_target"])
            self.assertIsNone(summary["constitutional_similarity_to_target"])

    def test_evaluate_trace_uses_last_submit_pathway_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "eval.sqlite"
            templates = tmp_path / "templates.txt"
            bbs = tmp_path / "bbs.txt"
            trace = tmp_path / "trace.jsonl"
            templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            bbs.write_text("SMILES\nCO\nCCO\n", encoding="utf-8")
            reactions.ingest_reaction_templates(templates, db_path=db_path, library="demo")
            db.ingest_building_blocks(bbs, db_path)
            first = {
                "steps": [{"reaction_id": "R001", "reactants": ["CO"], "product_smiles": "C=O"}],
                "pathway_text": "First candidate.",
            }
            # Template IDs are numbered per database; the recorded SMARTS says which one ran.
            second = {
                "steps": [{"reaction_id": "R033", "reactants": ["CCO"], "product_smiles": "CC=O"}],
                "pathway_text": "Second candidate.",
            }
            recorded = {
                "steps": [{"step_index": 1, "template": {"smarts": "[C:1][O:2]>>[C:1]=[O:2]"}}]
            }
            message = {
                "event_type": "message",
                "role": "user",
                "target_smiles": "CC=O",
                "payload": {"text": "Plan CC=O"},
            }
            events = [
                message,
                _call_event(first, "CC=O"),
                _result_event(recorded, "CC=O"),
                _call_event(second, "CC=O"),
                _result_event(recorded, "CC=O"),
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
            )

            summary = evaluate_trace_file(trace, db_path, model="mock-model")

            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["final_smiles"], "CC=O")
            self.assertEqual(summary["similarity_to_target"], 1.0)

            # Without a recorded SMARTS, the ID cannot name a template, so the run's verdict stands.
            rejected = {
                "accepted": False,
                "steps": [{"step_index": 1, "valid": False}],
                "issues": ["step 1: Unknown reaction template: R001"],
            }
            as_smarts = {
                **first,
                "steps": [{**first["steps"][0], "reaction_id": "[C:1][O:2]>>[C:1]=[O:2]"}],
            }
            for call in (first, as_smarts):
                events = [message, _call_event(call, "CC=O"), _result_event(rejected, "CC=O")]
                trace.write_text(
                    "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
                )
                summary = evaluate_trace_file(trace, db_path, model="mock-model")
                self.assertFalse(summary["accepted"])

    def test_evaluate_trace_scores_an_enantiomer_with_stereo_aware_similarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "eval.sqlite"
            trace = tmp_path / "trace.jsonl"
            events = [
                {
                    "event_type": "message",
                    "role": "user",
                    "target_smiles": "F[C@H](Cl)Br",
                    "payload": {"text": "Plan F[C@H](Cl)Br"},
                },
                {
                    "event_type": "tool_result",
                    "tool_name": "submit_pathway",
                    "payload": {
                        "outcome": "ok",
                        "result": {
                            "accepted": True,
                            "reaction_validation": "valid",
                            "pathway_text": "Opposite enantiomer.",
                            "actual_final_smiles": "F[C@@H](Cl)Br",
                            "final_smiles": "F[C@@H](Cl)Br",
                            "submitted_final_smiles": "F[C@@H](Cl)Br",
                            "steps": [{"valid": True, "actual_product_smiles": "F[C@@H](Cl)Br"}],
                        },
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
            )

            summary = evaluate_trace_file(trace, db_path, model="mock-model")

            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["constitutional_similarity_to_target"], 1.0)
            self.assertLess(summary["similarity_to_target"], 1.0)

    def test_evaluate_trace_keeps_the_best_accepted_submission_over_a_later_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "eval.sqlite"
            trace = tmp_path / "trace.jsonl"
            prepare_minimal_model_db(db_path)
            exact = {"steps": [], "building_block": "BB:1", "pathway_text": "Buy methanol."}
            analogue = {
                "steps": [{"reaction_id": "R001", "reactants": ["BB:1"], "product_smiles": "C=O"}],
                "pathway_text": "Oxidize methanol.",
            }
            events = [
                {
                    "event_type": "message",
                    "role": "user",
                    "target_smiles": "CO",
                    "payload": {"text": "Plan CO"},
                },
                *(_call_event(arguments, "CO") for arguments in (exact, analogue)),
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
            )

            summary = evaluate_trace_file(trace, db_path, model="mock-model")

            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["final_smiles"], "CO")
            self.assertEqual(summary["similarity_to_target"], 1.0)

    def test_evaluate_trace_grounds_a_reaction_smarts_from_recorded_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "eval.sqlite"
            trace = tmp_path / "trace.jsonl"
            prepare_minimal_model_db(db_path)
            analogue = {"steps": [], "building_block": "BB:1", "pathway_text": "Buy methanol."}
            exact = {
                "steps": [
                    {
                        "reaction_id": "[CH3:1][OH:2]>>[CH2:1]=[O:2]",
                        "reactants": ["BB:1"],
                        "product_smiles": "C=O",
                    }
                ],
                "pathway_text": "Oxidize methanol.",
            }
            events = [
                {
                    "event_type": "message",
                    "role": "user",
                    "target_smiles": "C=O",
                    "payload": {"text": "Plan C=O"},
                },
                _call_event(analogue, "C=O"),
                _result_event({}, "C=O"),
                _call_event(exact, "C=O"),
                # The run grounded the reaction SMARTS to this library template.
                _result_event(
                    {
                        "steps": [
                            {"step_index": 1, "template": {"smarts": "[C:1][O:2]>>[C:1]=[O:2]"}}
                        ]
                    },
                    "C=O",
                ),
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
            )

            summary = evaluate_trace_file(trace, db_path, model="mock-model")

            self.assertEqual(summary["final_smiles"], "C=O")
            self.assertEqual(summary["similarity_to_target"], 1.0)

    def test_evaluate_trace_uses_selected_submission_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "eval.sqlite"
            trace = tmp_path / "trace.jsonl"
            events = [
                {
                    "event_type": "message",
                    "role": "user",
                    "target_smiles": "CC=O",
                    "payload": {"text": "Plan CC=O"},
                },
                {
                    "event_type": "tool_result",
                    "tool_name": "submit_pathway",
                    "payload": {
                        "outcome": "ok",
                        "result": {
                            "accepted": True,
                            "reaction_validation": "valid",
                            "pathway_text": "Later lower-scoring candidate.",
                            "actual_final_smiles": "CCO",
                            "final_smiles": "CCO",
                            "submitted_final_smiles": "CCO",
                            "steps": [{"valid": True, "actual_product_smiles": "CCO"}],
                        },
                    },
                },
                {
                    "event_type": "selected_submission",
                    "tool_name": "submit_pathway",
                    "payload": {
                        "reason": "best_accepted_submission",
                        "result": {
                            "accepted": True,
                            "reaction_validation": "valid",
                            "pathway_text": "Best accepted candidate.",
                            "actual_final_smiles": "CC=O",
                            "final_smiles": "CC=O",
                            "submitted_final_smiles": "CC=O",
                            "steps": [{"valid": True, "actual_product_smiles": "CC=O"}],
                        },
                    },
                },
                {
                    "event_type": "run_summary",
                    "payload": {"runtime_seconds": 123.45},
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
            )

            summary = evaluate_trace_file(trace, db_path, model="mock-model")

            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["final_smiles"], "CC=O")
            self.assertEqual(summary["similarity_to_target"], 1.0)
            self.assertEqual(summary["runtime_seconds"], 123.45)

    def test_similarity_floor_does_not_define_run_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "eval.sqlite"
            trace = tmp_path / "trace.jsonl"
            events = [
                {
                    "event_type": "message",
                    "role": "user",
                    "target_smiles": "CC=O",
                    "payload": {"text": "Plan CC=O"},
                },
                {
                    "event_type": "selected_submission",
                    "tool_name": "submit_pathway",
                    "payload": {
                        "reason": "best_accepted_submission",
                        "result": {
                            "accepted": True,
                            "reaction_validation": "valid",
                            "pathway_text": "A valid analogue below the requested persistence floor.",
                            "actual_final_smiles": "CCO",
                            "final_smiles": "CCO",
                            "submitted_final_smiles": "CCO",
                            "steps": [{"valid": True, "actual_product_smiles": "CCO"}],
                        },
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
            )

            summary = evaluate_trace_file(
                trace,
                db_path,
                model="mock-model",
                tool_config={"stop_condition": 0.99},
            )

            self.assertTrue(summary["accepted"])
            self.assertLess(summary["similarity_to_target"], 0.99)
            self.assertFalse(summary["meets_similarity_floor"])
            self.assertTrue(summary["run_success"])

    def test_evaluate_trace_reports_token_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "eval.sqlite"
            trace = tmp_path / "trace.jsonl"
            events = [
                {
                    "event_type": "message",
                    "role": "user",
                    "target_smiles": "CCO",
                    "payload": {"text": "Plan CCO"},
                },
                {
                    "event_type": "message",
                    "role": "assistant",
                    "target_smiles": "CCO",
                    "payload": {
                        "content": "I will submit.",
                        "response_usage": {
                            "input_tokens": 11,
                            "output_tokens": 7,
                            "total_tokens": 18,
                            "cache_read_input_tokens": 2,
                            "output_tokens_details": {"reasoning_tokens": 3},
                            "cost": 0.002,
                        },
                    },
                },
                {
                    "event_type": "selected_submission",
                    "tool_name": "submit_pathway",
                    "target_smiles": "CCO",
                    "payload": {
                        "reason": "best_accepted_submission",
                        "result": {
                            "accepted": True,
                            "reaction_validation": "valid",
                            "pathway_text": "A valid exact route.",
                            "actual_final_smiles": "CCO",
                            "final_smiles": "CCO",
                            "submitted_final_smiles": "CCO",
                            "steps": [{"valid": True, "actual_product_smiles": "CCO"}],
                        },
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
            )

            summary = evaluate_trace_file(trace, db_path, model="mock-model")

            self.assertEqual(summary["input_tokens"], 11)
            self.assertEqual(summary["output_tokens"], 7)
            self.assertEqual(summary["total_tokens"], 18)
            self.assertEqual(summary["cache_read_input_tokens"], 2)
            self.assertEqual(summary["reasoning_tokens"], 3)
            self.assertEqual(summary["cost"], 0.002)

    def test_evaluate_trace_never_estimates_cost_when_provider_omits_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "eval.sqlite"
            trace = tmp_path / "trace.jsonl"
            events = [
                {
                    "event_type": "message",
                    "role": "user",
                    "target_smiles": "CCO",
                    "payload": {"text": "Plan CCO"},
                },
                {
                    "event_type": "message",
                    "role": "assistant",
                    "target_smiles": "CCO",
                    "payload": {
                        "content": "I will submit.",
                        "response_usage": {
                            "input_tokens": 1000,
                            "input_tokens_details": {"cached_tokens": 200},
                            "output_tokens": 100,
                            "total_tokens": 1100,
                            "cost": None,
                        },
                    },
                },
                {
                    "event_type": "selected_submission",
                    "tool_name": "submit_pathway",
                    "target_smiles": "CCO",
                    "payload": {
                        "reason": "best_accepted_submission",
                        "result": {
                            "accepted": True,
                            "reaction_validation": "valid",
                            "pathway_text": "A valid exact route.",
                            "actual_final_smiles": "CCO",
                            "final_smiles": "CCO",
                            "submitted_final_smiles": "CCO",
                            "steps": [{"valid": True, "actual_product_smiles": "CCO"}],
                        },
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
            )

            summary = evaluate_trace_file(trace, db_path, model="example-org/example-model")

            self.assertEqual(summary["input_tokens"], 1000)
            self.assertEqual(summary["output_tokens"], 100)
            self.assertEqual(summary["cache_read_input_tokens"], 200)
            self.assertIsNone(summary.get("cost"))

    def test_combine_summary_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "model_a_summary.csv"
            second = tmp_path / "model_b_summary.csv"
            output = tmp_path / "combined.csv"
            first.write_text("model,accepted\nmodel-a,True\n", encoding="utf-8")
            second.write_text("model,similarity_to_target\nmodel-b,0.9\n", encoding="utf-8")

            result = combine_summary_csvs([str(tmp_path / "*_summary.csv")], output)

            self.assertEqual(result["rows"], 2)
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "source_summary_file,model,accepted,similarity_to_target")
            self.assertIn("model-a,True,", lines[1])
            self.assertIn("model-b,,0.9", lines[2])

    def test_public_data_manifest_lists_expected_assets(self):
        files = list_public_data_files()
        names = {item["name"] for item in files}

        self.assertIn("synformer-reaction-templates", names)
        self.assertIn("synformer-chembl-1k", names)
        self.assertIn("synformer-enamine-real-1k", names)

    def test_model_runner_rejects_empty_model_run_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "empty.sqlite"
            db.init_db(db_path)

            with self.assertRaisesRegex(RuntimeError, "Model-run database is not ready"):
                db.validate_model_run_database(db_path)

    def test_model_runner_rejects_populated_unprepared_database_with_rebuild_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "unprepared.sqlite"
            templates = Path(tmp) / "templates.txt"
            blocks = Path(tmp) / "blocks.txt"
            templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
            blocks.write_text("SMILES\nCO\n", encoding="utf-8")
            reactions.ingest_reaction_templates(templates, db_path=db_path, library="demo")
            db.ingest_building_blocks(blocks, db_path)

            with self.assertRaisesRegex(
                RuntimeError,
                rf"synclaw db prepare-search --db {db_path}",
            ):
                db.validate_model_run_database(db_path)

    def test_trace_metrics_count_tool_calls_once_and_invalid_submissions(self):
        events = [
            {
                "event_type": "message",
                "role": "assistant",
                "payload": {"tool_calls": [{"function": {"name": "search_building_blocks"}}]},
            },
            {
                "event_type": "tool_call",
                "tool_name": "search_building_blocks",
                "payload": {"arguments": {"query": "CCO"}},
            },
            {
                "event_type": "message",
                "role": "assistant",
                "payload": {"content": "thinking"},
            },
            {
                "event_type": "tool_result",
                "tool_name": "search_building_blocks",
                "payload": {"result": {"count": 1}},
            },
            {
                "event_type": "tool_call",
                "tool_name": "submit_pathway",
                "payload": {"arguments": {"target_smiles": "CCO"}},
            },
            {
                "event_type": "tool_result",
                "tool_name": "submit_pathway",
                "payload": {"result": {"accepted": False}},
            },
        ]

        summary = summarize_events(events)

        self.assertEqual(summary["tool_calls"], 2)
        self.assertNotIn("tool_results", summary)
        self.assertEqual(summary["assistant_turns"], 2)
        self.assertEqual(summary["tool_call_turns"], 1)
        self.assertEqual(summary["non_tool_turns"], 1)
        self.assertEqual(summary["search_calls"], 1)
        self.assertEqual(summary["submission_calls"], 1)
        self.assertEqual(summary["invalid_submissions"], 1)

    def test_reaction_template_marks_abstract_reagent_supplied_atoms(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "templates.txt"
            source.write_text(
                "[NH2;$([N][#6]);!$([N]C=[O,S,N]):1]."
                "[NX3;!$(N[C,S]=[S,O,N]);$(N[#6]);!$(N[#7]);H1,H2:2]"
                ">>[N:1][C](=[O])[N:2]\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "test.sqlite"

            reactions.ingest_reaction_templates(source, db_path, library="synformer")
            with db.session(db_path) as conn:
                result = db.get_reaction_template(conn, "R001")

            self.assertTrue(result["introduces_unmapped_product_atoms"])
            self.assertNotIn("reagent-supplied", result["description"])
            self.assertEqual(result["execution_hints"], [])

    def test_model_runner_exposes_reverse_reaction_tool_schema(self):
        schema_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}

        self.assertIn("simulate_reverse_reaction", schema_names)
        self.assertIn("search_reaction_templates", schema_names)
        self.assertIn("mces", schema_names)
        self.assertNotIn("search_reaction_templates_by_molecule", schema_names)
        template_schema = next(
            schema
            for schema in TOOL_SCHEMAS
            if schema["function"]["name"] == "search_reaction_templates"
        )
        properties = template_schema["function"]["parameters"]["properties"]
        self.assertIn("query", properties)
        self.assertIn("reactant_smiles", properties)
        self.assertIn("product_smiles", properties)
        self.assertNotIn("library", properties)
        self.assertNotIn("rank_by_reverse", properties)
        self.assertNotIn("reverse_rank_limit", properties)
        self.assertNotIn("reverse_max_outcomes", properties)
        self.assertNotIn("near_limit", properties)

    def test_local_tool_dispatcher_executes_mces(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools = SynthesisTools(Path(tmp) / "test.sqlite")

            result = call_local_tool(
                tools,
                "mces",
                {"smiles_a": "CCCO", "smiles_b": "CCCN"},
            )

            self.assertTrue(result["valid"])
            self.assertEqual(result["mces_smiles"], "CCC")
            self.assertEqual(result["similarity"], 0.5)

    def test_model_runner_filters_excluded_tool_schemas(self):
        excluded = normalize_excluded_tools(["simulate_reverse_reaction,search_reaction_templates"])

        schema_names = {schema["function"]["name"] for schema in filter_chat_tool_schemas(excluded)}

        self.assertNotIn("simulate_reverse_reaction", schema_names)
        self.assertNotIn("search_reaction_templates", schema_names)
        self.assertIn("simulate_reaction", schema_names)
        self.assertIn("submit_pathway", schema_names)

    def test_model_runner_rejects_excluding_submit_pathway(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot exclude required tool"):
            normalize_excluded_tools(["submit_pathway"])

    def test_forced_submit_tool_choice_only_without_prior_submission(self):
        tools = filter_chat_tool_schemas(set())
        forced = {"type": "function", "function": {"name": "submit_pathway"}}
        self.assertEqual(
            submit_tool_choice(
                forced_submit_requested=True, has_accepted_submission=False, tools=tools
            ),
            forced,
        )
        self.assertEqual(
            submit_tool_choice(
                forced_submit_requested=True, has_accepted_submission=True, tools=tools
            ),
            "auto",
        )
        self.assertEqual(
            submit_tool_choice(
                forced_submit_requested=False, has_accepted_submission=False, tools=tools
            ),
            "auto",
        )
        self.assertEqual(
            submit_tool_choice(
                forced_submit_requested=True, has_accepted_submission=False, tools=[]
            ),
            "auto",
        )

    def test_submit_pathway_schema_allows_zero_step_direct_purchase(self):
        schema = next(
            schema for schema in TOOL_SCHEMAS if schema["function"]["name"] == "submit_pathway"
        )
        parameters = schema["function"]["parameters"]
        properties = parameters["properties"]

        self.assertIn("building_block", properties)
        self.assertNotIn("minItems", properties["steps"])
        self.assertEqual(parameters["required"], ["pathway_text"])

    def test_initial_prompt_allows_direct_purchase_submission(self):
        task = build_llm_task(
            "CO",
            task_id="toy-0001",
            dataset="toy",
            source_file="toy.smi",
        )
        messages = build_initial_messages(
            task=task,
            config=policy.RolloutPolicyConfig(max_model_len=262144),
        )

        self.assertIn("You are performing synthesis planning", messages[0]["content"])
        self.assertIn("single valid building_block for direct purchase", messages[0]["content"])
        self.assertIn(
            "Submit pathways with submit_pathway, which you may call more than once",
            messages[0]["content"],
        )
        # The fixed limits and the run-start context limits share the final paragraph.
        self.assertTrue(
            messages[0]["content"]
            .split("\n\n")[-1]
            .startswith(
                "Use no more than 150 total tool calls or 80 assistant turns. Your context-window budget"
            )
        )
        self.assertEqual(messages[1]["content"], "Target molecule: CO")

    def test_local_tool_dispatcher_allows_building_block_only_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite"
            prepare_minimal_model_db(db_path)

            result = call_local_tool(
                SynthesisTools(db_path),
                "submit_pathway",
                {
                    "building_block": "BB:1",
                    "pathway_text": "The molecule is directly purchasable.",
                },
                target_smiles="CO",
            )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["pathway_type"], "direct_purchase")
        self.assertEqual(result["final_smiles"], "CO")

    def test_harness_prompt_names_no_optional_tool(self):
        optional = [name for name in TOOL_NAMES if name not in {"submit_pathway", "similarity"}]
        task = build_llm_task(
            "CCO", task_id="fixture-0001", dataset="fixture", source_file="fixture"
        )
        prompts = [
            message["content"]
            for message in build_initial_messages(
                task=task, config=policy.RolloutPolicyConfig(max_tool_calls=10)
            )
        ]

        for prompt, name in itertools.product(prompts, optional):
            self.assertNotIn(name, prompt)
        self.assertNotIn("with similarity", prompts[0])
        # Only the validator is always present to verify a route.
        for prompt in prompts:
            self.assertNotIn("verified", prompt)
            self.assertNotIn("reaction tools", prompt)

    def test_single_tool_call_budget_reads_in_the_singular(self):
        messages = build_initial_messages(
            task=build_llm_task(
                "CCO", task_id="fixture-0001", dataset="fixture", source_file="fixture"
            ),
            config=policy.RolloutPolicyConfig(max_tool_calls=1, max_turns=8),
        )

        self.assertIn(
            "Use no more than 1 total tool call or 8 assistant turns.", messages[0]["content"]
        )

    def test_messages_tool_schema_uses_anthropic_input_schema(self):
        schemas = filter_messages_tool_schemas({"simulate_reverse_reaction"})
        schema_by_name = {schema["name"]: schema for schema in schemas}

        self.assertIn("submit_pathway", schema_by_name)
        self.assertNotIn("simulate_reverse_reaction", schema_by_name)
        self.assertIn("input_schema", schema_by_name["submit_pathway"])

    def test_model_runner_rejects_non_object_tool_arguments(self):
        with self.assertRaises(ValueError):
            parse_arguments("[]")

    def test_model_runner_coerces_json_string_reactants(self):
        arguments, coercions = coerce_tool_arguments(
            "simulate_reaction",
            {
                "reaction_id": "R036",
                "reactants": '["Nc1nccc2occc12", "N#CC1CC12CNC2"]',
            },
        )

        self.assertEqual(arguments["reactants"], ["Nc1nccc2occc12", "N#CC1CC12CNC2"])
        self.assertEqual(coercions[0]["field"], "reactants")
        self.assertEqual(coercions[0]["coercion"], "json_string_to_array")

    def test_model_runner_coerces_structure_search_smiles_arrays(self):
        arguments, coercions = coerce_tool_arguments(
            "search_reaction_templates",
            {
                "reactant_smiles": '["N", "O=CO"]',
                "product_smiles": '["NC=O"]',
            },
        )

        self.assertEqual(arguments["reactant_smiles"], ["N", "O=CO"])
        self.assertEqual(arguments["product_smiles"], ["NC=O"])
        self.assertEqual(
            [item["field"] for item in coercions], ["reactant_smiles", "product_smiles"]
        )

    def test_model_runner_does_not_coerce_non_json_reactant_text(self):
        arguments, coercions = coerce_tool_arguments(
            "simulate_reaction",
            {"reaction_id": "R036", "reactants": "Nc1nccc2occc12, N#CC1CC12CNC2"},
        )

        self.assertEqual(arguments["reactants"], "Nc1nccc2occc12, N#CC1CC12CNC2")
        self.assertEqual(coercions, [])

    def test_model_runner_coerces_submit_step_reactants(self):
        arguments, coercions = coerce_tool_arguments(
            "submit_pathway",
            {
                "steps": [
                    {
                        "reaction_id": "R036",
                        "reactants": '["BB:82779", "BB:203506"]',
                        "product_smiles": "product",
                    }
                ],
                "pathway_text": "route",
            },
        )

        self.assertEqual(arguments["steps"][0]["reactants"], ["BB:82779", "BB:203506"])
        self.assertEqual(coercions[0]["field"], "steps[0].reactants")

    def test_model_runner_coerces_json_string_submit_steps(self):
        arguments, coercions = coerce_tool_arguments(
            "submit_pathway",
            {
                "steps": json.dumps(
                    [
                        {
                            "reaction_id": "R036",
                            "reactants": '["BB:82779", "BB:203506"]',
                            "product_smiles": "product",
                        }
                    ]
                ),
                "pathway_text": "route",
            },
        )

        self.assertIsInstance(arguments["steps"], list)
        self.assertEqual(arguments["steps"][0]["reactants"], ["BB:82779", "BB:203506"])
        self.assertEqual([item["field"] for item in coercions], ["steps", "steps[0].reactants"])

    def test_query_less_template_search_returns_the_family_overview(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "templates.txt"
            annotations = tmp_path / "annotations.jsonl"
            db_path = tmp_path / "test.sqlite"
            source.write_text(
                "[NX3;!$(N[C]=[S,O,N]);$(N[#6,S]);!$(N[#7]);H1,H2:1]."
                "[#6:4][C:5](=[O:6])[OH,O-]>>[N:1][C:5](=[O:6])[#6:4]\n",
                encoding="utf-8",
            )
            annotations.write_text(
                json.dumps(
                    {
                        "library": "demo",
                        "reaction_id": "demo:R000",
                        "name": "Curated coupling template",
                        "family": "amide formation",
                        "family_metadata": {
                            "description": "Forms amides from amine and acyl-like partners.",
                            "search_terms": ["amide formation", "carboxylic acid plus amine"],
                        },
                        "description": "Curated template description.",
                        "search_terms": ["amide coupling", "carboxylic acid plus amine"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            reactions.ingest_reaction_templates(
                source,
                db_path,
                library="demo",
                annotations_path=annotations,
            )

            overview = call_local_tool(SynthesisTools(db_path), "search_reaction_templates", {})

            self.assertEqual(overview["families"][0]["family"], "amide formation")
            self.assertIn("carboxylic acid plus amine", overview["families"][0]["search_terms"])


if __name__ == "__main__":
    unittest.main()
