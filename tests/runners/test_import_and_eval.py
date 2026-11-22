# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from synclaw.core import db, reactions
from synclaw.evaluation.evaluate import evaluate_trace_file, summarize_events
from synclaw.evaluation.results import combine_summary_csvs
from synclaw.paths import REPO_ROOT
from synclaw.rollout import policy
from synclaw.rollout.loop import call_local_tool
from synclaw.rollout.policy import (
    coerce_tool_arguments,
    parse_arguments,
    response_or_chat_tool_choice,
)
from synclaw.rollout.prompts import build_initial_messages, validate_model_run_database
from synclaw.runners.model_runner import (
    _HostedTransport,
    add_arguments,
    anthropic_completion,
    anthropic_response_to_chat_message,
    apply_provider_defaults,
    build_anthropic_initial_state,
    checkpoint_matches_run,
    resolve_task_db_path,
    responses_completion,
    responses_to_chat_message,
    run_model_from_args,
)
from synclaw.tasks.datasets import (
    PROMPT_FINGERPRINT,
    TOOL_NAMES,
    build_llm_task,
    generate_llm_taskset,
)
from synclaw.tasks.public_data import list_public_data_files
from synclaw.tools.contracts import (
    TOOL_SCHEMAS,
    filter_chat_tool_schemas,
    filter_messages_tool_schemas,
    filter_responses_tool_schemas,
    normalize_excluded_tools,
)
from synclaw.tools.synthesis import SynthesisTools


def _hosted_transport(api_mode: str, **state):
    args = SimpleNamespace(
        api_mode=api_mode,
        model="mock-model",
        responses_state_mode="full",
        responses_url="https://example.test/v1/responses",
        max_tokens=16,
        temperature=None,
        top_p=None,
        sampling_seed=None,
        reasoning_effort=None,
        retries=0,
        request_timeout=1.0,
    )
    fields = {
        "responses_previous_id": None,
        "responses_pending_input": [],
        "responses_input_history": [],
        "anthropic_system": [],
        "anthropic_messages": [],
        "anthropic_tool_results": [],
        "chat_tool_schemas": [],
        "responses_tool_schemas": [],
        "messages_tool_schemas": [],
        **state,
    }
    config = policy.RolloutPolicyConfig(max_tokens=args.max_tokens)
    return _HostedTransport(args=args, config=config, api_key="test-key", **fields)


def _provider_args(**overrides):
    import argparse

    fields = {
        "provider": "nvbuild",
        "model": "example-model",
        "api_mode": None,
        "base_url": None,
        "responses_url": None,
        "messages_url": None,
        "responses_state_mode": None,
        "reasoning_effort": None,
        **overrides,
    }
    return argparse.Namespace(**fields)


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
    def _prepare_minimal_model_db(self, db_path: Path) -> None:
        templates = db_path.with_name("templates.jsonl")
        bbs = db_path.with_name("bbs.txt")
        templates.write_text(
            json.dumps(
                {
                    "reaction_id": "R000",
                    "smarts": "[C:1][O:2]>>[C:1]=[O:2]",
                    "family": "oxidation",
                    "family_metadata": {
                        "description": "Fixture alcohol-to-carbonyl transformation.",
                        "search_terms": ["alcohol oxidation"],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        bbs.write_text("SMILES\nCO\n", encoding="utf-8")
        reactions.ingest_reaction_templates(templates, db_path=db_path, library="demo")
        db.ingest_building_blocks(bbs, db_path)
        db.prepare_building_block_search(db_path)

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
            self._prepare_minimal_model_db(db_path)
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
            self._prepare_minimal_model_db(db_path)
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
            self._prepare_minimal_model_db(db_path)
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
                        "token_usage": {
                            "input_tokens": 11,
                            "output_tokens": 7,
                            "total_tokens": 18,
                            "cached_input_tokens": 2,
                            "reasoning_tokens": 3,
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
            self.assertEqual(summary["cached_input_tokens"], 2)
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
            self.assertEqual(summary["cached_input_tokens"], 200)
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

    def test_model_runner_provider_defaults(self):
        import argparse

        args = argparse.Namespace(
            provider="nvbuild",
            api_mode=None,
            base_url=None,
            responses_url=None,
            responses_state_mode=None,
        )

        apply_provider_defaults(args)

        self.assertEqual(args.api_mode, "chat")
        self.assertEqual(args.base_url, "https://integrate.api.nvidia.com/v1")
        self.assertEqual(args.responses_state_mode, "previous_id")
        self.assertEqual(args.responses_url, "")
        self.assertEqual(args.messages_url, "")

    def test_model_runner_api_mode_comes_from_the_preset_not_the_model_name(self):
        for provider, api_mode in (
            ("openrouter", "chat"),
            ("openai", "responses"),
            ("anthropic", "messages"),
        ):
            args = _provider_args(provider=provider, model="anthropic/example-model")

            apply_provider_defaults(args)

            self.assertEqual(args.api_mode, api_mode, provider)
            self.assertEqual(args.responses_state_mode, "previous_id", provider)
            self.assertIsNone(args.reasoning_effort, provider)

    def test_model_runner_flags_override_the_preset(self):
        args = _provider_args(
            provider="custom",
            api_mode="responses",
            responses_url="https://example.test/v1/responses",
            responses_state_mode="full",
        )

        apply_provider_defaults(args)

        self.assertEqual(args.api_mode, "responses")
        self.assertEqual(args.responses_url, "https://example.test/v1/responses")
        self.assertEqual(args.responses_state_mode, "full")

    def test_model_runner_anthropic_provider_uses_the_messages_endpoint(self):
        args = _provider_args(provider="anthropic")

        apply_provider_defaults(args)

        self.assertEqual(args.api_mode, "messages")
        self.assertEqual(args.messages_url, "https://api.anthropic.com/v1/messages")

    def test_checkpoint_must_record_the_same_responses_state_mode(self):
        import argparse

        args = argparse.Namespace(
            model="example-model",
            provider="openai",
            api_mode="responses",
            responses_state_mode="previous_id",
        )
        task = {"task_id": "fixture-0001", "target_smiles": "CCO"}
        messages = [{"role": "user", "content": "Plan CCO."}]
        checkpoint = {
            "model": args.model,
            "provider": args.provider,
            "api_mode": args.api_mode,
            "task_id": task["task_id"],
            "target_smiles": task["target_smiles"],
            "next_turn_index": 0,
            "responses_pending_input": messages,
            "responses_input_history": messages,
        }
        config = policy.RolloutPolicyConfig()

        self.assertFalse(checkpoint_matches_run(checkpoint, args, task, config))
        self.assertTrue(
            checkpoint_matches_run(
                {**checkpoint, "responses_state_mode": "previous_id"}, args, task, config
            )
        )

    def test_model_runner_keeps_max_effort_for_messages_api(self):
        args = _provider_args(provider="anthropic", reasoning_effort="max")

        apply_provider_defaults(args)

        self.assertEqual(args.api_mode, "messages")
        self.assertEqual(args.reasoning_effort, "max")

    def test_model_runner_maps_max_effort_to_xhigh_for_other_apis(self):
        for provider in ("openai", "nvbuild"):
            args = _provider_args(provider=provider, reasoning_effort="max")

            apply_provider_defaults(args)

            self.assertEqual(args.reasoning_effort, "xhigh", provider)

    def test_responses_completion_sets_store_for_previous_response_id_mode(self):
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self):
                return b'{"id":"resp_123","output":[],"usage":{"input_tokens":1,"output_tokens":1}}'

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            response = responses_completion(
                responses_url="https://example.test/v1/responses",
                api_key="test-key",
                model="example-model",
                input_items=[{"role": "user", "content": "Question"}],
                previous_response_id=None,
                store=True,
                max_output_tokens=128,
                temperature=None,
                top_p=1.0,
                sampling_seed=7,
                reasoning_effort="xhigh",
                retries=0,
                request_timeout=12.0,
                tool_choice="auto",
                tools=[],
            )

        self.assertEqual(response["id"], "resp_123")
        payload = captured["payload"]
        self.assertIs(payload["store"], True)
        self.assertEqual(payload["reasoning"], {"effort": "xhigh"})
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["seed"], 7)

    def test_model_runner_low_similarity_cap_defaults_to_none(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(
            [
                "--provider",
                "openai",
                "--api-key-env",
                "TEST_AGENTIC_SYNTHESIS_API_KEY",
                "--taskset",
                "tasks.jsonl",
                "--task-index",
                "0",
                "--db",
                "test.sqlite",
                "--model",
                "mock-model",
            ]
        )

        self.assertIsNone(args.max_low_similarity_submissions)
        self.assertEqual(args.max_turns, 80)
        self.assertEqual(args.max_tool_calls, 150)
        self.assertEqual(args.turn_warning_remaining, 3)
        self.assertEqual(args.force_submit_after_non_tool_turns, 2)
        self.assertEqual(args.same_pathway_finish_after, 5)

    def test_model_runner_openai_provider_defaults(self):
        args = _provider_args(provider="openai")

        apply_provider_defaults(args)

        self.assertEqual(args.api_mode, "responses")
        self.assertEqual(args.base_url, "https://api.openai.com/v1")
        self.assertEqual(args.responses_url, "https://api.openai.com/v1/responses")

    def test_public_data_manifest_lists_expected_assets(self):
        files = list_public_data_files()
        names = {item["name"] for item in files}

        self.assertIn("synformer-reaction-templates", names)
        self.assertIn("synformer-chembl-1k", names)
        self.assertIn("synformer-enamine-real-1k", names)

    def test_model_runner_defaults_to_the_synclaw_db(self):
        self.assertEqual(resolve_task_db_path(None), Path("var/synclaw.sqlite"))

    def test_model_runner_respects_explicit_db_override(self):
        self.assertEqual(resolve_task_db_path("var/custom.sqlite"), Path("var/custom.sqlite"))

    def test_model_runner_rejects_empty_model_run_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "empty.sqlite"
            db.init_db(db_path)

            with self.assertRaisesRegex(RuntimeError, "Model-run database is not ready"):
                validate_model_run_database(db_path)

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
                validate_model_run_database(db_path)

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

    def test_forced_submit_tool_choice_only_for_responses_and_chat_without_prior_submission(self):
        self.assertEqual(
            response_or_chat_tool_choice(
                api_mode="responses",
                forced_submit_requested=True,
                has_submission=False,
                tools=filter_responses_tool_schemas(set()),
            ),
            {"type": "function", "name": "submit_pathway"},
        )
        self.assertEqual(
            response_or_chat_tool_choice(
                api_mode="chat",
                forced_submit_requested=True,
                has_submission=False,
                tools=filter_chat_tool_schemas(set()),
            ),
            {"type": "function", "function": {"name": "submit_pathway"}},
        )
        self.assertEqual(
            response_or_chat_tool_choice(
                api_mode="messages",
                forced_submit_requested=True,
                has_submission=False,
                tools=filter_messages_tool_schemas(set()),
            ),
            "auto",
        )
        self.assertEqual(
            response_or_chat_tool_choice(
                api_mode="responses",
                forced_submit_requested=True,
                has_submission=True,
                tools=filter_responses_tool_schemas(set()),
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
            self._prepare_minimal_model_db(db_path)

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

    def test_model_runner_omits_run_and_task_excluded_tools_from_schema(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taskset = tmp_path / "tasks.jsonl"
            trace = tmp_path / "trace.jsonl"
            db_path = tmp_path / "test.sqlite"
            self._prepare_minimal_model_db(db_path)
            task = build_llm_task(
                "CCO",
                task_id="fixture-0001",
                dataset="fixture",
                source_file="fixture.smi",
                excluded_tools=["search_reaction_templates", "mces"],
            )
            taskset.write_text(json.dumps(task) + "\n", encoding="utf-8")
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(
                [
                    "--provider",
                    "openai",
                    "--api-mode",
                    "chat",
                    "--api-key-env",
                    "TEST_AGENTIC_SYNTHESIS_API_KEY",
                    "--taskset",
                    str(taskset),
                    "--task-index",
                    "0",
                    "--db",
                    str(db_path),
                    "--trace",
                    str(trace),
                    "--model",
                    "mock-model",
                    "--max-turns",
                    "1",
                    "--max-tokens",
                    "16",
                    "--exclude-tool",
                    "simulate_reverse_reaction",
                ]
            )
            seen_kwargs: list[dict] = []

            def fake_chat_completion(**kwargs):
                seen_kwargs.append(kwargs)
                return {"choices": [{"message": {"role": "assistant", "content": "No tool call."}}]}

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.chat_completion",
                    side_effect=fake_chat_completion,
                ),
            ):
                summary = run_model_from_args(args)

            self.assertEqual(
                summary["excluded_tools"],
                ["mces", "search_reaction_templates", "simulate_reverse_reaction"],
            )
            tool_names = {schema["function"]["name"] for schema in seen_kwargs[0]["tools"]}
            prompt = (
                seen_kwargs[0]["messages"][0]["content"]
                + "\n\n"
                + seen_kwargs[0]["messages"][1]["content"]
            )
            self.assertNotIn("simulate_reverse_reaction", tool_names)
            self.assertNotIn("search_reaction_templates", tool_names)
            self.assertNotIn("mces", tool_names)
            for name in ("simulate_reverse_reaction", "search_reaction_templates", "mces"):
                self.assertNotIn(name, prompt)

    def test_anthropic_initial_state_separates_system_and_messages(self):
        system, messages = build_anthropic_initial_state(
            [
                {"role": "system", "content": "Stable harness instructions."},
                {"role": "user", "content": "Target-specific question."},
            ]
        )

        self.assertEqual(
            system,
            [
                {
                    "type": "text",
                    "text": "Stable harness instructions.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Target-specific question."}],
                }
            ],
        )

    def test_messages_tool_schema_uses_anthropic_input_schema(self):
        schemas = filter_messages_tool_schemas({"simulate_reverse_reaction"})
        schema_by_name = {schema["name"]: schema for schema in schemas}

        self.assertIn("submit_pathway", schema_by_name)
        self.assertNotIn("simulate_reverse_reaction", schema_by_name)
        self.assertIn("input_schema", schema_by_name["submit_pathway"])

    def test_anthropic_response_to_chat_message_converts_tool_use(self):
        message = anthropic_response_to_chat_message(
            {
                "id": "msg_123",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "I will search."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "search_building_blocks",
                        "input": {"query": "CCO"},
                    },
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "output_tokens_details": {"thinking_tokens": 2},
                },
            },
        )

        self.assertEqual(message["content"], "I will search.")
        self.assertEqual(message["anthropic_stop_reason"], "tool_use")
        self.assertEqual(message["token_usage"]["reasoning_tokens"], 2)
        self.assertEqual(message["tool_calls"][0]["id"], "toolu_123")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "search_building_blocks")
        self.assertEqual(
            json.loads(message["tool_calls"][0]["function"]["arguments"]),
            {"query": "CCO"},
        )

    def test_anthropic_completion_uses_api_key_header_cache_control_and_adaptive_thinking(self):
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self):
                return b'{"id":"msg_123","content":[],"usage":{"input_tokens":1,"output_tokens":1}}'

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            response = anthropic_completion(
                messages_url="https://api.anthropic.com/v1/messages",
                api_key="test-key",
                model="example-model",
                system="Stable harness instructions.",
                messages=[{"role": "user", "content": [{"type": "text", "text": "Question"}]}],
                max_tokens=1024,
                temperature=None,
                top_p=None,
                reasoning_effort="high",
                retries=0,
                request_timeout=12.0,
                tool_choice={"type": "tool", "name": "submit_pathway"},
                tools=[
                    {
                        "name": "similarity",
                        "description": "Compute",
                        "input_schema": {"type": "object"},
                    }
                ],
            )

        self.assertEqual(response["id"], "msg_123")
        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        headers = {str(key).lower(): value for key, value in captured["headers"].items()}
        self.assertEqual(headers["x-api-key"], "test-key")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertNotIn("authorization", headers)
        payload = captured["payload"]
        self.assertEqual(
            payload["system"],
            [
                {
                    "type": "text",
                    "text": "Stable harness instructions.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
        self.assertEqual(payload["cache_control"], {"type": "ephemeral"})
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertEqual(payload["output_config"], {"effort": "high"})
        self.assertEqual(payload["tool_choice"], {"type": "tool", "name": "submit_pathway"})
        self.assertEqual(payload["tools"][0]["input_schema"], {"type": "object"})

    def test_model_runner_anthropic_messages_mode_uses_tool_results(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taskset = tmp_path / "tasks.jsonl"
            trace = tmp_path / "trace.jsonl"
            db_path = tmp_path / "test.sqlite"
            self._prepare_minimal_model_db(db_path)
            taskset.write_text(
                json.dumps(
                    {
                        "task_id": "fixture-0001",
                        "target_smiles": "C=O",
                        "system_prompt": "You plan syntheses.",
                        "prompt_fingerprint": PROMPT_FINGERPRINT,
                        "question": "Plan C=O.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(
                [
                    "--provider",
                    "anthropic",
                    "--api-key-env",
                    "TEST_AGENTIC_SYNTHESIS_API_KEY",
                    "--taskset",
                    str(taskset),
                    "--task-index",
                    "0",
                    "--db",
                    str(db_path),
                    "--trace",
                    str(trace),
                    "--model",
                    "example-model",
                    "--reasoning-effort",
                    "high",
                    "--max-turns",
                    "4",
                    "--max-tokens",
                    "1024",
                ]
            )
            seen_kwargs: list[dict] = []

            def fake_anthropic_completion(**kwargs):
                seen_kwargs.append(
                    {
                        **kwargs,
                        "messages": json.loads(json.dumps(kwargs["messages"])),
                        "tools": json.loads(json.dumps(kwargs["tools"])),
                    }
                )
                if len(seen_kwargs) == 1:
                    return {
                        "id": "msg-search",
                        "stop_reason": "tool_use",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu-search",
                                "name": "search_building_blocks",
                                "input": {"query": "CO", "limit": 1},
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 3},
                    }
                return {
                    "id": "msg-submit",
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu-submit",
                            "name": "submit_pathway",
                            "input": {
                                "steps": [
                                    {
                                        "reaction_id": "R001",
                                        "reactants": ["BB:1"],
                                        "product_smiles": "C=O",
                                    }
                                ],
                                "pathway_text": "Oxidize methanol to formaldehyde.",
                            },
                        }
                    ],
                    "usage": {"input_tokens": 20, "output_tokens": 4},
                }

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.anthropic_completion",
                    side_effect=fake_anthropic_completion,
                ),
            ):
                summary = run_model_from_args(args)

            self.assertEqual(args.api_mode, "messages")
            self.assertEqual(args.reasoning_effort, "high")
            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["similarity_to_target"], 1.0)
            self.assertEqual(seen_kwargs[0]["reasoning_effort"], "high")
            tool_result = next(
                block
                for message in seen_kwargs[1]["messages"]
                for block in message.get("content", [])
                if block.get("type") == "tool_result"
            )
            self.assertEqual(tool_result["type"], "tool_result")
            self.assertEqual(tool_result["tool_use_id"], "toolu-search")
            self.assertIn("results", json.loads(tool_result["content"]))

    def test_anthropic_transport_checkpoints_and_flushes_tool_results_before_repairs(self):
        assistant_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu-invalid",
                    "name": "search_building_blocks",
                    "input": {"query": ""},
                }
            ],
        }
        transport = _hosted_transport(
            "messages",
            anthropic_messages=[assistant_message],
        )
        transport.append_tool(
            SimpleNamespace(call_id="toolu-invalid"),
            '{"error":"invalid_tool_call","retryable":true}',
        )

        checkpoint = transport.checkpoint_fields()

        self.assertEqual(
            checkpoint["anthropic_tool_results"][0]["tool_use_id"],
            "toolu-invalid",
        )
        resumed_messages = [assistant_message]
        resumed = _hosted_transport(
            "messages",
            anthropic_messages=resumed_messages,
            anthropic_tool_results=checkpoint["anthropic_tool_results"],
        )
        resumed.append_user({"role": "user", "content": "Repair the invalid call."})

        self.assertEqual(resumed_messages[1]["role"], "user")
        self.assertEqual(resumed_messages[1]["content"][0]["type"], "tool_result")
        self.assertEqual(resumed_messages[2]["role"], "user")
        self.assertEqual(resumed_messages[2]["content"][0]["type"], "text")
        self.assertEqual(resumed.checkpoint_fields()["anthropic_tool_results"], [])

    def test_model_runner_anthropic_messages_mode_requests_final_submit_without_forced_tool_choice(
        self,
    ):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taskset = tmp_path / "tasks.jsonl"
            trace = tmp_path / "trace.jsonl"
            db_path = tmp_path / "test.sqlite"
            self._prepare_minimal_model_db(db_path)
            taskset.write_text(
                json.dumps(
                    {
                        "task_id": "fixture-0001",
                        "target_smiles": "C=O",
                        "system_prompt": "You plan syntheses.",
                        "prompt_fingerprint": PROMPT_FINGERPRINT,
                        "question": "Plan C=O.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(
                [
                    "--provider",
                    "anthropic",
                    "--api-key-env",
                    "TEST_AGENTIC_SYNTHESIS_API_KEY",
                    "--taskset",
                    str(taskset),
                    "--task-index",
                    "0",
                    "--db",
                    str(db_path),
                    "--trace",
                    str(trace),
                    "--model",
                    "example-model",
                    "--max-turns",
                    "1",
                    "--max-tokens",
                    "1024",
                ]
            )
            seen_kwargs: list[dict] = []

            def fake_anthropic_completion(**kwargs):
                seen_kwargs.append(
                    {
                        **kwargs,
                        "messages": json.loads(json.dumps(kwargs["messages"])),
                        "tools": json.loads(json.dumps(kwargs["tools"])),
                    }
                )
                return {
                    "id": "msg-submit",
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu-submit",
                            "name": "submit_pathway",
                            "input": {
                                "steps": [
                                    {
                                        "reaction_id": "R001",
                                        "reactants": ["BB:1"],
                                        "product_smiles": "C=O",
                                    }
                                ],
                                "pathway_text": "Oxidize methanol to formaldehyde.",
                            },
                        }
                    ],
                    "usage": {"input_tokens": 20, "output_tokens": 4},
                }

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.anthropic_completion",
                    side_effect=fake_anthropic_completion,
                ),
            ):
                summary = run_model_from_args(args)

            self.assertIsNone(seen_kwargs[0]["tool_choice"])
            final_turn_reminders = [
                block["text"]
                for message in seen_kwargs[0]["messages"]
                for block in message.get("content", [])
                if block.get("type") == "text" and "final allowed turn" in block.get("text", "")
            ]
            self.assertEqual(len(final_turn_reminders), 1)
            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["similarity_to_target"], 1.0)

    def test_model_runner_checkpoint_survives_failure_and_can_resume(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taskset = tmp_path / "tasks.jsonl"
            trace = tmp_path / "trace.jsonl"
            checkpoint = tmp_path / "checkpoints" / "run.json"
            db_path = tmp_path / "test.sqlite"
            self._prepare_minimal_model_db(db_path)
            taskset.write_text(
                json.dumps(
                    {
                        "task_id": "fixture-0001",
                        "target_smiles": "C=O",
                        "system_prompt": "You plan syntheses.",
                        "prompt_fingerprint": PROMPT_FINGERPRINT,
                        "question": "Plan C=O.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(
                [
                    "--provider",
                    "openai",
                    "--api-mode",
                    "chat",
                    "--api-key-env",
                    "TEST_AGENTIC_SYNTHESIS_API_KEY",
                    "--taskset",
                    str(taskset),
                    "--task-index",
                    "0",
                    "--db",
                    str(db_path),
                    "--trace",
                    str(trace),
                    "--checkpoint",
                    str(checkpoint),
                    "--model",
                    "mock-model",
                    "--max-turns",
                    "3",
                    "--max-tokens",
                    "16",
                ]
            )
            calls = 0

            def failing_chat_completion(**_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call-search",
                                            "type": "function",
                                            "function": {
                                                "name": "search_building_blocks",
                                                "arguments": json.dumps(
                                                    {"query": "CO", "limit": 1}
                                                ),
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                raise RuntimeError("rate limit after first turn")

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.chat_completion",
                    side_effect=failing_chat_completion,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    run_model_from_args(args)

            self.assertTrue(checkpoint.exists())
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertIn("tool_result", [event["event_type"] for event in events])
            self.assertEqual(events[-1]["event_type"], "run_error")
            search_result_event = next(
                event
                for event in events
                if event["event_type"] == "tool_result"
                and event.get("tool_name") == "search_building_blocks"
            )
            self.assertIn("submission_guidance", search_result_event["payload"]["result"])
            self.assertIn("model_result", search_result_event["payload"])
            self.assertIn("model_result_compaction", search_result_event["payload"])
            self.assertNotIn(
                "submission_note", search_result_event["payload"]["model_result"]["results"][0]
            )
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            tool_messages = [
                message
                for message in checkpoint_payload["messages"]
                if message.get("role") == "tool"
            ]
            self.assertEqual(len(tool_messages), 1)
            model_visible_result = json.loads(tool_messages[0]["content"])
            self.assertIn("results", model_visible_result)
            self.assertNotIn("submission_note", model_visible_result["results"][0])
            self.assertNotIn("trace_events", checkpoint_payload)
            self.assertLess(
                search_result_event["payload"]["model_result_compaction"]["model_json_bytes"],
                search_result_event["payload"]["model_result_compaction"]["full_json_bytes"],
            )

            def successful_chat_completion(**_kwargs):
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call-submit",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_pathway",
                                            "arguments": json.dumps(
                                                {
                                                    "steps": [
                                                        {
                                                            "reaction_id": "R001",
                                                            "reactants": ["BB:1"],
                                                            "product_smiles": "C=O",
                                                        }
                                                    ],
                                                    "pathway_text": "Oxidize methanol to formaldehyde.",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }

            moved_trace = trace.with_name("moved-trace.jsonl")
            trace.rename(moved_trace)
            with patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}):
                with self.assertRaisesRegex(RuntimeError, "which does not exist"):
                    run_model_from_args(args)
            moved_trace.rename(trace)

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.chat_completion",
                    side_effect=successful_chat_completion,
                ),
            ):
                summary = run_model_from_args(args)

            self.assertFalse(checkpoint.exists())
            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["similarity_to_target"], 1.0)

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

    def test_responses_parser_keeps_text_and_response_metadata(self):
        response = {
            "id": "resp-test",
            "model": "example-org/example-model-revision-1",
            "system_fingerprint": "deployment-abc",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": 'No results. Search building blocks next. {"limit": 10, "query": "C1CC2CN"}',
                        }
                    ],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Searching next."}],
                },
            ],
        }

        message = responses_to_chat_message(response)

        self.assertEqual(message["content"], "Searching next.")
        self.assertNotIn("tool_calls", message)
        self.assertEqual(message["response_model"], "example-org/example-model-revision-1")
        self.assertEqual(message["system_fingerprint"], "deployment-abc")

    def test_responses_parser_preserves_token_usage(self):
        response = {
            "id": "resp-test",
            "status": "completed",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cache_creation_input_tokens": 7,
                "cache_read_input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 2},
                "cost": 0.001,
            },
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done."}],
                }
            ],
        }

        message = responses_to_chat_message(response)

        self.assertEqual(message["response_usage"]["total_tokens"], 15)
        self.assertEqual(
            message["token_usage"],
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cached_input_tokens": 3,
                "cache_creation_input_tokens": 7,
                "cache_read_input_tokens": 3,
                "reasoning_tokens": 2,
                "cost": 0.001,
            },
        )

    def test_responses_parser_marks_max_output_token_hit(self):
        response = {
            "id": "resp-test",
            "status": "completed",
            "max_output_tokens": 16,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 16,
                "total_tokens": 26,
            },
            "output": [{"type": "reasoning", "summary": []}],
        }

        message = responses_to_chat_message(response)

        self.assertEqual(message["max_output_tokens"], 16)
        self.assertTrue(message["hit_max_output_tokens"])

    def test_responses_transport_propagates_incomplete_output_limit(self):
        transport = _hosted_transport("responses")
        response = {
            "id": "resp-incomplete",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "summary": []}],
        }

        with patch("synclaw.runners.model_runner.responses_completion", return_value=response):
            turn = asyncio.run(
                transport.complete([], tools=[], tool_choice="auto", max_output_tokens=16)
            )

        self.assertEqual(turn.finish_reason, "max_output_tokens")

    def test_responses_parser_never_estimates_cost_when_provider_omits_it(self):
        response = {
            "id": "resp-test",
            "status": "completed",
            "usage": {
                "input_tokens": 1000,
                "input_tokens_details": {"cached_tokens": 200},
                "output_tokens": 100,
                "total_tokens": 1100,
                "cost": None,
            },
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done."}],
                }
            ],
        }

        message = responses_to_chat_message(response)

        self.assertEqual(message["response_usage"]["cost"], None)
        self.assertEqual(message["token_usage"]["cached_input_tokens"], 200)
        self.assertNotIn("cost", message["token_usage"])

    def test_model_runner_defers_repair_until_after_all_tool_outputs(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taskset = tmp_path / "tasks.jsonl"
            trace = tmp_path / "trace.jsonl"
            db_path = tmp_path / "test.sqlite"
            self._prepare_minimal_model_db(db_path)
            taskset.write_text(
                json.dumps(
                    {
                        "task_id": "fixture-0001",
                        "target_smiles": "CCO",
                        "system_prompt": "You plan syntheses.",
                        "prompt_fingerprint": PROMPT_FINGERPRINT,
                        "question": "Plan CCO.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(
                [
                    "--provider",
                    "openai",
                    "--api-mode",
                    "chat",
                    "--api-key-env",
                    "TEST_AGENTIC_SYNTHESIS_API_KEY",
                    "--taskset",
                    str(taskset),
                    "--task-index",
                    "0",
                    "--db",
                    str(db_path),
                    "--trace",
                    str(trace),
                    "--model",
                    "mock-model",
                    "--max-turns",
                    "2",
                    "--max-tokens",
                    "16",
                ]
            )
            first_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "search_building_blocks", "arguments": "{}"},
                    },
                    {
                        "id": "call_good",
                        "type": "function",
                        "function": {
                            "name": "similarity",
                            "arguments": json.dumps({"smiles_a": "CCO", "smiles_b": "CCO"}),
                        },
                    },
                ],
            }
            seen_messages: list[list[dict]] = []

            def fake_chat_completion(**kwargs):
                seen_messages.append(list(kwargs["messages"]))
                if len(seen_messages) == 1:
                    return {"choices": [{"message": first_message}]}
                return {"choices": [{"message": {"role": "assistant", "content": "No tool call."}}]}

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.chat_completion",
                    side_effect=fake_chat_completion,
                ),
            ):
                run_model_from_args(args)

            second_messages = seen_messages[1]
            call_bad_index = next(
                i
                for i, message in enumerate(second_messages)
                if message.get("tool_call_id") == "call_bad"
            )
            call_good_index = next(
                i
                for i, message in enumerate(second_messages)
                if message.get("tool_call_id") == "call_good"
            )
            self.assertLess(call_bad_index, call_good_index)
            self.assertNotIn("notice", json.loads(second_messages[call_bad_index]["content"]))
            self.assertIn(
                "tool call failed",
                json.loads(second_messages[call_good_index]["content"])["notice"],
            )
            self.assertEqual(second_messages[-1]["tool_call_id"], "call_good")


if __name__ == "__main__":
    unittest.main()


def test_checkpoint_rejects_a_different_submission_floor():
    import argparse

    args = argparse.Namespace(
        model="m",
        provider="p",
        api_mode="chat",
        responses_state_mode=None,
        stop_condition="exact",
    )
    task = {"task_id": "t", "target_smiles": "CCO"}
    checkpoint = {
        "model": "m",
        "provider": "p",
        "api_mode": "chat",
        "task_id": "t",
        "target_smiles": "CCO",
        "stop_condition": 1.0,
    }

    assert not checkpoint_matches_run(checkpoint, args, task, policy.RolloutPolicyConfig())
    checkpoint["stop_condition"] = "exact"
    assert checkpoint_matches_run(checkpoint, args, task, policy.RolloutPolicyConfig())


def test_checkpoint_rejects_a_different_endpoint_or_auth_header():
    import argparse

    args = argparse.Namespace(
        model="m",
        provider="custom",
        api_mode="responses",
        responses_state_mode="previous_id",
        base_url="",
        responses_url="https://a.example/v1/responses",
        messages_url="",
        auth_header="bearer",
    )
    task = {"task_id": "t", "target_smiles": "CCO"}
    checkpoint = {
        "model": "m",
        "provider": "custom",
        "api_mode": "responses",
        "responses_state_mode": "previous_id",
        "responses_previous_id": "resp_1",
        "base_url": "",
        "responses_url": "https://a.example/v1/responses",
        "messages_url": "",
        "auth_header": "bearer",
        "task_id": "t",
        "target_smiles": "CCO",
    }

    assert checkpoint_matches_run(checkpoint, args, task, policy.RolloutPolicyConfig())
    args.responses_url = "https://b.example/v1/responses"
    assert not checkpoint_matches_run(checkpoint, args, task, policy.RolloutPolicyConfig())
    args.responses_url = checkpoint["responses_url"]
    args.auth_header = "x-api-key"
    assert not checkpoint_matches_run(checkpoint, args, task, policy.RolloutPolicyConfig())


def test_cost_is_omitted_when_only_some_responses_report_it():
    from synclaw.token_usage import sum_token_usage

    assert (
        sum_token_usage([{"input_tokens": 1, "cost": 0.5}, {"input_tokens": 2, "cost": 0.25}])[
            "cost"
        ]
        == 0.75
    )
    partial = sum_token_usage([{"input_tokens": 1, "cost": 0.5}, {"input_tokens": 2}])
    assert partial == {"input_tokens": 3}
