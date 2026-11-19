# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

from synclaw.paths import REPO_ROOT


def _load_renderer():
    script = REPO_ROOT / "scripts" / "render_tool_trace.py"
    spec = importlib.util.spec_from_file_location("render_tool_trace", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_shows_budget_snapshots_separately_from_billing_and_feedback():
    renderer = _load_renderer()
    reminder = (
        "Your context-window budget is 1050000 tokens, shared by the conversation. "
        "Remaining: ~1040k context tokens."
    )
    events = [
        {"event_type": "message", "role": "user", "payload": {"content": reminder}},
        {
            "event_type": "token_budget",
            "payload": {
                "turn_index": 0,
                "context_window_tokens": 1_050_000,
                "input_tokens": 10_000,
                "remaining_tokens": 1_040_000,
                "max_output_tokens": 32_768,
                "estimated": True,
                "source": "utf8_estimate",
                "status": "emitted",
            },
        },
        {
            "event_type": "token_budget",
            "payload": {
                "turn_index": 1,
                "context_window_tokens": 1_050_000,
                "input_tokens": None,
                "remaining_tokens": None,
                "max_output_tokens": 32_768,
                "estimated": True,
                "source": "unavailable",
                "status": "unavailable_or_unchanged",
            },
        },
    ]
    markdown = renderer.render_trace(events, trace_path=Path("trace.jsonl"))
    assert "## Token Budget" in markdown
    assert (
        "| 1 | 1,050,000 | 10,000 | 1,040,000 | 32,768 | estimated: `utf8_estimate` | `emitted` |"
        in markdown
    )
    assert "| 2 | 1,050,000 | — | — | 32,768 |" in markdown
    assert "- Token usage:" not in markdown
    assert "## Harness Interventions" in markdown
    assert f"- {reminder}" in markdown
    assert renderer.render_harness_interventions(
        [
            {
                "event_type": "message",
                "role": "user",
                "payload": {
                    "content": "Remaining: 79 turns, 145 tool calls, ~1040k context tokens."
                },
            },
            {
                "event_type": "message",
                "role": "user",
                "payload": {
                    "content": "Your previous response hit its output-token limit and was truncated."
                },
            },
        ]
    ) == [
        "- Remaining: 79 turns, 145 tool calls, ~1040k context tokens.",
        "- Your previous response hit its output-token limit and was truncated.",
    ]
    assert "## Token Budget" not in renderer.render_trace([], trace_path=Path("old.jsonl"))


def test_report_shows_attached_feedback_and_effective_output_cap():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "token_budget",
            "payload": {
                "turn_index": 1,
                "context_window_tokens": 262144,
                "input_tokens": 231935,
                "remaining_tokens": 30209,
                "max_output_tokens": 32768,
                "request_output_tokens": 7015,
                "estimated": True,
                "source": "provider_calibrated_utf8",
                "status": "emitted",
                "content": "Remaining: 5 turns. This response is limited to 7015 output tokens.",
                "message_role": "tool",
            },
        },
        {"event_type": "harness_feedback", "payload": {"content": "Repair the pathway."}},
    ]
    text = renderer.render_trace(events, trace_path=Path("fixture.jsonl"))
    assert "| 2 | 262,144 | 231,935 | 30,209 | 7,015 |" in text
    assert "- Remaining: 5 turns. This response is limited to 7015 output tokens." in text
    assert "- Repair the pathway." in text


def test_summary_includes_final_molecule_similarity_computed_from_trace():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "message",
            "role": "user",
            "target_smiles": "CCO",
            "payload": {"content": "Plan CCO"},
        },
        {
            "event_type": "tool_result",
            "tool_name": "submit_pathway",
            "target_smiles": "CCO",
            "payload": {
                "result": {
                    "accepted": True,
                    "actual_final_smiles": "CCO",
                    "final_smiles": "CCO",
                    "reaction_validation": "valid",
                    "similarity_to_target": None,
                }
            },
        },
    ]

    markdown = renderer.render_trace(events, trace_path=Path("trace.jsonl"))

    assert "- Active final-molecule similarity to target: `1.0`" in markdown
    assert "- Similarity to target: `None`" not in markdown


def test_renderer_shows_zero_step_direct_purchase_without_validated_step():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "tool_call",
            "tool_name": "submit_pathway",
            "target_smiles": "CO",
            "payload": {
                "dispatch_arguments": {
                    "steps": [],
                    "building_block": "BB:1",
                    "pathway_text": "Direct purchase.",
                }
            },
        },
        {
            "event_type": "tool_result",
            "tool_name": "submit_pathway",
            "target_smiles": "CO",
            "payload": {
                "result": {
                    "accepted": True,
                    "actual_final_smiles": "CO",
                    "final_smiles": "CO",
                    "pathway_type": "direct_purchase",
                    "direct_building_block_ref": "BB:1",
                    "direct_building_block": {
                        "id": 1,
                        "submit_reference": "BB:1",
                        "canonical_smiles": "CO",
                    },
                    "reaction_validation": "valid",
                    "building_block_grounded": True,
                    "referenced_building_block_ids": [1],
                    "steps": [],
                    "pathway_text": "Direct purchase.",
                }
            },
        },
    ]

    markdown = renderer.render_trace(events, trace_path=Path("trace.jsonl"))

    assert "Direct-purchase building block: `BB:1`" in markdown
    assert "Pathway type: `direct_purchase`" in markdown
    assert "### Validated Step 1" not in markdown


def test_summary_can_fall_back_to_run_summary_similarity():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "run_summary",
            "payload": {
                "evaluation_summary": {
                    "final_smiles": "not-a-smiles",
                    "similarity_to_target": 0.42,
                }
            },
        }
    ]

    markdown = renderer.render_trace(events, trace_path=Path("trace.jsonl"))

    assert "- Active final-molecule similarity to target: `0.42`" in markdown


def test_summary_and_turns_render_token_usage():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "message",
            "role": "assistant",
            "payload": {
                "content": "I will inspect the route.",
                "token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "cached_input_tokens": 1,
                    "reasoning_tokens": 2,
                    "cost": 0.001,
                },
            },
        },
        {
            "event_type": "message",
            "role": "assistant",
            "payload": {
                "content": "I will submit.",
                "response_usage": {
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_tokens": 28,
                    "cost": 0.003,
                },
            },
        },
    ]

    markdown = renderer.render_trace(events, trace_path=Path("trace.jsonl"))

    assert (
        "- Token usage: input `30`, output `13`, total `43`, cached input `1`, reasoning `2`, cost `0.004`"
        in markdown
    )
    assert (
        "**Token usage:** input `10`, output `5`, total `15`, cached input `1`, reasoning `2`, cost `0.001`"
        in markdown
    )
    assert "**Token usage:** input `20`, output `8`, total `28`, cost `0.003`" in markdown


def test_report_never_estimates_cost_when_provider_omits_it():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "message",
            "role": "assistant",
            "payload": {
                "content": "I will inspect the route.",
                "response_usage": {
                    "input_tokens": 1000,
                    "input_tokens_details": {"cached_tokens": 200},
                    "output_tokens": 100,
                    "total_tokens": 1100,
                    "cost": None,
                },
            },
        },
    ]

    markdown = renderer.render_trace(
        events,
        trace_path=Path("trace.jsonl"),
        model="example-model",
    )

    assert (
        "- Token usage: input `1000`, output `100`, total `1100`, cached input `200`, cache read `200`"
        in markdown
    )
    assert (
        "**Token usage:** input `1000`, output `100`, total `1100`, cached input `200`, cache read `200`"
        in markdown
    )
    assert "cost" not in markdown


def test_validator_outcome_has_single_blank_lines_around_validated_steps():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "tool_call",
            "tool_name": "submit_pathway",
            "payload": {
                "dispatch_arguments": {
                    "steps": [
                        {"reaction_id": "demo:R000", "reactants": ["BB:1"], "product_smiles": "CCO"}
                    ],
                    "pathway_text": "Make ethanol.",
                },
                "raw_tool_call": {"id": "call_1"},
            },
        },
        {
            "event_type": "tool_result",
            "tool_name": "submit_pathway",
            "target_smiles": "CCO",
            "payload": {
                "result": {
                    "accepted": True,
                    "reaction_validation": "valid",
                    "building_block_grounded": True,
                    "referenced_building_block_ids": [1],
                    "actual_final_smiles": "CCO",
                    "steps": [
                        {
                            "valid": True,
                            "reaction_id": "demo:R000",
                            "generated_products": ["CCO"],
                            "resolved_reactants": [
                                {
                                    "ref": "BB:1",
                                    "smiles": "CCO",
                                    "source": "building_block",
                                    "building_block": {"id": 1},
                                }
                            ],
                        }
                    ],
                }
            },
        },
    ]

    markdown = renderer.render_trace(events, trace_path=Path("trace.jsonl"))

    assert "- Issues: none\n\n### Validated Step 1" in markdown
    assert "- Issues: none\n### Validated Step 1" not in markdown
    assert "\n\n\n## Tool Interaction" not in markdown
    assert "BB id `1`)\n\n## Tool Interaction" in markdown


def test_submitted_pathway_uses_selected_submission_result_not_last_call():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "tool_call",
            "tool_name": "submit_pathway",
            "payload": {
                "dispatch_arguments": {
                    "steps": [
                        {"reaction_id": "demo:R001", "reactants": ["BB:1"], "product_smiles": "CCO"}
                    ],
                    "pathway_text": "Selected route.",
                }
            },
        },
        {
            "event_type": "tool_result",
            "tool_name": "submit_pathway",
            "payload": {
                "result": {
                    "accepted": True,
                    "reaction_validation": "valid",
                    "actual_final_smiles": "CCO",
                    "pathway_text": "Selected route.",
                    "steps": [
                        {
                            "valid": True,
                            "reaction_id": "demo:R001",
                            "submitted_step": {
                                "reaction_id": "demo:R001",
                                "reactants": ["BB:1"],
                                "product_smiles": "CCO",
                            },
                        }
                    ],
                }
            },
        },
        {
            "event_type": "tool_call",
            "tool_name": "submit_pathway",
            "payload": {
                "dispatch_arguments": {
                    "steps": [
                        {"reaction_id": "demo:R002", "reactants": ["BB:2"], "product_smiles": "CCN"}
                    ],
                    "pathway_text": "Later route.",
                }
            },
        },
        {
            "event_type": "tool_result",
            "tool_name": "submit_pathway",
            "payload": {
                "result": {
                    "accepted": True,
                    "reaction_validation": "valid",
                    "actual_final_smiles": "CCN",
                    "pathway_text": "Later route.",
                    "steps": [
                        {
                            "valid": True,
                            "reaction_id": "demo:R002",
                            "submitted_step": {
                                "reaction_id": "demo:R002",
                                "reactants": ["BB:2"],
                                "product_smiles": "CCN",
                            },
                        }
                    ],
                }
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
                    "actual_final_smiles": "CCO",
                    "pathway_text": "Selected route.",
                    "steps": [
                        {
                            "valid": True,
                            "reaction_id": "demo:R001",
                            "submitted_step": {
                                "reaction_id": "demo:R001",
                                "reactants": ["BB:1"],
                                "product_smiles": "CCO",
                            },
                        }
                    ],
                },
            },
        },
    ]

    markdown = renderer.render_trace(events, trace_path=Path("trace.jsonl"))

    submitted_section = markdown.split("## Submitted Pathway", 1)[1].split(
        "## Validator Outcome", 1
    )[0]
    assert "Selected route." in submitted_section
    assert "`demo:R001`" in submitted_section
    assert "`demo:R002`" not in submitted_section


def test_simulate_reverse_reaction_renders_readable_result():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "tool_call",
            "tool_name": "simulate_reverse_reaction",
            "payload": {
                "dispatch_arguments": {
                    "reaction_id": "demo:R000",
                    "product_smiles": "C=O",
                    "max_outcomes": 5,
                    "include_near": True,
                },
                "raw_tool_call": {"id": "call_1"},
            },
        },
        {
            "event_type": "tool_result",
            "tool_name": "simulate_reverse_reaction",
            "payload": {
                "result": {
                    "valid": True,
                    "reaction_id": "demo:R000",
                    "product_canonical_smiles": "C=O",
                    "returned_candidate_count": 1,
                    "template_summary": {
                        "reaction_family": "oxidation",
                        "description": "Convert an alcohol-like reactant pattern to a carbonyl-like product.",
                    },
                    "reverse_planning_summary": {
                        "fully_grounded_candidate_sets": 1,
                        "recommended_next_step": "Verify it forward with simulate_reaction before submitting.",
                    },
                    "building_block_match_summary": {
                        "exact_match_count": 1,
                        "returned_near_match_count": 1,
                    },
                    "candidate_reactant_sets": [
                        {
                            "reactants": ["CO"],
                            "all_reactants_have_exact_building_block_match": True,
                            "reactant_details": [
                                {
                                    "smiles": "CO",
                                    "exact_building_block_matches": [
                                        {"id": 1, "submit_reference": "BB:1", "similarity": 1.0}
                                    ],
                                    "top_near_building_block_matches": [
                                        {"id": 2, "submit_reference": "BB:2", "similarity": 0.66}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            },
        },
    ]

    markdown = renderer.render_trace(events, trace_path=Path("trace.jsonl"))

    assert "### 1. `simulate_reverse_reaction`" in markdown
    assert "- Products: `C=O`" in markdown
    assert "**Result:** valid `true`, 1 candidate precursor set(s)" in markdown
    assert "- Candidate precursor sets:" in markdown
    assert "Set 1: `CO` (fully grounded `true`)" in markdown
    assert "`CO`; exact `BB:1` sim `1.0`; near `BB:2` sim `0.66`" in markdown
    assert '**Result:** `{"candidate_reactant_sets"' not in markdown


def test_similarity_call_renders_ordered_markdown_arguments():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "message",
            "role": "assistant",
            "payload": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "similarity",
                            "arguments": (
                                '{"smiles_b":"Cc1nc(-c2ccncc2)sc1CC#N",'
                                '"smiles_a":"Cc1nsc(-c2ccc[n+](CC#N)c2)n1"}'
                            ),
                        }
                    }
                ]
            },
        },
        {
            "event_type": "tool_call",
            "tool_name": "similarity",
            "payload": {
                "dispatch_arguments": {
                    "smiles_a": "Cc1nsc(-c2ccc[n+](CC#N)c2)n1",
                    "smiles_b": "Cc1nc(-c2ccncc2)sc1CC#N",
                },
                "raw_tool_call": {"id": "call_1"},
            },
        },
        {
            "event_type": "tool_result",
            "tool_name": "similarity",
            "payload": {
                "result": {
                    "valid": True,
                    "method": "rdkit_morgan_tanimoto",
                    "similarity": 0.75,
                    "left": {"canonical_smiles": "Cc1nsc(-c2ccc[n+](CC#N)c2)n1"},
                    "right": {"canonical_smiles": "Cc1nc(-c2ccncc2)sc1CC#N"},
                }
            },
        },
    ]

    markdown = renderer.render_trace(events, trace_path=Path("trace.jsonl"))

    assert "- `similarity`" in markdown
    assert (
        '- arguments: `{"smiles_a":"Cc1nsc(-c2ccc[n+](CC#N)c2)n1",'
        '"smiles_b":"Cc1nc(-c2ccncc2)sc1CC#N"}`'
    ) in markdown
    assert (
        '- Arguments: `{"smiles_a":"Cc1nsc(-c2ccc[n+](CC#N)c2)n1",'
        '"smiles_b":"Cc1nc(-c2ccncc2)sc1CC#N"}`'
    ) in markdown
    assert "**Result:** valid `true`, similarity `0.75`" in markdown
    assert '**Call:** `{"smiles_a"' not in markdown


def test_tool_interaction_reports_model_visible_compaction():
    renderer = _load_renderer()
    events = [
        {
            "event_type": "tool_call",
            "tool_name": "search_building_blocks",
            "payload": {
                "dispatch_arguments": {"query": "CCO"},
                "raw_tool_call": {"id": "call_1"},
            },
        },
        {
            "event_type": "tool_result",
            "tool_name": "search_building_blocks",
            "payload": {
                "result": {
                    "count": 1,
                    "results": [{"id": 1, "smiles": "CCO", "match_type": "exact"}],
                },
                "model_result": {
                    "count": 1,
                    "results": [{"id": 1, "smiles": "CCO", "match_type": "exact"}],
                },
                "model_result_compaction": {
                    "full_json_bytes": 1000,
                    "model_json_bytes": 250,
                },
            },
        },
    ]

    markdown = renderer.render_trace(events, trace_path=Path("trace.jsonl"))

    assert "**Model-visible result:** compacted from `1000` to `250` JSON byte(s)" in markdown
    assert "**Result:** 1 building-block hits" in markdown
