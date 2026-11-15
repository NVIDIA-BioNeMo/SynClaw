# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

from synclaw.rollout.policy import prepare_tool_call
from synclaw.tools.contracts import (
    MESSAGES_TOOL_SCHEMAS,
    RESPONSES_TOOL_SCHEMAS,
    TOOL_ARGUMENT_DEFAULTS,
    TOOL_SCHEMAS,
    filter_chat_tool_schemas,
)
from synclaw.tools.synthesis import SynthesisTools


def test_transport_schemas_preserve_one_canonical_contract() -> None:
    chat_by_name = {tool["function"]["name"]: tool["function"] for tool in TOOL_SCHEMAS}
    responses_by_name = {tool["name"]: tool for tool in RESPONSES_TOOL_SCHEMAS}
    messages_by_name = {tool["name"]: tool for tool in MESSAGES_TOOL_SCHEMAS}

    assert chat_by_name.keys() == responses_by_name.keys() == messages_by_name.keys()
    for name, function in chat_by_name.items():
        assert responses_by_name[name]["description"] == function["description"]
        assert responses_by_name[name]["parameters"] == function["parameters"]
        assert messages_by_name[name]["description"] == function["description"]
        assert messages_by_name[name]["input_schema"] == function["parameters"]


def test_tool_descriptions_carry_the_rules_prompts_leave_out() -> None:
    descriptions = {tool["function"]["name"]: tool["function"]["description"] for tool in TOOL_SCHEMAS}

    assert "not purchasable in this database" in descriptions["search_building_blocks"]
    assert "intermediate" in descriptions["search_building_blocks"]
    assert "functional-group transformation" in descriptions["search_reaction_templates"]
    assert "reactant-role order" in descriptions["simulate_reaction"]
    assert "may still be made in earlier steps" in descriptions["simulate_reverse_reaction"]
    assert "not guaranteed to work forward" in descriptions["simulate_reverse_reaction"]
    for rule in ("BB:<id>", "INT:<step_number>", "steps: []", "best accepted route is kept"):
        assert rule in descriptions["submit_pathway"]


def test_tool_method_defaults_match_the_contract() -> None:
    for name, defaults in TOOL_ARGUMENT_DEFAULTS.items():
        parameters = inspect.signature(getattr(SynthesisTools, name)).parameters
        for key, value in defaults.items():
            if key != "steps":
                assert parameters[key].default == value, (name, key)


def test_filtered_schemas_are_independent_copies() -> None:
    schemas = filter_chat_tool_schemas({"mces"})
    schemas[0]["function"]["description"] = "changed"

    assert all(tool["function"]["name"] != "mces" for tool in schemas)
    assert TOOL_SCHEMAS[0]["function"]["description"] != "changed"


def test_prepare_tool_call_applies_canonical_defaults_before_dispatch() -> None:
    prepared = prepare_tool_call(
        raw_name="search_building_blocks",
        raw_arguments={"query": "CCO"},
        raw_tool_call={"id": "call-1"},
        event_index=1,
    )

    assert prepared.dispatch_arguments == {
        "query": "CCO",
        "limit": 10,
        "include_near": True,
        "min_similarity": 0.15,
    }
    assert prepared.event_payload["defaults_applied"] == {
        "limit": 10,
        "include_near": True,
        "min_similarity": 0.15,
    }


def test_prepare_tool_call_ignores_non_contract_arguments_consistently() -> None:
    prepared = prepare_tool_call(
        raw_name="simulate_reaction",
        raw_arguments={
            "reaction_id": "synformer:R001",
            "reactants": ["CCO"],
            "provider_only": "ignored",
        },
        raw_tool_call={"id": "call-1"},
        event_index=1,
    )

    assert "provider_only" not in prepared.dispatch_arguments
    assert prepared.event_payload["ignored_arguments"] == {
        "provider_only": "ignored"
    }
    assert prepared.dispatch_arguments["max_products"] == 100
    assert prepared.dispatch_arguments["try_permutations"] is True
