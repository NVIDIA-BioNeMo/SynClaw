# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replaying a trace must select the submission the run itself selected.

Expectations come from the real rollout policy, and the traces from the loop's own call
preparation and dispatch, so a change to the events the loop records, to the selection
rule, or to how a call is dispatched that the evaluator does not follow fails here.
"""

import itertools
import json

import pytest

from synclaw.core import db, reactions
from synclaw.evaluation.evaluate import evaluate_trace_file
from synclaw.rollout.loop import call_local_tool
from synclaw.rollout.policy import (
    RolloutPolicyConfig,
    RolloutPolicyState,
    handle_tool_result,
    prepare_tool_call,
    selected_submission_event,
    tool_argument_error,
)
from synclaw.tools.synthesis import SynthesisTools

DIOL = "OCCCCC(C)O"  # BB:1; oxidation gives an aldehyde or a ketone
KETONE = "CC(=O)CCCCO"
OXIDIZE = "demo:R000"

SCORED_FIELDS = (
    "accepted",
    "final_smiles",
    "submitted_final_smiles",
    "reaction_validation",
    "building_block_grounded",
    "referenced_building_block_ids",
    "similarity_to_target",
    "identical_to_target",
    "meets_similarity_floor",
)


def _step(reaction_id: str, reactant: str, product: str) -> dict[str, object]:
    return {"reaction_id": reaction_id, "reactants": [reactant], "product_smiles": product}


# One call of each kind the selection rule tells apart, against the target KETONE. A dict
# is the call's arguments; a string is the raw text of arguments that do not parse.
CALLS = {
    "exact": {"steps": [_step(OXIDIZE, "BB:1", KETONE)]},
    "analogue": {"steps": [], "building_block": "BB:1"},
    "rejected": {"steps": [_step(OXIDIZE, "BB:1", "CCC")]},
    # The run rejects an unknown template, and replay cannot resolve it either.
    "unreplayable": {"steps": [_step("retired:R999", "BB:1", "CCCC")]},
    # The run rejects steps that are not objects, and replay dispatches them as the run did.
    "malformed": {"steps": ["not a step"]},
    # The run never dispatches this call; its result is the recorded argument error.
    "unparseable": '{"steps": [',
    # Past the tool-call budget the run records the call but never executes it. Replayed,
    # it would be the best submission, so scoring it would differ from the run.
    "over_budget": {"steps": [_step(OXIDIZE, "BB:1", KETONE)]},
}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("submission-agreement")
    path = root / "fixture.sqlite"
    templates = root / "templates.txt"
    stock = root / "stock.txt"
    templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
    stock.write_text(f"SMILES\n{DIOL}\n", encoding="utf-8")
    reactions.ingest_reaction_templates(templates, db_path=path, library="demo")
    db.ingest_building_blocks(stock, path)
    return path


@pytest.fixture(scope="module")
def tools(db_path):
    return SynthesisTools(db_path)


def _evaluate(path, events, db_path, target):
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return evaluate_trace_file(path, db_path, target_smiles=target)


def _tool_event(kind, name, target, payload):
    return {"event_type": kind, "tool_name": name, "target_smiles": target, "payload": payload}


@pytest.mark.parametrize("kinds", list(itertools.product(CALLS, repeat=3)), ids="-".join)
def test_replayed_trace_scores_the_submission_the_run_selected(
    tools, db_path, tmp_path, kinds,
):
    """Without a selected_submission event, the evaluator must still score the run's choice."""

    target = KETONE
    config = RolloutPolicyConfig(stop_condition="model")
    state = RolloutPolicyState()
    events = [
        {"event_type": "message", "role": "user", "target_smiles": target, "payload": {"text": target}}
    ]
    for index, kind in enumerate(kinds):
        raw = CALLS[kind]
        if isinstance(raw, dict):
            raw = {**raw, "pathway_text": f"call {index}: {kind}"}
        call = prepare_tool_call(
            raw_name="submit_pathway",
            raw_arguments=raw if isinstance(raw, str) else json.dumps(raw),
            raw_tool_call={"id": f"call-{index}"},
            event_index=index,
        )
        if kind == "over_budget":
            # As rollout.loop does: the result is recorded, but the policy never sees it.
            result = tool_argument_error(
                "submit_pathway", "budget reached", call.dispatch_arguments
            )
            outcome = "over_budget"
        elif call.argument_error is not None:
            result = tool_argument_error("submit_pathway", call.argument_error, call.arguments)
            outcome = "argument_error"
        else:
            result = call_local_tool(
                tools, "submit_pathway", call.dispatch_arguments, target_smiles=target
            )
            outcome = "ok"
        if kind != "over_budget":
            handle_tool_result(
                config, state, name="submit_pathway", arguments=call.dispatch_arguments,
                result=result, target_smiles=target,
            )
        events += [
            _tool_event("tool_call", "submit_pathway", target, call.event_payload),
            _tool_event(
                "tool_result", "submit_pathway", target, {"result": result, "outcome": outcome}
            ),
        ]
    replayed = _evaluate(tmp_path / "replayed.jsonl", events, db_path, target)
    submission, arguments = state.selected_submission()
    if submission is None:
        # The run executed no submission, so there is nothing to score.
        assert replayed["has_final_submission"] is False
        return
    with_selection = [*events, selected_submission_event(
        task={"target_smiles": target}, submission=submission,
        submission_arguments=arguments, reason="test",
    )]

    selected = _evaluate(tmp_path / "selected.jsonl", with_selection, db_path, target)

    assert {key: replayed[key] for key in SCORED_FIELDS} == {
        key: selected[key] for key in SCORED_FIELDS
    }
