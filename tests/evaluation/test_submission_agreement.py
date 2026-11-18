# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc evaluation must reproduce the product and the submission the run itself chose.

Expectations come from the real submit_pathway tool and the real rollout policy, not
from fixed answers, so a change on either side that the evaluator does not follow fails
here: a new or dropped step-result field, a different candidate or tie-break rule, or a
different selection rule.
"""

import itertools
import json

import pytest

from synclaw.core import db, reactions
from synclaw.evaluation.evaluate import evaluate_trace_file
from synclaw.rollout.policy import (
    RolloutPolicyConfig,
    RolloutPolicyState,
    append_selected_submission_event,
    handle_tool_result,
    tool_argument_error,
)
from synclaw.tools.synthesis import SynthesisTools

DIOL = "OCCCCC(C)O"  # BB:1; oxidation gives an aldehyde or a ketone
ETHER = "CCOC"  # BB:2; cleavage gives two product sets, each with a coproduct
ALDEHYDE = "CC(O)CCCC=O"
KETONE = "CC(=O)CCCCO"
OXIDIZE = "demo:R000"
CLEAVE = "demo:R001"

# Fields from which a result reports its final product directly; without them the
# evaluator must choose the product from the recorded steps.
FINAL_PRODUCT_FIELDS = {"actual_final_smiles", "final_smiles", "final", "actual_final_products"}
# Every field of a step result that records a product the step made.
STEP_PRODUCT_FIELDS = (
    "actual_product_smiles", "product", "generated_products", "generated_product_sets",
)
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


# Steps and target: several candidates with and without a target, coproduct sets,
# an intermediate, and rejections with and without generated products.
SUBMISSIONS = {
    "aldehyde-no-target": ([_step(OXIDIZE, "BB:1", ALDEHYDE)], None),
    "ketone-no-target": ([_step(OXIDIZE, "BB:1", KETONE)], None),
    "aldehyde-ketone-target": ([_step(OXIDIZE, "BB:1", ALDEHYDE)], KETONE),
    "aldehyde-unrelated-target": ([_step(OXIDIZE, "BB:1", ALDEHYDE)], "Oc1ccccc1"),
    "coproduct-sets": ([_step(CLEAVE, "BB:2", "CO")], "CCO"),
    "intermediate": (
        [_step(OXIDIZE, "BB:1", ALDEHYDE), _step(OXIDIZE, "INT:1", "CC(=O)CCCC=O")], None,
    ),
    "product-not-generated": ([_step(OXIDIZE, "BB:1", "CCC")], KETONE),
    "ungrounded-reactant": ([_step(OXIDIZE, "CCCCO", "CCCC=O")], None),
}

# One call of each kind the selection rule tells apart, against the target KETONE.
CALLS = {
    "exact": {"steps": [_step(OXIDIZE, "BB:1", KETONE)]},
    "analogue": {"steps": [], "building_block": "BB:1"},
    "rejected": {"steps": [_step(OXIDIZE, "BB:1", "CCC")]},
    # The run rejects an unknown template, and replay cannot resolve it either.
    "unreplayable": {"steps": [_step("retired:R999", "BB:1", "CCCC")]},
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
    templates.write_text(
        "[C:1][O:2]>>[C:1]=[O:2]\n[C:1][O:2][C:3]>>[C:1][O:2].[C:3]\n", encoding="utf-8"
    )
    stock.write_text(f"SMILES\n{DIOL}\n{ETHER}\n", encoding="utf-8")
    reactions.ingest_reaction_templates(templates, db_path=path, library="demo")
    db.ingest_building_blocks(stock, path)
    return path


@pytest.fixture(scope="module")
def tools(db_path):
    return SynthesisTools(db_path)


def _evaluate(path, events, db_path, target):
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return evaluate_trace_file(path, db_path, target_smiles=target)


def _result_event(result):
    return {"event_type": "tool_result", "tool_name": "submit_pathway", "payload": {"result": result}}


@pytest.mark.parametrize("name", SUBMISSIONS)
def test_recorded_steps_yield_the_product_submit_pathway_chose(tools, db_path, tmp_path, name):
    steps, target = SUBMISSIONS[name]
    result = tools.submit_pathway(steps=steps, pathway_text=name, target_smiles=target)
    recorded = {key: value for key, value in result.items() if key not in FINAL_PRODUCT_FIELDS}

    summary = _evaluate(tmp_path / "trace.jsonl", [_result_event(recorded)], db_path, target)

    # A rejected route is not scored, so only an accepted one keeps its product.
    assert summary["accepted"] is result["accepted"]
    assert summary["final_smiles"] == (result["actual_final_smiles"] if result["accepted"] else None)
    assert summary["identical_to_target"] == result["identical_to_target"]


@pytest.mark.parametrize("field", STEP_PRODUCT_FIELDS)
@pytest.mark.parametrize("name", ["aldehyde-no-target", "coproduct-sets"])
def test_any_recorded_step_product_field_yields_a_product(tools, db_path, tmp_path, name, field):
    """An older or trimmed trace may keep only some of a step's product fields."""

    steps, target = SUBMISSIONS[name]
    result = tools.submit_pathway(steps=steps, pathway_text=name, target_smiles=target)
    final_step = result["steps"][-1]
    # If submit_pathway stops recording this field, update STEP_PRODUCT_FIELDS and
    # final_product_candidates together.
    assert final_step.get(field)
    recorded = {
        "accepted": True,
        "reaction_validation": "valid",
        "steps": [{"valid": True, field: final_step[field]}],
    }

    summary = _evaluate(tmp_path / "trace.jsonl", [_result_event(recorded)], db_path, target)

    assert summary["final_smiles"] in result["actual_final_products"]


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
        arguments = {**CALLS[kind], "pathway_text": f"call {index}: {kind}"}
        if kind == "over_budget":
            # As rollout.loop does: the result is recorded, but the policy never sees it.
            result = tool_argument_error("submit_pathway", "budget reached", arguments)
            recorded = {"result": result, "outcome": "over_budget"}
        else:
            result = tools.submit_pathway(**arguments, target_smiles=target)
            handle_tool_result(
                config, state, name="submit_pathway", arguments=arguments, result=result,
                target_smiles=target,
            )
            recorded = {"result": result}
        events += [
            {"event_type": "tool_call", "tool_name": "submit_pathway", "payload": {"arguments": arguments}},
            {"event_type": "tool_result", "tool_name": "submit_pathway", "payload": recorded},
        ]
    replayed = _evaluate(tmp_path / "replayed.jsonl", events, db_path, target)
    submission, arguments = state.selected_submission()
    if submission is None:
        # The run executed no submission, so there is nothing to score.
        assert replayed["has_final_submission"] is False
        return
    with_selection = list(events)
    append_selected_submission_event(
        with_selection, task={"target_smiles": target}, submission=submission,
        submission_arguments=arguments, reason="test",
    )

    selected = _evaluate(tmp_path / "selected.jsonl", with_selection, db_path, target)

    assert {key: replayed[key] for key in SCORED_FIELDS} == {
        key: selected[key] for key in SCORED_FIELDS
    }


@pytest.mark.parametrize("field", [*STEP_PRODUCT_FIELDS, "actual_final_smiles"])
def test_product_fields_in_call_arguments_are_never_scored(db_path, tmp_path, field):
    """Only a submit_pathway result carries a validated product; the model writes the arguments."""

    step = {**_step("retired:R999", "BB:1", KETONE), "valid": True}
    arguments = {"steps": [step], "pathway_text": "claimed", "accepted": True}
    if field == "actual_final_smiles":
        arguments[field] = KETONE
    else:
        step[field] = {"valid": True, "canonical_smiles": KETONE} if field == "product" else (
            KETONE if field == "actual_product_smiles"
            else [[KETONE]] if field == "generated_product_sets" else [KETONE]
        )
    call = {"event_type": "tool_call", "tool_name": "submit_pathway", "payload": {"arguments": arguments}}

    summary = _evaluate(tmp_path / "trace.jsonl", [call], db_path, KETONE)

    assert summary["final_smiles"] is None
    assert summary["meets_similarity_floor"] is False


@pytest.mark.parametrize("name", SUBMISSIONS)
def test_a_submission_without_a_verdict_is_not_ranked_against_the_target(
    tools, db_path, tmp_path, name,
):
    """Without a verdict, the target must not pick a product the run may not have made."""

    steps, target = SUBMISSIONS[name]
    result = tools.submit_pathway(steps=steps, pathway_text=name, target_smiles=target)
    recorded = {
        key: value for key, value in result.items()
        if key not in FINAL_PRODUCT_FIELDS and key != "accepted"
    }
    targetless = tools.submit_pathway(steps=steps, pathway_text=name, target_smiles=None)

    summary = _evaluate(tmp_path / "trace.jsonl", [_result_event(recorded)], db_path, target)

    assert summary["accepted"] is None
    assert summary["final_smiles"] == targetless["actual_final_smiles"]
