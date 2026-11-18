# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc evaluation must agree with the rollout's exact stopping criterion."""

import json

import pytest

from synclaw.evaluation.evaluate import _similarity_floor, evaluate_trace_file


@pytest.mark.parametrize(
    "tool_config", [None, {"stop_condition": "exact"}, {"stop_condition": "model"}]
)
@pytest.mark.parametrize(
    ("target", "product", "accepted", "expected"),
    [
        ("CCCCCCC", "CCCCCCCC", True, False),
        ("O=c1cccc[nH]1", "Oc1ccccn1", True, True),
        ("CCO", "OCC", True, True),
        ("CCO", "OCC", False, False),
    ],
)
def test_exact_trace_evaluation(target, product, accepted, expected, tool_config, tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({
        "event_type": "tool_result", "tool_name": "submit_pathway",
        "payload": {"result": {
            "accepted": accepted, "actual_final_smiles": product,
            "final_smiles": product, "reaction_validation": "valid" if accepted else "invalid",
            "steps": [{"valid": accepted, "actual_product_smiles": product}],
        }},
    }) + "\n")
    result = evaluate_trace_file(
        trace, tmp_path / "eval.sqlite", target_smiles=target,
        tool_config=tool_config,
    )
    # A rollout the model ends is scored like an exact one.
    assert result["similarity_floor"] == (tool_config or {}).get("stop_condition", "exact")
    assert result["accepted"] is accepted
    assert result["meets_similarity_floor"] is expected
    assert (result["identical_to_target"] is True) is expected


@pytest.mark.parametrize("explicit_product", [None, "O=c1ccccn1C"])
def test_generated_product_fallback_shares_identity_selection(tmp_path, explicit_product):
    target = "O=c1cccc[nH]1"
    products = ["O=c1ccccn1C", "Oc1ccccn1"]
    step = {"valid": True, "generated_products": products}
    if explicit_product is not None:
        step["actual_product_smiles"] = explicit_product
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({
        "event_type": "tool_result", "tool_name": "submit_pathway",
        "payload": {"result": {
            "accepted": True, "reaction_validation": "valid", "steps": [step],
        }},
    }) + "\n")
    evaluation = evaluate_trace_file(
        trace, tmp_path / "eval.sqlite", target_smiles=target,
        tool_config={"stop_condition": "exact"},
    )

    # As in submit_pathway, every generated product is a candidate, so the identical tautomer
    # outranks the analogue even when the step recorded the analogue as its actual product.
    assert evaluation["accepted"] is True
    assert evaluation["final_smiles"] == "Oc1ccccn1"
    assert evaluation["identical_to_target"] is True
    assert evaluation["meets_similarity_floor"] is True


@pytest.mark.parametrize("recorded", ["none", None])
def test_malformed_or_missing_recorded_floor_falls_back_to_the_default(recorded):
    assert _similarity_floor({"stop_condition": recorded}) == "exact"
