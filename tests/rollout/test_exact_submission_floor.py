# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An exact stopping floor must use validated molecular identity, not fingerprints."""

import pytest

from synclaw.core import chem
from synclaw.rollout.policy import (
    NON_TOOL_PERSISTENCE_GUARANTEE_SENTENCE,
    RolloutPolicyConfig,
    RolloutPolicyState,
    handle_no_tool_response,
    handle_tool_result,
    parse_stop_condition,
)
from synclaw.tools.synthesis import score_pathway_against_target


def test_exact_is_the_default_floor():
    assert RolloutPolicyConfig().stop_condition == "exact"


# One pair per distinguishable outcome: identical, similarity 1.0 without identity,
# and an ordinary analogue. The full pair table lives in tests/core/test_molecular_identity.py.
@pytest.mark.parametrize("floor", [1.0, "exact"])
@pytest.mark.parametrize(
    ("target", "product", "identical"),
    [
        ("O=c1cccc[nH]1", "Oc1ccccn1", True),
        ("CCCCCCC", "CCCCCCCC", False),
        ("CCO", "CCN", False),
    ],
)
def test_stopping_uses_the_same_identity_as_scoring(floor, target, product, identical):
    submission = {
        "accepted": True,
        **score_pathway_against_target(
            [product],
            target_smiles=target,
            target_validation=chem.validate_smiles(target),
            preferred_smiles=None,
            accepted=True,
        ),
    }
    assert submission["identical_to_target"] is identical
    decision = handle_tool_result(
        RolloutPolicyConfig(**({} if floor is None else {"stop_condition": floor})),
        RolloutPolicyState(),
        name="submit_pathway",
        arguments={"steps": [], "building_block": "BB:1"},
        result=submission,
        target_smiles=target,
    )
    expected_stop = identical or (floor == 1.0 and submission["similarity_to_target"] >= 1.0)
    assert (decision.terminal_submission is not None) is expected_stop
    if not expected_stop:
        if submission["similarity_to_target"] == 1.0:
            assert "did not establish an exact match" in decision.repair
            assert "verify an exact synthesis route before submitting again." in decision.repair
        else:
            numeric_decision = handle_tool_result(
                RolloutPolicyConfig(stop_condition=1.0),
                RolloutPolicyState(),
                name="submit_pathway",
                arguments={"steps": [], "building_block": "BB:1"},
                result=submission,
                target_smiles=target,
            )
            assert decision.repair == numeric_decision.repair
            assert "A better reconstruction is guaranteed to be possible" in decision.repair


def test_unverified_identity_never_meets_exact_floor(inchi_unavailable):
    decision = handle_tool_result(
        RolloutPolicyConfig(stop_condition="exact"),
        RolloutPolicyState(),
        name="submit_pathway",
        arguments={},
        target_smiles="CCO",
        result={
            "accepted": True,
            "final_smiles": "OCC",
            "similarity_to_target": 1.0,
            "identical_to_target": True,
        },
    )
    assert decision.terminal_submission is None


def test_cached_identity_does_not_override_the_validated_product():
    decision = handle_tool_result(
        RolloutPolicyConfig(stop_condition="exact"),
        RolloutPolicyState(),
        name="submit_pathway",
        arguments={},
        target_smiles="CCCCCCC",
        result={
            "accepted": True,
            "actual_final_smiles": "CCCCCCCC",
            "final_smiles": "CCCCCCC",
            "similarity_to_target": 1.0,
            "identical_to_target": True,
        },
    )
    assert decision.terminal_submission is None


def test_exact_floor_keeps_persistence_at_perfect_fingerprint_similarity():
    state = RolloutPolicyState(
        total_tool_calls=2,
        best_accepted_similarity=1.0,
        best_accepted_submission={"accepted": True, "final_smiles": "CCCCCCCC"},
    )
    decision = handle_no_tool_response(
        RolloutPolicyConfig(stop_condition="exact"),
        state,
        content="This is close enough.",
    )
    assert NON_TOOL_PERSISTENCE_GUARANTEE_SENTENCE in decision.reminder


def test_exact_floor_has_no_pathway_to_persist_with_before_a_submission():
    decision = handle_no_tool_response(
        RolloutPolicyConfig(stop_condition="exact"),
        RolloutPolicyState(total_tool_calls=1),
        content="still thinking",
    )
    assert NON_TOOL_PERSISTENCE_GUARANTEE_SENTENCE not in decision.reminder


def test_exact_floor_still_returns_best_valid_analogue_at_tool_limit():
    submission = {"accepted": True, "final_smiles": "CCCCCCCC", "similarity_to_target": 1.0}
    decision = handle_tool_result(
        RolloutPolicyConfig(stop_condition="exact", max_tool_calls=1),
        RolloutPolicyState(),
        name="submit_pathway",
        arguments={},
        result=submission,
        target_smiles="CCCCCCC",
    )
    assert decision.terminal_due_to_tool_limit
    assert decision.terminal_submission == submission


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("exact", "exact"),
        (" EXACT ", "exact"),
        ("model", "model"),
        (" Model ", "model"),
        (0, 0.0),
        ("0.8", 0.8),
        (1, 1.0),
    ],
)
def test_parse_stop_condition_accepts_numbers_exact_and_model(value, expected):
    assert parse_stop_condition(value) == expected


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -0.1, 1.1, "excat", None, [0.5], True]
)
def test_parse_stop_condition_rejects_out_of_range_and_unknown_values(value):
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_stop_condition(value)


def test_invalid_floor_is_a_configuration_error():
    with pytest.raises(ValueError):
        RolloutPolicyConfig(stop_condition="excat")
