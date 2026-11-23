# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from synclaw.rollout.policy import (
    NON_TOOL_PERSISTENCE_GUARANTEE_SENTENCE,
    RolloutPolicyConfig,
    RolloutPolicyState,
    handle_no_tool_response,
    handle_tool_result,
    prepare_tool_call,
    submission_summary_fields,
    task_budget_config,
    turn_reminders,
)
from synclaw.tools.contracts import TASK_BUDGET_FIELDS


def _config(**overrides: object) -> RolloutPolicyConfig:
    values = {
        "max_turns": 10,
        "max_tool_calls": 20,
        "stop_condition": 1.0,
        "turn_warning_remaining": 3,
        "force_submit_after_non_tool_turns": 2,
        "same_pathway_finish_after": 2,
        "max_low_similarity_submissions": None,
    }
    values.update(overrides)
    return RolloutPolicyConfig(**values)  # type: ignore[arg-type]


def _accepted_submission(similarity: float) -> dict[str, object]:
    return {
        "accepted": True,
        "constitutional_similarity_to_target": similarity,
        "similarity_to_target": similarity,
    }


def test_policy_checkpoint_fields_round_trip() -> None:
    state = RolloutPolicyState.from_checkpoint(
        {
            "total_tool_calls": 7,
            "consecutive_non_tool_turns_after_tools": 2,
            "forced_submit_requested": True,
            "last_submission": {"accepted": False},
            "best_accepted_similarity": 0.75,
        }
    )

    fields = state.checkpoint_fields()

    assert fields["total_tool_calls"] == 7
    assert fields["consecutive_non_tool_turns_after_tools"] == 2
    assert fields["forced_submit_requested"] is True
    assert fields["last_submission"] == {"accepted": False}
    assert state.has_submission is True
    assert fields["best_accepted_similarity"] == 0.75


def test_turn_reminders_are_emitted_once_and_force_the_final_turn() -> None:
    state = RolloutPolicyState(best_accepted_submission={"accepted": True})
    config = _config()
    no_submission_warning = turn_reminders(
        config,
        RolloutPolicyState(),
        turn_index=7,
    )

    warning = turn_reminders(config, state, turn_index=7)
    duplicate = turn_reminders(config, state, turn_index=7)
    final = turn_reminders(config, state, turn_index=9)

    assert len(warning) == 1
    assert "<FINISH>" in warning[0]
    assert len(no_submission_warning) == 1
    assert "call submit_pathway for it now so it can be scored" in no_submission_warning[0]
    assert "verify a clearly better candidate" in no_submission_warning[0]
    assert "<FINISH>" not in no_submission_warning[0]
    assert duplicate == []
    assert final == [
        "This is the final allowed turn. Call submit_pathway now with your best route."
    ]
    assert state.forced_submit_requested is True


@pytest.mark.parametrize("tool_calls", [0, 1])
def test_no_tool_response_uses_shared_persistence_and_force_submit_policy(tool_calls: int) -> None:
    state = RolloutPolicyState(
        total_tool_calls=1,
        best_accepted_similarity=0.5,
    )
    config = _config(force_submit_after_non_tool_turns=2)

    first = handle_no_tool_response(config, state, content="still thinking")
    second = handle_no_tool_response(config, state, content="still thinking")

    assert first.reminder is not None
    assert NON_TOOL_PERSISTENCE_GUARANTEE_SENTENCE in first.reminder
    assert second.reminder is not None
    assert second.reminder.startswith("Call submit_pathway now")
    assert state.forced_submit_requested is True

    search_only_state = RolloutPolicyState(total_tool_calls=tool_calls)
    search_only_first = handle_no_tool_response(config, search_only_state, content="still thinking")
    search_only_second = handle_no_tool_response(
        config, search_only_state, content="still thinking"
    )
    assert search_only_first.reminder is not None
    assert search_only_first.reminder.startswith("Continue. Use additional tools")
    assert search_only_second.reminder is not None
    assert search_only_second.reminder.startswith("Call submit_pathway now")
    assert "maximum achievable route" not in search_only_first.reminder
    assert "maximum achievable route" not in search_only_second.reminder


def test_prepare_tool_call_normalizes_payloads_independent_of_event_index() -> None:
    arguments = (
        '{"target_smiles":"hidden","final_smiles":"ignored","steps":"['
        '{\\"reaction_id\\":\\"R1\\",\\"reactants\\":\\"[\\\\\\"CC\\\\\\"]\\"}'
        ']"}'
    )
    endpoint = prepare_tool_call(
        raw_name="submit_pathway<|end|>",
        raw_arguments=arguments,
        raw_tool_call={"id": "call-1", "type": "function"},
        event_index=1,
    )
    later = prepare_tool_call(
        raw_name="submit_pathway<|end|>",
        raw_arguments=arguments,
        raw_tool_call={"id": "call-1", "type": "function"},
        event_index=99,
    )

    assert endpoint.name == later.name == "submit_pathway"
    assert endpoint.dispatch_arguments == later.dispatch_arguments
    assert endpoint.dispatch_arguments == {"steps": [{"reaction_id": "R1", "reactants": ["CC"]}]}
    assert endpoint.event_payload["ignored_target_smiles"] == "hidden"
    assert endpoint.event_payload["ignored_final_smiles"] == "ignored"


def test_repair_reminders_name_no_optional_tool() -> None:
    optional = ("search_building_blocks", "search_reaction_templates", "simulate_", "mces")
    invalid_state = RolloutPolicyState()
    invalid = handle_tool_result(
        _config(),
        invalid_state,
        name="submit_pathway",
        arguments={"steps": []},
        result={"accepted": False, "error_message": "no grounded building block"},
        target_smiles="CC",
    )
    low = handle_tool_result(
        _config(),
        RolloutPolicyState(),
        name="submit_pathway",
        arguments={"steps": [], "building_block": "BB:1"},
        result=_accepted_submission(0.5),
        target_smiles="CC",
    )

    assert invalid.repair is not None and low.repair is not None
    for repair in (invalid.repair, low.repair):
        assert not any(name in repair for name in optional)
    assert (
        "Consider near building-block analogues. Compare the similarity of candidate products to the "
        "target, and verify "
    ) in low.repair
    assert invalid_state.invalid_submissions == 1
    assert invalid_state.has_submission is True


def test_low_similarity_sequence_tracks_best_route_and_allows_finish_token() -> None:
    state = RolloutPolicyState()
    config = _config()
    first_arguments = {"steps": [], "building_block": "BB:1"}
    second_arguments = {"steps": [], "building_block": "BB:2"}

    first = handle_tool_result(
        config,
        state,
        name="submit_pathway",
        arguments=first_arguments,
        result=_accepted_submission(0.4),
        target_smiles="CC",
    )
    second = handle_tool_result(
        config,
        state,
        name="submit_pathway",
        arguments=second_arguments,
        result=_accepted_submission(0.8),
        target_smiles="CC",
    )
    finish = handle_no_tool_response(config, state, content="<FINISH>")

    assert first.repair is not None
    assert second.repair is not None
    assert state.best_accepted_similarity == 0.8
    assert state.best_accepted_submission_arguments == second_arguments
    assert finish.finish_reason == "finish_token_best_accepted_submission"


def test_repeated_low_similarity_pathway_eventually_allows_finish_token() -> None:
    state = RolloutPolicyState()
    config = _config(same_pathway_finish_after=3)
    arguments = {"steps": [], "building_block": "BB:1"}

    repairs = [
        handle_tool_result(
            config,
            state,
            name="submit_pathway",
            arguments=arguments,
            result=_accepted_submission(0.4),
            target_smiles="CC",
        ).repair
        for _ in range(3)
    ]

    assert all(repair is not None for repair in repairs)
    assert all("<FINISH>" not in repair for repair in repairs[:2] if repair is not None)
    assert repairs[2] is not None and "<FINISH>" in repairs[2]
    assert state.repeated_submission_count == 3


def test_tool_limit_accepts_the_best_low_similarity_submission() -> None:
    state = RolloutPolicyState()
    config = _config(max_tool_calls=1)
    arguments = {"steps": [], "building_block": "BB:1"}

    decision = handle_tool_result(
        config,
        state,
        name="submit_pathway",
        arguments=arguments,
        result=_accepted_submission(0.4),
        target_smiles="CC",
    )

    assert decision.terminal_due_to_tool_limit is True
    assert decision.terminal_submission == _accepted_submission(0.4)


def test_tool_limit_rejects_invalid_submission_without_selecting_it() -> None:
    state = RolloutPolicyState()
    decision = handle_tool_result(
        _config(max_tool_calls=1),
        state,
        name="submit_pathway",
        arguments={"steps": []},
        result={
            "accepted": False,
            "actual_final_smiles": "CC",
            "constitutional_similarity_to_target": 1.0,
        },
        target_smiles="CC",
    )

    assert decision.terminal_submission is None
    assert decision.terminal_due_to_tool_limit is True
    assert state.tool_limit_finalization_requested is True
    assert state.best_accepted_submission is None
    assert state.selected_submission()[0] is not None


def test_summary_fields_keep_a_rejected_submission_unscored() -> None:
    rejected = {
        "accepted": False,
        "final_smiles": "CCO",
        "constitutional_similarity_to_target": 1.0,
        "reaction_validation": "invalid",
        "building_block_grounded": False,
    }

    fields = submission_summary_fields(rejected, "CCO")
    absent = submission_summary_fields(None, "CCO")

    assert fields["submitted"] is True
    assert absent["submitted"] is False
    assert fields["accepted"] is False
    assert fields["similarity_to_target"] is None
    assert fields["final_smiles"] is None
    # Rejection diagnostics stay available for the run report.
    assert fields["reaction_validation"] == "invalid"
    assert fields["building_block_grounded"] is False


def test_tool_limit_requests_one_final_submission_after_other_tools() -> None:
    state = RolloutPolicyState()
    config = _config(max_tool_calls=1)

    handle_tool_result(
        config,
        state,
        name="search_building_blocks",
        arguments={"query": "CC"},
        result={"valid": True, "results": []},
        target_smiles="CC",
    )
    reminders = turn_reminders(config, state, turn_index=1)
    final = handle_tool_result(
        config,
        state,
        name="submit_pathway",
        arguments={"steps": [], "building_block": "BB:1"},
        result=_accepted_submission(0.4),
        target_smiles="CC",
    )

    assert len(reminders) == 1
    assert "tool-call budget has been reached" in reminders[0]
    assert state.forced_submit_requested is True
    assert state.tool_limit_finalization_requested is True
    assert final.terminal_due_to_tool_limit is True


@pytest.mark.parametrize(
    "warning, critical",
    [(0.5, 0.6), (-0.1, 0.25), (0.5, 1.1)],
)
def test_context_remaining_fractions_reject_out_of_range_or_inverted_values(warning, critical):
    with pytest.raises(ValueError):
        _config(
            context_warning_remaining_fraction=warning,
            context_critical_remaining_fraction=critical,
        )


def test_task_budgets_override_the_runner_and_keep_its_unset_values() -> None:
    runner = RolloutPolicyConfig(max_model_len=262144, max_tokens=32768)

    assert task_budget_config({}, runner) == runner
    assert task_budget_config({"max_turns": None}, runner) == runner
    resolved = task_budget_config(
        {
            "max_turns": 40,
            "max_tool_calls": 75,
            "max_model_len": 131072,
            "max_tokens": 16384,
            "budget_feedback": False,
        },
        runner,
    )
    assert resolved.budget_fields == {
        "max_turns": 40,
        "max_tool_calls": 75,
        "max_model_len": 131072,
        "max_tokens": 16384,
        "context_limit_stated": True,
        "context_safety_margin_fraction": 1 / 32,
        "budget_feedback": False,
        "stop_condition": "exact",
    }
    # Turn, tool-call and output budgets may exceed the runner's; nothing caps them at the server.
    assert task_budget_config({"max_turns": 200, "max_tokens": 65536}, runner).max_turns == 200


@pytest.mark.parametrize("value", [True, "40", 40.0, 0, -1])
def test_task_budgets_must_be_positive_integers(value: object) -> None:
    with pytest.raises(ValueError, match="max_turns must be a positive integer"):
        task_budget_config({"max_turns": value}, RolloutPolicyConfig())


def test_task_context_may_lower_but_not_raise_the_runner_limit() -> None:
    runner = RolloutPolicyConfig(max_model_len=131072)

    assert task_budget_config({"max_model_len": 65536}, runner).max_model_len == 65536


@pytest.mark.parametrize(
    ("task", "runner_context", "match"),
    [
        ({"max_model_len": 262144}, 131072, "exceeds the runner's context limit 131072"),
        (
            {"max_model_len": 512},
            131072,
            "leaves no room for output",
        ),  # Nothing left after the margin.
        ({"budget_feedback": "false"}, 131072, "must be true or false"),
        # Unlimited turns or output need a context limit to end or clip the rollout.
        ({"max_tokens": "inf"}, None, "needs a known context limit"),
        ({"max_turns": "inf"}, None, "Unlimited turns need a known context limit"),
    ],
)
def test_invalid_task_budgets_are_refused(
    task: dict, runner_context: int | None, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        task_budget_config(task, RolloutPolicyConfig(max_model_len=runner_context))


def test_turn_warning_is_progress_feedback_but_final_turn_notices_are_not() -> None:
    config = _config(budget_feedback=False)
    state = RolloutPolicyState()

    assert turn_reminders(config, state, turn_index=7) == []
    final = turn_reminders(config, state, turn_index=9)
    assert final and "final allowed turn" in final[0]


def _accepted_analogue(
    state: RolloutPolicyState, config: RolloutPolicyConfig, similarity: float = 0.6
):
    return handle_tool_result(
        config,
        state,
        name="submit_pathway",
        arguments={"steps": [], "building_block": "BB:1"},
        result={**_accepted_submission(similarity), "actual_final_smiles": "CCN"},
        target_smiles="CCO",
    )


def test_when_the_model_stops_an_analogue_gets_only_the_validator_result() -> None:
    nudged = _accepted_analogue(RolloutPolicyState(), _config(stop_condition="exact"))
    assert "guaranteed" in nudged.repair

    config = _config(stop_condition="model", same_pathway_finish_after=1)
    state = RolloutPolicyState()
    quiet = _accepted_analogue(state, config)
    assert quiet.repair is None and quiet.terminal_submission is None
    assert "<FINISH>" not in turn_reminders(config, state, turn_index=7)[0]
    decision = handle_no_tool_response(config, state, content="This is my best route.")
    assert decision.finish_reason == "model_stop_best_accepted_submission"


def test_when_the_model_stops_an_exact_match_leaves_the_rollout_open_for_its_answer() -> None:
    config = _config(stop_condition="model")
    state = RolloutPolicyState()
    decision = handle_tool_result(
        config,
        state,
        name="submit_pathway",
        arguments={"steps": [], "building_block": "BB:1"},
        result={**_accepted_submission(1.0), "actual_final_smiles": "CCO"},
        target_smiles="CCO",
    )

    assert decision.terminal_submission is None and decision.repair is None
    assert state.best_accepted_identical and state.low_similarity_submissions == 0
    answer = handle_no_tool_response(
        config, state, content="The route above makes the exact target."
    )
    assert answer.finish_reason == "model_stop_best_accepted_submission"


@pytest.mark.parametrize("tool_calls", [3, 0])  # Including a first reply with no tool call.
def test_when_the_model_stops_it_is_asked_once_for_a_submission_when_none_was_accepted(
    tool_calls: int,
) -> None:
    config = _config(stop_condition="model")
    state = RolloutPolicyState(total_tool_calls=tool_calls)

    first = handle_no_tool_response(config, state, content="I cannot find a route.")
    assert (
        first.reminder.startswith("Call submit_pathway now") and "guaranteed" not in first.reminder
    )
    assert state.forced_submit_requested
    second = handle_no_tool_response(config, state, content="Still no route.")
    assert second.finish_reason == "model_stop_no_accepted_submission"


def test_when_the_model_stops_a_rejected_submission_does_not_earn_a_second_request() -> None:
    config = _config(stop_condition="model")
    state = RolloutPolicyState(total_tool_calls=3)

    assert handle_no_tool_response(config, state, content="I cannot find a route.").reminder
    handle_tool_result(
        config,
        state,
        name="submit_pathway",
        arguments={"steps": []},
        result={"accepted": False, "error_message": "Ungrounded starting reactant."},
        target_smiles="CCO",
    )
    decision = handle_no_tool_response(config, state, content="Still no route.")

    assert decision.finish_reason == "model_stop_no_accepted_submission"
    restored = RolloutPolicyState.from_checkpoint(state.checkpoint_fields())
    assert restored.model_stop_submit_requested


def test_inf_budgets_are_unlimited_and_the_context_stays_enforced_unstated() -> None:
    runner = RolloutPolicyConfig(max_model_len=131072)
    unlimited = dict.fromkeys(TASK_BUDGET_FIELDS, "inf")

    config = task_budget_config(unlimited, runner)

    assert (config.max_turns, config.max_tool_calls, config.max_tokens) == (None, 0, None)
    assert config.max_model_len == 131072 and config.stated_max_model_len is None
    state = RolloutPolicyState()
    assert all(turn_reminders(config, state, turn_index=index) == [] for index in (0, 200))


@pytest.mark.parametrize("stop_condition", ["exact", "model"])
def test_when_the_model_stops_failed_calls_get_only_their_own_errors(stop_condition: str) -> None:
    config = _config(stop_condition=stop_condition)
    rejected = handle_tool_result(
        config,
        RolloutPolicyState(),
        name="submit_pathway",
        arguments={"steps": []},
        result={
            "accepted": False,
            "error_message": "Ungrounded starting reactant.",
            "issues": ["x"],
        },
        target_smiles="CCO",
    )
    failed = handle_tool_result(
        config,
        RolloutPolicyState(),
        name="search_building_blocks",
        arguments={},
        result={"error": "invalid arguments", "retryable": True, "issues": ["query is missing"]},
        target_smiles="CCO",
    )

    coached = stop_condition == "exact"
    assert (
        rejected.repair is not None and "Repair it using the tools" in rejected.repair
    ) == coached
    assert (failed.repair is not None and "Correct the arguments" in failed.repair) == coached


def test_out_of_range_margins_are_refused() -> None:
    for fraction in (-0.1, 1.0):
        with pytest.raises(ValueError, match="at least 0 and below 1"):
            RolloutPolicyConfig(context_safety_margin_fraction=fraction)
    with pytest.raises(ValueError, match="leaves no room for output"):
        RolloutPolicyConfig(max_model_len=512)


@pytest.mark.parametrize(
    ("context", "fraction", "margin"),
    [
        (262144, 1 / 32, 8192),  # The default context keeps the old fixed margin.
        (131072, 1 / 32, 4096),
        (8192, 1 / 32, 512),  # Floored, rather than 256.
        (262144, 0.0, 0),
        (None, 1 / 32, 0),  # No context limit, so nothing to clip against.
    ],
)
def test_the_safety_margin_scales_with_the_context_limit(context, fraction, margin) -> None:
    config = RolloutPolicyConfig(max_model_len=context, context_safety_margin_fraction=fraction)
    assert config.context_safety_margin_tokens == margin


def test_a_stop_at_the_tool_call_limit_is_reported_as_the_tool_limit() -> None:
    config = _config(stop_condition="model", max_tool_calls=1)
    state = RolloutPolicyState()
    _accepted_analogue(state, config)
    turn_reminders(config, state, turn_index=1)

    decision = handle_no_tool_response(config, state, content="That is my best route.")

    assert decision.finish_reason == "tool_limit_best_accepted_submission"


def test_a_zero_low_similarity_cap_is_rejected_not_clamped() -> None:
    with pytest.raises(ValueError, match="max_low_similarity_submissions must be at least 1"):
        RolloutPolicyConfig(max_low_similarity_submissions=0)


def test_an_equally_unscored_later_submission_does_not_replace_the_best() -> None:
    from synclaw.rollout.policy import is_better_accepted_submission

    assert not is_better_accepted_submission(False, None, False, None)
    assert is_better_accepted_submission(False, 0.4, False, None)
    assert not is_better_accepted_submission(False, None, False, 0.4)
