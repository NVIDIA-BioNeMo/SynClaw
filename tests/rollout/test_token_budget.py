# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import math
from dataclasses import replace

import pytest

from synclaw.rollout.budget import (
    FEEDBACK_ALLOWANCE_TOKENS,
    BudgetFeedbackState,
    ContextBudgetSnapshot,
    ContextTokenEstimator,
    budget_summary_fields,
    initialize_budget_feedback,
    prepare_budget_feedback,
    serialized_input_bytes,
)
from synclaw.rollout.loop import (
    AssistantTurn,
    ControlledTermination,
    RolloutState,
    RolloutTask,
    ToolExecution,
    request_output_budget,
    run_rollout,
)
from synclaw.rollout.policy import RolloutPolicyConfig
from synclaw.rollout.prompts import build_initial_messages


def provider(messages, tools):
    return ContextBudgetSnapshot(
        50000,
        serialized_input_bytes({"messages": messages, "tools": tools}),
        1000,
        "fixture_counter",
        estimated=False,
    )


def with_allowance(snapshot):
    """The reported assessment is the raw measurement plus the reminder allowance."""
    return replace(snapshot, input_tokens=snapshot.input_tokens + FEEDBACK_ALLOWANCE_TOKENS)


@pytest.mark.parametrize("context", [50000, None])
def test_initial_limits_are_in_system_prompt_and_not_repeated_in_feedback(context):
    # A finite turn budget, so the config is valid with no context limit too.
    config = RolloutPolicyConfig(
        max_tool_calls=150, max_turns=40, max_model_len=context, max_tokens=1000
    )
    messages = build_initial_messages(
        task={"system_prompt": "Plan syntheses.", "question": "Plan CCO."},
        config=config,
    )
    state = initialize_budget_feedback(config)
    assert len(messages) == 2
    assert "Use no more than 150 total tool calls or 40 assistant turns. " in messages[0]["content"]
    assert "Each response is capped at 1000" in messages[0]["content"]
    assert ("context-window budget is 50000" in messages[0]["content"]) == (context is not None)
    assert "context tokens" not in messages[0]["content"]
    assert messages[1]["content"] == "Plan CCO."

    def configured(messages, tools):
        return replace(provider(messages, tools), context_window_tokens=context)

    content, snapshot, status = prepare_budget_feedback(
        configured,
        messages,
        [],
        state,
        turn_index=0,
    )
    if context is None:
        assert content is None and status == "unavailable_or_unchanged"
    else:
        assert status == "emitted" and "context tokens." in content
        assert "capped" not in content and "budget is" not in content
        assert snapshot == with_allowance(configured(messages, []))


def test_feedback_reserves_an_allowance_and_announces_two_distinct_limits():
    messages = [{"role": "user", "content": "Plan a synthesis."}]
    state = BudgetFeedbackState()
    content, snapshot, status = prepare_budget_feedback(provider, messages, [], state, turn_index=0)
    assert status == "emitted"
    assert "context-window budget is 50000" in content
    assert "Each response is capped at 1000 output tokens" in content
    assert "output tokens, including reasoning. This cap resets" in content
    assert "This cap resets for each response" in content
    assert "~" not in content
    assert snapshot.input_tokens == provider(messages, []).input_tokens + FEEDBACK_ALLOWANCE_TOKENS
    assert f"Remaining: {snapshot.remaining_tokens} context tokens." in content


def test_combined_feedback_counts_turn_and_tool_text_in_its_token_estimate():
    def estimated(messages, tools):
        return replace(provider(messages, tools), estimated=True)

    state = BudgetFeedbackState(50000, 1000)
    messages = [{"role": "user", "content": "plan"}]
    content, snapshot, status = prepare_budget_feedback(
        estimated,
        messages,
        [],
        state,
        turn_index=1,
        turns_remaining=79,
        tool_calls_remaining=145,
    )
    assert status == "emitted"
    assert content.startswith(
        f"Remaining: 79 turns, 145 tool calls, ~{snapshot.remaining_tokens // 1000}k context tokens."
    )
    assert snapshot == with_allowance(estimated(messages, []))


@pytest.mark.parametrize("estimated", [True, False])
@pytest.mark.parametrize("input_tokens", [10000, None])
def test_context_assessment_does_not_change_requested_output_cap(estimated, input_tokens):
    snapshot = ContextBudgetSnapshot(262144, input_tokens, 32768, estimated=estimated)
    assert snapshot.trace_fields()["request_output_tokens"] == 32768
    assert "output_safety_margin_tokens" not in snapshot.trace_fields()


def test_feedback_warns_near_context_limit_without_reducing_output_cap():
    messages = [{"role": "tool", "tool_call_id": "a", "content": '{"result":"CCO"}'}]

    def near_limit(messages, tools):
        return ContextBudgetSnapshot(
            262144,
            231000 + serialized_input_bytes(messages),
            32768,
            estimated=True,
        )

    text, snapshot, status = prepare_budget_feedback(
        near_limit,
        messages,
        [],
        BudgetFeedbackState(262144, 32768),
        turn_index=5,
    )
    assert status == "emitted"
    assert "This response is limited to" not in text
    assert snapshot == with_allowance(near_limit(messages, []))
    assert snapshot.remaining_tokens < 32768
    assert snapshot.trace_fields()["request_output_tokens"] == 32768
    assert "You must call submit_pathway now with your best verified route" in text
    assert json.loads(messages[-1]["content"]) == {"result": "CCO"}  # Preview is pure.


@pytest.mark.parametrize(
    "remaining,warning_sent,expected",
    [
        (5001, False, None),
        (5000, False, "Context is running low."),
        (4999, False, "Context is running low."),
        (5000, True, None),
        (2501, True, None),
        (2500, False, "Your context window is nearly exhausted."),
        (2500, True, "Your context window is nearly exhausted."),
        (2499, True, "Your context window is nearly exhausted."),
        (None, False, None),
    ],
)
def test_context_submission_warning_thresholds_do_not_consume_state(
    remaining, warning_sent, expected
):
    def assessed(messages, tools):
        occupied = None if remaining is None else 10000 - remaining - FEEDBACK_ALLOWANCE_TOKENS
        return ContextBudgetSnapshot(10000, occupied, 1000, estimated=False)

    state = BudgetFeedbackState(10000, 1000, context_warning_sent=warning_sent)
    original = state.checkpoint_fields()
    content, _, status = prepare_budget_feedback(
        assessed,
        [{"role": "user", "content": "plan"}],
        [],
        state,
        turn_index=0,
        turns_remaining=80,
    )
    assert status == "emitted"
    assert ("submit_pathway" in content) == (expected is not None)
    if expected is not None:
        assert expected in content
        assert "best verified route, even if it is an analogue, so it can be scored" in content
    assert state.checkpoint_fields() == original  # Only committed feedback consumes the warning.


def test_feedback_never_creates_a_user_turn_after_an_assistant():
    messages = [{"role": "assistant", "content": "done", "reasoning_content": "retained"}]
    text, _, status = prepare_budget_feedback(
        provider, messages, [], BudgetFeedbackState(), turn_index=1
    )
    assert text is None and status == "no_pending_input"
    assert messages[0]["reasoning_content"] == "retained"


@pytest.mark.parametrize(
    "tool_calls,expected",
    [
        (None, "Remaining: 1 turn."),
        (0, "Remaining: 1 turn, 0 tool calls."),
        (1, "Remaining: 1 turn, 1 tool call."),
    ],
)
def test_known_rollout_counts_remain_available_without_context_usage(tool_calls, expected):
    def unknown(messages, tools):
        return ContextBudgetSnapshot(context_window_tokens=50000, max_output_tokens=1000)

    content, snapshot, status = prepare_budget_feedback(
        unknown,
        [],
        [],
        BudgetFeedbackState(50000, 1000),
        turn_index=3,
        turns_remaining=1,
        tool_calls_remaining=tool_calls,
    )
    assert status == "emitted" and content == expected
    assert snapshot.remaining_tokens is None


def test_repeated_limits_are_omitted_but_changed_output_cap_is_announced():
    state = BudgetFeedbackState(50000, 1000, 0)
    content, _, _ = prepare_budget_feedback(provider, [], [], state, turn_index=1)
    assert "context tokens" in content
    assert "capped" not in content
    assert "budget is" not in content

    def changed(messages, tools):
        return replace(provider(messages, tools), max_output_tokens=2000)

    content, _, _ = prepare_budget_feedback(changed, [], [], state, turn_index=2)
    assert "capped at 2000" in content


def test_unknown_context_announces_only_output_cap_once():
    def unknown(messages, tools):
        return ContextBudgetSnapshot(max_output_tokens=512)

    state = BudgetFeedbackState()
    content, _, _ = prepare_budget_feedback(unknown, [], [], state, turn_index=0)
    assert "512 output tokens" in content
    assert "context tokens" not in content
    assert "context-window budget" not in content
    state.announced_output_limit = 512
    content, _, status = prepare_budget_feedback(unknown, [], [], state, turn_index=1)
    assert content is None
    assert status == "unavailable_or_unchanged"


def test_feedback_still_warns_when_assessment_reports_exhausted_context():
    def tiny(messages, tools):
        return replace(provider(messages, tools), context_window_tokens=50)

    content, snapshot, status = prepare_budget_feedback(
        tiny, [], [], BudgetFeedbackState(), turn_index=0
    )
    assert "Remaining: 0 context tokens." in content
    assert "You must call submit_pathway now" in content
    assert snapshot == with_allowance(tiny([], []))
    assert snapshot.remaining_tokens == 0
    assert status == "emitted"


def test_calibration_rebases_full_replay_instead_of_summing_billed_history():
    estimator = ContextTokenEstimator()
    payload = {"messages": [{"role": "user", "content": "C" * 300}], "tools": []}
    first = estimator.snapshot(payload, context_window_tokens=10000, max_output_tokens=1000)
    assert first.input_tokens == math.ceil(serialized_input_bytes(payload) / 3)
    assert first.estimated and first.source == "utf8_estimate"
    for _ in range(3):
        estimator.observe(payload, {"prompt_tokens": 600, "completion_tokens": 200})
    current = estimator.snapshot(payload, context_window_tokens=10000, max_output_tokens=1000)
    assert current.input_tokens == 600
    assert current.remaining_tokens == 9400
    assert current.source == "provider_calibrated_utf8"
    compacted = estimator.snapshot(
        {"messages": []}, context_window_tokens=10000, max_output_tokens=1000
    )
    assert compacted.input_tokens < current.input_tokens


def test_cache_and_reasoning_usage_are_not_double_counted():
    estimator = ContextTokenEstimator()
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 200,
        "cache_read_input_tokens": 300,
        "output_tokens_details": {"reasoning_tokens": 40},
    }
    estimator.observe({"messages": []}, usage, separate_cache_usage=True)
    assert estimator.continuation_tokens == 650
    estimator.observe(
        {},
        {
            "prompt_tokens": 600,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 500},
            "completion_tokens_details": {"reasoning_tokens": 40},
        },
    )
    assert estimator.continuation_tokens == 650


def test_server_history_uses_usage_anchor_plus_only_new_input():
    estimator = ContextTokenEstimator()
    initial = {"input": [{"role": "user", "content": "hello"}], "tools": []}
    estimator.observe(initial, {"input_tokens": 1000, "output_tokens": 250})
    pending = {"input": [{"type": "function_call_output", "output": "verified"}]}
    snapshot = estimator.snapshot(
        pending, context_window_tokens=10000, max_output_tokens=1000, server_managed=True
    )
    assert snapshot.input_tokens == 1250 + math.ceil(
        serialized_input_bytes(pending) * estimator.tokens_per_byte
    )
    rate = estimator.tokens_per_byte
    estimator.observe(pending, {"input_tokens": 1500, "output_tokens": 100}, server_managed=True)
    assert estimator.tokens_per_byte == rate  # tiny continuation is not the full input
    assert estimator.continuation_tokens == 1600  # rebase, never accumulate
    estimator.observe(pending, {}, server_managed=True)
    assert (
        estimator.snapshot(
            pending, context_window_tokens=10000, max_output_tokens=1000, server_managed=True
        ).input_tokens
        is None
    )


@pytest.mark.parametrize(
    "item",
    [
        {"type": "reasoning", "encrypted_content": "opaque"},
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "item_reference", "id": "rs_1"},
        {"type": "redacted_thinking", "data": "opaque"},
    ],
)
def test_opaque_state_does_not_become_a_misleading_byte_count(item):
    estimator = ContextTokenEstimator()
    snapshot = estimator.snapshot(
        {"input": [item]},
        context_window_tokens=10000,
        max_output_tokens=1000,
    )
    assert snapshot.remaining_tokens is None
    assert snapshot.source == "unavailable"
    # JSON Schema union types are ordinary input, not opaque content.
    assert (
        estimator.snapshot(
            {"tools": [{"type": ["string", "null"]}]},
            context_window_tokens=10000,
            max_output_tokens=1000,
        ).remaining_tokens
        is not None
    )


def test_estimator_and_feedback_state_survive_json_checkpoint():
    estimator = ContextTokenEstimator()
    estimator.observe({"input": "hello"}, {"input_tokens": 100, "output_tokens": 20})
    assert ContextTokenEstimator.from_checkpoint(estimator.checkpoint_fields()) == estimator
    state = BudgetFeedbackState.from_checkpoint(
        BudgetFeedbackState(1000, 300, 2).checkpoint_fields()
    )
    content, snapshot, status = prepare_budget_feedback(provider, [], [], state, turn_index=2)
    assert (content, snapshot, status) == (None, None, "already_reported")


@pytest.mark.parametrize("ciphertext_size", [10, 100_000])
def test_opaque_replay_uses_latest_usage_and_only_new_readable_input(ciphertext_size):
    estimator = ContextTokenEstimator()
    request = {"input": [{"role": "user", "content": "plan"}], "tools": []}
    output = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "x" * ciphertext_size}]
    estimator.observe(
        request,
        {
            "input_tokens": 16_725,
            "output_tokens": 349,
            "input_tokens_details": {"cached_tokens": 7_857},
            "output_tokens_details": {"reasoning_tokens": 99},
        },
        replay_output=output,
    )
    retained = {**request, "input": [*request["input"], *output]}

    def snapshot(payload):
        return estimator.snapshot(payload, context_window_tokens=100_000, max_output_tokens=1000)

    # Cached tokens and reasoning are already included; ciphertext adds nothing.
    assert snapshot(retained).input_tokens == 17_074
    assert snapshot(retained).source == "provider_usage_plus_replay_delta"
    assert snapshot(retained).estimated
    following = {
        **retained,
        "input": [
            *retained["input"],
            {
                "type": "function_call_output",
                "call_id": "call1",
                "output": "result",
            },
        ],
    }
    assert snapshot(following).input_tokens > 17_074
    rate = estimator.tokens_per_byte
    estimator.observe(following, {"input_tokens": 31_519, "output_tokens": 415}, replay_output=[])
    assert estimator.tokens_per_byte == rate  # Opaque bytes never calibrate the ratio.
    assert snapshot(following).input_tokens == 31_934  # Rebase, don't sum prior usage.


@pytest.mark.parametrize("change", ["history", "tools", "truncate", "new_opaque", "missing_usage"])
def test_replay_usage_anchor_rejects_uncovered_or_changed_input(change):
    estimator = ContextTokenEstimator()
    request = {"input": [{"role": "user", "content": "plan"}], "tools": []}
    output = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}]
    usage = {"input_tokens": 1000, "output_tokens": 100}
    estimator.observe(request, usage, replay_output=output)
    following = copy.deepcopy({**request, "input": [*request["input"], *output]})
    if change == "history":
        following["input"][0]["content"] = "different prompt"
    elif change == "tools":
        following["tools"] = [{"name": "new_tool"}]
    elif change == "truncate":
        following["input"].pop(0)
    elif change == "new_opaque":
        following["input"].append({"type": "input_image", "image_url": "unmeasured"})
    else:
        estimator.observe(following, {}, replay_output=[])
    snapshot = estimator.snapshot(following, context_window_tokens=10000, max_output_tokens=1000)
    assert snapshot.input_tokens is None
    assert snapshot.source == "unavailable"


def test_replay_anchor_survives_sorted_json_checkpoint():
    estimator = ContextTokenEstimator()
    request = {"input": [{"role": "user", "content": "plan"}], "tools": []}
    output = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}]
    estimator.observe(request, {"input_tokens": 1000, "output_tokens": 100}, replay_output=output)
    checkpoint = json.loads(
        json.dumps(
            {
                "estimator": estimator.checkpoint_fields(),
                "payload": {**request, "input": [*request["input"], *output]},
            },
            sort_keys=True,
        )
    )
    restored = ContextTokenEstimator.from_checkpoint(checkpoint["estimator"])
    assert (
        restored.snapshot(
            checkpoint["payload"],
            context_window_tokens=10000,
            max_output_tokens=1000,
        ).input_tokens
        == 1100
    )


class _Transport:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.requests = []

    async def complete(self, messages, *, tools, tool_choice, max_output_tokens=None):
        self.requests.append((list(messages), tool_choice))
        turn = next(self.turns)
        if isinstance(turn, Exception):
            raise turn
        return turn

    def append_user(self, message):
        pass

    def append_tool(self, call, result_json):
        pass

    def replace_last_input(self, message):
        pass


class _Tools:
    async def execute(self, calls, **kwargs):
        return [
            ToolExecution({"results": [{"id": 1, "canonical_smiles": "C" * 300}]}) for _ in calls
        ]


def _run(
    transport,
    *,
    state=None,
    start=0,
    enabled=True,
    budget_provider=provider,
    max_tool_calls=150,
    max_turns=4,
):
    return asyncio.run(
        run_rollout(
            task=RolloutTask("fixture", "CCO", ({"role": "user", "content": "plan"},)),
            # No safety margin, so near-full contexts still send requests and reach the warnings.
            config=RolloutPolicyConfig(
                max_turns=max_turns,
                turn_warning_remaining=3,
                budget_feedback=enabled,
                max_tool_calls=max_tool_calls,
                context_safety_margin_fraction=0,
            ),
            transport=transport,
            tool_executor=_Tools(),
            tool_schemas=[],
            state=state,
            start_turn_index=start,
            budget_provider=budget_provider,
        )
    )


@pytest.mark.parametrize(
    "remaining",
    [
        [5001, 5000, 4000, 2501, 2500, 2000, 3000],
        [2000, 3000, 2000],  # Jump straight to the stronger warning, then recalibrate upward.
    ],
)
def test_context_warnings_repeat_only_below_quarter_and_preserve_reasoning(remaining):
    def assessed(messages, tools):
        turn = sum(m["role"] == "assistant" for m in messages)
        occupied = 10000 - remaining[turn] - FEEDBACK_ALLOWANCE_TOKENS
        return ContextBudgetSnapshot(10000, occupied, 1000, estimated=False)

    transport = _Transport(
        [
            *[
                AssistantTurn(
                    {
                        "reasoning_content": "retained reasoning",
                        "tool_calls": [
                            {
                                "id": f"call{i}",
                                "function": {
                                    "name": "search_building_blocks",
                                    "arguments": '{"query":"CCO"}',
                                },
                            }
                        ],
                    }
                )
                for i in range(len(remaining) - 1)
            ],
            ControlledTermination("fixture", "end"),
        ]
    )
    result = _run(transport, budget_provider=assessed, max_turns=80)
    budgets = [e["payload"] for e in result.state.events if e["event_type"] == "token_budget"]
    assert [b["remaining_tokens"] for b in budgets] == remaining
    first_warning = next(i for i, value in enumerate(remaining) if value <= 5000)
    for i, (budget, (messages, choice)) in enumerate(zip(budgets, transport.requests, strict=True)):
        content = budget["content"]
        strong = remaining[i] <= 2500
        mild = i == first_warning and not strong
        assert ("Context is running low." in content) == mild
        assert ("You must call submit_pathway now" in content) == strong
        assert choice == "auto"  # A warning does not force or terminate the rollout.
        assert sum(m["role"] == "user" for m in messages) == 1
        if i:
            assert messages[-1]["role"] == "tool"
            assert json.loads(messages[-1]["content"])["budget"] == content
            assert all(
                m["reasoning_content"] == "retained reasoning"
                for m in messages
                if m["role"] == "assistant"
            )
    restored = BudgetFeedbackState.from_checkpoint(
        json.loads(json.dumps(result.state.budget_feedback.checkpoint_fields()))
    )
    assert restored.context_warning_sent


def test_context_warning_survives_failed_request_resume_without_repetition():
    def assessed(messages, tools):
        return ContextBudgetSnapshot(10000, 6000, 1000, estimated=False)

    state = RolloutState(messages=[{"role": "user", "content": "plan"}], events=[])
    with pytest.raises(RuntimeError, match="offline"):
        _run(_Transport([RuntimeError("offline")]), state=state, budget_provider=assessed)
    assert state.budget_feedback.context_warning_sent
    resumed = RolloutState(
        messages=copy.deepcopy(state.messages),
        events=copy.deepcopy(state.events),
        budget_feedback=BudgetFeedbackState.from_checkpoint(
            json.loads(json.dumps(state.budget_feedback.checkpoint_fields()))
        ),
    )
    transport = _Transport([ControlledTermination("fixture", "end")])
    _run(transport, state=resumed, budget_provider=assessed)
    assert transport.requests[0][0] == state.messages
    assert sum(e["event_type"] == "token_budget" for e in resumed.events) == 1
    # A later turn at the same occupancy keeps its counts but omits the one-time instruction.
    content, _, status = prepare_budget_feedback(
        assessed,
        [{"role": "tool", "content": "{}"}],
        [],
        resumed.budget_feedback,
        turn_index=1,
    )
    assert status == "emitted" and "submit_pathway" not in content


def test_loop_emits_strong_warning_even_when_context_is_estimated_exhausted():
    state = RolloutState(messages=[{"role": "user", "content": "plan"}], events=[])

    def tiny(messages, tools):
        return ContextBudgetSnapshot(100, 60 + serialized_input_bytes(messages), 100)

    _run(_Transport([ControlledTermination("fixture", "end")]), state=state, budget_provider=tiny)
    assert state.budget_feedback.context_warning_sent
    budget = next(e["payload"] for e in state.events if e["event_type"] == "token_budget")
    assert budget["status"] == "emitted"
    assert "~0 context tokens" in budget["content"]
    assert "You must call submit_pathway now" in budget["content"]


def test_loop_reports_after_tool_results_and_turn_reminders_without_forcing_submission():
    transport = _Transport(
        [
            AssistantTurn(
                {
                    "tool_calls": [
                        {
                            "id": "call1",
                            "function": {
                                "name": "search_building_blocks",
                                "arguments": '{"query":"CCO"}',
                            },
                        },
                        {
                            "id": "call2",
                            "function": {
                                "name": "search_building_blocks",
                                "arguments": '{"query":"N"}',
                            },
                        },
                    ]
                }
            ),
            ControlledTermination("context_window_exceeded", "fixture end"),
        ]
    )
    result = _run(transport)
    budgets = [
        event["payload"] for event in result.state.events if event["event_type"] == "token_budget"
    ]
    assert len(budgets) == 2
    assert [budget["turns_remaining"] for budget in budgets] == [4, 3]
    assert [budget["tool_calls_remaining"] for budget in budgets] == [150, 148]
    assert budgets[1]["remaining_tokens"] < budgets[0]["remaining_tokens"]
    for (messages, choice), budget in zip(transport.requests, budgets, strict=True):
        assert choice == "auto"
        assert (
            f"Remaining: {budget['turns_remaining']} turns, "
            f"{budget['tool_calls_remaining']} tool calls, "
        ) in messages[-1]["content"]
        content = messages[-1]["content"]
        if messages[-1]["role"] == "tool":
            content = json.loads(content)["budget"]
        assert content.endswith(f" {budget['remaining_tokens']} context tokens.")
    assert any(row["role"] == "tool" for row in transport.requests[1][0])
    # Neither the regular budget nor the final-turn warning opens a user turn.
    assert [m["role"] for m in transport.requests[1][0]] == ["user", "assistant", "tool", "tool"]
    last_result = json.loads(transport.requests[1][0][-1]["content"])
    assert "budget" in last_result and "notice" in last_result
    assert any(
        "3 assistant turns remaining" in row.get("content", "") for row in transport.requests[1][0]
    )
    requests = [e["payload"] for e in result.state.events if e["event_type"] == "request_budget"]
    assert (
        budget_summary_fields(result.state.events)["last_context_remaining_tokens"]
        == requests[-1]["remaining_tokens"]
    )


def test_disabled_feedback_leaves_the_request_unchanged_but_still_clips_it():
    transport = _Transport([ControlledTermination("context_window_exceeded", "end")])
    result = _run(transport, enabled=False)
    assert transport.requests == [([{"role": "user", "content": "plan"}], "auto")]
    assert not any(event["event_type"] == "token_budget" for event in result.state.events)
    assert any(event["event_type"] == "request_budget" for event in result.state.events)


def test_failed_request_resume_does_not_duplicate_budget_message():
    state = RolloutState(messages=[{"role": "user", "content": "plan"}], events=[])
    state.policy.total_tool_calls = 5
    with pytest.raises(RuntimeError, match="offline"):
        _run(_Transport([RuntimeError("offline")]), state=state, start=1)
    resumed = RolloutState(
        messages=list(state.messages),
        events=list(state.events),
        policy=copy.deepcopy(state.policy),
        budget_feedback=BudgetFeedbackState.from_checkpoint(
            state.budget_feedback.checkpoint_fields()
        ),
    )
    transport = _Transport([ControlledTermination("context_window_exceeded", "end")])
    _run(transport, state=resumed, start=1)
    assert transport.requests[0][0] == state.messages
    assert "Remaining: 3 turns, 145 tool calls," in state.messages[-1]["content"]
    assert sum("context-window budget" in row.get("content", "") for row in resumed.messages) == 1


@pytest.mark.parametrize("limit,used,turns,remaining", [(0, 151, 4, None), (150, 151, 1, 0)])
def test_loop_feedback_handles_disabled_or_exhausted_tool_limit(limit, used, turns, remaining):
    state = RolloutState(messages=[{"role": "user", "content": "plan"}], events=[])
    state.policy.total_tool_calls = used
    transport = _Transport([ControlledTermination("fixture", "end")])
    result = _run(transport, state=state, max_tool_calls=limit)
    budget = next(
        e["payload"] for e in result.state.events if e.get("event_type") == "token_budget"
    )
    assert budget["turns_remaining"] == turns
    assert budget["tool_calls_remaining"] == remaining
    content = transport.requests[0][0][-1]["content"]
    assert ("0 tool calls" in content) == (remaining == 0)
    if remaining is None:
        assert "tool calls" not in content
    else:
        assert "Remaining: 1 turn, 0 tool calls," in content
        assert "Call submit_pathway" in content


def test_unlimited_budgets_drop_their_clauses_from_the_prompt():
    task = {"system_prompt": "Plan syntheses.", "question": "Plan CCO."}

    def prompt(**budgets):
        return build_initial_messages(task=task, config=RolloutPolicyConfig(**budgets))[0][
            "content"
        ]

    assert (
        prompt(
            max_tool_calls=0,
            max_turns=None,
            max_tokens=None,
            max_model_len=50000,
            context_limit_stated=False,
        )
        == "Plan syntheses."
    )
    turns_only = prompt(
        max_tool_calls=0,
        max_turns=40,
        max_tokens=None,
        max_model_len=50000,
        context_limit_stated=False,
    )
    assert turns_only == "Plan syntheses.\n\nUse no more than 40 assistant turns."
    single = prompt(
        max_tool_calls=1,
        max_turns=1,
        max_tokens=None,
        max_model_len=50000,
        context_limit_stated=False,
    )
    assert single == "Plan syntheses.\n\nUse no more than 1 total tool call or 1 assistant turn."
    context_only = prompt(max_tool_calls=0, max_turns=None, max_tokens=None, max_model_len=50000)
    assert context_only.startswith("Plan syntheses.\n\nYour context-window budget is 50000 tokens")
    assert "Use no more than" not in context_only and "capped" not in context_only


@pytest.mark.parametrize(
    ("input_tokens", "expected"), [(100000, 50000 - 8192 - 0), (None, 50000 - 8192)]
)
def test_an_unlimited_output_cap_is_bounded_only_by_the_context(input_tokens, expected):
    snapshot = ContextBudgetSnapshot(150000 if input_tokens else 50000, input_tokens, None)

    budget = request_output_budget(snapshot, safety_margin_tokens=8192)

    assert budget.max_output_tokens == expected


@pytest.mark.parametrize(
    ("input_tokens", "expected"),
    [
        (100000, 32768),  # Room for the full cap.
        (240000, 13952),  # 262144 - 240000 - 8192: clipped to the context left.
        (254000, None),  # Nothing left after the margin: end before the request.
    ],
)
def test_each_request_is_clipped_and_recorded_before_it_is_sent(input_tokens, expected):
    sent = []

    class Recording(_Transport):
        async def complete(self, messages, *, tools, tool_choice, max_output_tokens=None):
            sent.append(max_output_tokens)
            raise ControlledTermination("context_window_exceeded", "provider rejected the request")

    result = asyncio.run(
        run_rollout(
            task=RolloutTask("fixture", "CCO", ({"role": "user", "content": "plan"},)),
            config=RolloutPolicyConfig(max_model_len=262144),
            transport=Recording([]),
            tool_executor=_Tools(),
            tool_schemas=[],
            budget_provider=lambda *_: ContextBudgetSnapshot(262144, input_tokens, 32768),
        )
    )

    budget = next(e["payload"] for e in result.state.events if e["event_type"] == "request_budget")
    assert result.termination_kind == "context_window_exceeded"
    # A request the provider rejects keeps the budget it was sent with.
    assert sent == ([] if expected is None else [expected])
    assert budget["request_output_tokens"] == (expected or 0)
    assert budget["sent"] is (expected is not None)
    assert budget_summary_fields(result.state.events)["last_request_output_tokens"] == (
        expected or 0
    )
