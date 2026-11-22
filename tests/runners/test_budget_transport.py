# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import copy
import json

import pytest

from synclaw.core import db
from synclaw.rollout.budget import (
    BudgetFeedbackState,
    ContextBudgetSnapshot,
    ContextTokenEstimator,
    prepare_budget_feedback,
    with_harness_feedback,
)
from synclaw.rollout.loop import RolloutTask, run_rollout
from synclaw.rollout.policy import RolloutPolicyConfig
from synclaw.runners import model_runner
from synclaw.tasks.datasets import PROMPT_FINGERPRINT


def _transport(api_mode="chat", state_mode="full", **overrides):
    args = argparse.Namespace(
        api_mode=api_mode,
        responses_state_mode=state_mode,
        model="fixture",
        max_model_len=262144,
        max_tokens=32768,
        responses_url="https://example.test",
        messages_url="https://example.test",
        base_url="https://example.test",
        token_param="max_tokens",
        temperature=1.0,
        reasoning_effort=None,
        retries=0,
        request_timeout=1,
    )
    config = dict(
        args=args,
        config=RolloutPolicyConfig(max_tokens=args.max_tokens, max_model_len=args.max_model_len),
        api_key="test",
        responses_previous_id=None,
        responses_pending_input=[],
        responses_input_history=[],
        anthropic_system=[],
        anthropic_messages=[],
        anthropic_tool_results=[],
        chat_tool_schemas=[],
        responses_tool_schemas=[],
        messages_tool_schemas=[],
    )
    config.update(overrides)
    return model_runner._HostedTransport(**config)


@pytest.mark.parametrize("api_mode", ["chat", "responses", "messages"])
def test_hosted_requests_send_the_output_cap_the_loop_chose(monkeypatch, api_mode):
    transport = _transport(api_mode)
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return {"status": "completed", "output": [], "content": [], "choices": []}

    function = {
        "chat": "chat_completion",
        "responses": "responses_completion",
        "messages": "anthropic_completion",
    }[api_mode]
    monkeypatch.setattr(model_runner, function, complete)
    asyncio.run(transport.complete([], tools=[], tool_choice="auto", max_output_tokens=13952))
    key = "max_output_tokens" if api_mode == "responses" else "max_tokens"
    assert requests[0][key] == 13952


@pytest.mark.parametrize("api_mode", ["chat", "responses", "messages"])
def test_hosted_provider_errors_still_propagate(monkeypatch, api_mode):
    transport = _transport(api_mode)
    error = RuntimeError("Provider rejected the request: context length exceeded")

    def rejected(**kwargs):
        assert kwargs.get("max_output_tokens", kwargs.get("max_tokens")) == 32768
        raise error

    function = {
        "chat": "chat_completion",
        "responses": "responses_completion",
        "messages": "anthropic_completion",
    }[api_mode]
    monkeypatch.setattr(model_runner, function, rejected)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(transport.complete([], tools=[], tool_choice="auto", max_output_tokens=32768))
    assert caught.value is error


@pytest.mark.parametrize(
    "api_mode,state_mode",
    [
        ("chat", "full"),
        ("responses", "full"),
        ("responses", "previous_id"),
        ("messages", "full"),
    ],
)
def test_tool_feedback_preserves_native_reasoning_and_resume(monkeypatch, api_mode, state_mode):
    initial = [{"role": "user", "content": "plan"}]
    thinking = {"type": "thinking", "thinking": "keep my plan", "signature": "signed-unchanged"}
    opaque = {"type": "reasoning", "id": "rs1", "encrypted_content": "encrypted-unchanged"}
    calls = [
        {"type": "function_call", "call_id": key, "name": "search", "arguments": "{}"}
        for key in ("a", "b")
    ]
    native_reply = [
        thinking,
        *[{"type": "tool_use", "id": key, "name": "search", "input": {}} for key in ("a", "b")],
    ]
    response = {
        "id": "r1",
        "status": "completed",
        "output": [opaque, *calls],
        "content": native_reply,
        "usage": {"input_tokens": 5000, "output_tokens": 1000},
    }
    captured = []

    def complete(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        return response

    function = {
        "chat": "chat_completion",
        "responses": "responses_completion",
        "messages": "anthropic_completion",
    }[api_mode]
    monkeypatch.setattr(model_runner, function, complete)
    transport = _transport(
        api_mode,
        state_mode,
        responses_input_history=copy.deepcopy(initial),
        responses_pending_input=copy.deepcopy(initial),
        anthropic_messages=[model_runner.anthropic_text_message("plan")],
    )
    asyncio.run(transport.complete(initial, tools=[], tool_choice="auto", max_output_tokens=32768))
    messages = [
        *initial,
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "keep my plan",
            "tool_calls": [
                {
                    "id": c["call_id"],
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
                for c in calls
            ],
        },
    ]
    for key in ("a", "b"):
        messages.append({"role": "tool", "tool_call_id": key, "content": '{"result":"CCO"}'})
        transport.append_tool(argparse.Namespace(call_id=key), '{"result":"CCO"}')
    before = copy.deepcopy(transport.checkpoint_fields())
    content, snapshot, status = prepare_budget_feedback(
        transport.budget_snapshot,
        messages,
        [],
        BudgetFeedbackState(262144, 32768),
        turn_index=1,
        turns_remaining=79,
        tool_calls_remaining=148,
    )
    assert status == "emitted" and snapshot.input_tokens is not None
    assert transport.checkpoint_fields() == before
    updated = with_harness_feedback(messages, content)
    transport.replace_last_input(updated[-1])
    attached = transport.budget_snapshot(updated, [])
    checkpoint = json.loads(json.dumps(transport.checkpoint_fields()))
    checkpoint["budget_estimator"] = checkpoint.pop("token_budget_estimator")
    resumed = _transport(api_mode, state_mode, **checkpoint)
    assert resumed.budget_snapshot(updated, []) == attached
    asyncio.run(resumed.complete(updated, tools=[], tool_choice="auto", max_output_tokens=32768))
    request = captured[-1]
    if api_mode == "responses":
        history = request["input_items"]
        assert history[-1]["type"] == "function_call_output" and history[-1]["call_id"] == "b"
        assert json.loads(history[-1]["output"])["budget"] == content
        if state_mode == "full":
            assert history[1:4] == [opaque, *calls]
            assert sum(m.get("role") == "user" for m in history) == 1
        else:
            assert request["previous_response_id"] == "r1"
            assert all(m["type"] == "function_call_output" for m in history)
    elif api_mode == "messages":
        history = request["messages"]
        assert len(history) == 3  # Original user, assistant, one batch of tool results.
        assert history[1]["content"] == native_reply
        results = history[2]["content"]
        assert [b["type"] for b in results] == ["tool_result", "tool_result"]
        assert results[0]["content"] == '{"result":"CCO"}'
        assert json.loads(results[1]["content"])["budget"] == content
    else:
        history = request["messages"]
        assert [m["role"] for m in history] == ["user", "assistant", "tool", "tool"]
        assert history[1]["reasoning_content"] == "keep my plan"


@pytest.mark.parametrize(
    "api_mode,state_mode",
    [
        ("chat", "full"),
        ("responses", "full"),
        ("responses", "previous_id"),
        ("messages", "full"),
    ],
)
@pytest.mark.parametrize("with_calls", [True, False])
def test_output_limit_recovery_preserves_provider_history(
    monkeypatch, api_mode, state_mode, with_calls
):
    initial = [{"role": "user", "content": "plan"}]
    thinking = {"type": "thinking", "thinking": "keep my plan", "signature": "signed-unchanged"}
    opaque = {"type": "reasoning", "id": "rs1", "encrypted_content": "encrypted-unchanged"}
    response_calls = (
        [
            {
                "type": "function_call",
                "call_id": "partial",
                "name": "search_building_blocks",
                "arguments": '{"query":',
            }
        ]
        if with_calls
        else []
    )
    native_reply = [
        thinking,
        *(
            [
                {
                    "type": "tool_use",
                    "id": "partial",
                    "name": "search_building_blocks",
                    "input": {},
                },
            ]
            if with_calls
            else []
        ),
    ]
    chat_reply = {"role": "assistant", "content": "", "reasoning_content": "keep my plan"}
    if with_calls:
        chat_reply["tool_calls"] = [
            {
                "id": "partial",
                "type": "function",
                "function": {"name": "search_building_blocks", "arguments": '{"query":'},
            }
        ]
    truncated = {
        "id": "r1",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [opaque, *response_calls],
        "content": native_reply,
        "stop_reason": "max_tokens",
        "choices": [{"message": chat_reply, "finish_reason": "length"}],
        "usage": {"input_tokens": 5000, "output_tokens": 32768},
    }
    captured = []

    def complete(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        if len(captured) == 1:
            return truncated
        return {"id": "r2", "status": "completed", "output": [], "content": [], "choices": []}

    function = {
        "chat": "chat_completion",
        "responses": "responses_completion",
        "messages": "anthropic_completion",
    }[api_mode]
    monkeypatch.setattr(model_runner, function, complete)
    transport = _transport(
        api_mode,
        state_mode,
        responses_input_history=copy.deepcopy(initial),
        responses_pending_input=copy.deepcopy(initial),
        anthropic_messages=[model_runner.anthropic_text_message("plan")],
    )

    class NoTools:
        async def execute(self, *args, **kwargs):
            raise AssertionError("Truncated calls must not execute")

    result = asyncio.run(
        run_rollout(
            task=RolloutTask("test", "CCO", tuple(initial)),
            config=RolloutPolicyConfig(max_turns=2, turn_warning_remaining=0),
            transport=transport,
            tool_executor=NoTools(),
            tool_schemas=[],
            budget_provider=transport.budget_snapshot,
        )
    )
    assert result.state.assistant_turns == 2
    assert result.state.policy.total_tool_calls == 0
    assert result.state.policy.output_limit_recovery_used
    assert not result.partial
    request = captured[1]
    assert 0 < request.get("max_output_tokens", request.get("max_tokens")) <= 32768
    if api_mode == "responses":
        history = request["input_items"]
        if state_mode == "full":
            assert history[1 : 2 + len(response_calls)] == [opaque, *response_calls]
        else:
            assert request["previous_response_id"] == "r1"
        reply = history[-1]
        if with_calls:
            assert reply["type"] == "function_call_output" and reply["call_id"] == "partial"
            assert json.loads(reply["output"])["executed"] is False
            assert "only output-limit recovery attempt" in reply["output"]
        else:
            assert reply["role"] == "user"
            assert any(
                "only output-limit recovery attempt" in m.get("content", "") for m in history
            )
    elif api_mode == "messages":
        history = request["messages"]
        assert history[1]["content"] == native_reply
        reply = history[2]["content"][0]
        if with_calls:
            assert reply["type"] == "tool_result" and reply["tool_use_id"] == "partial"
            assert json.loads(reply["content"])["executed"] is False
        else:
            assert reply["type"] == "text"
        assert "only output-limit recovery attempt" in reply.get("content", reply.get("text"))
    else:
        history = request["messages"]
        assert "keep my plan" in (history[1]["reasoning_content"] or history[1]["content"])
        if with_calls:
            assert history[-1]["role"] == "tool"
            assert json.loads(history[-1]["content"])["executed"] is False
        else:
            assert history[-1]["role"] == "user"
        assert any("only output-limit recovery attempt" in m.get("content", "") for m in history)


def test_recovery_request_is_clipped_to_the_context_it_leaves(monkeypatch):
    transport = _transport()
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}

    def budget(messages, tools):
        recovering = any(
            "only output-limit recovery attempt" in m.get("content", "") for m in messages
        )
        return ContextBudgetSnapshot(262144, 240000 if recovering else 220000, 32768)

    monkeypatch.setattr(model_runner, "chat_completion", complete)
    monkeypatch.setattr(transport, "budget_snapshot", budget)

    class NoTools:
        async def execute(self, *args, **kwargs):
            raise AssertionError("No tools expected")

    result = asyncio.run(
        run_rollout(
            task=RolloutTask("test", "CCO", ({"role": "user", "content": "plan"},)),
            config=RolloutPolicyConfig(max_turns=4, turn_warning_remaining=0, max_model_len=262144),
            transport=transport,
            tool_executor=NoTools(),
            tool_schemas=[],
            budget_provider=budget,
        )
    )
    # The cap is recalculated after the recovery note is appended.
    assert [r["max_tokens"] for r in requests] == [32768, 262144 - 240000 - 8192]
    # The recovery's cap was clipped to the context left, so hitting it means the context ran out.
    assert result.partial and result.reason == "context_window_exceeded"
    assert result.state.assistant_turns == 2
    request_budgets = [
        e["payload"] for e in result.state.events if e["event_type"] == "request_budget"
    ]
    assert [b["request_output_tokens"] for b in request_budgets] == [32768, 13952]
    assert all(b["configured_output_tokens"] == 32768 for b in request_budgets)


def test_full_replay_never_relies_on_stored_responses(monkeypatch):
    # For endpoints without server-side state (e.g. zero data retention), the
    # caller passes --responses-state-mode full; nothing infers it from the model.
    transport = _transport("responses", "previous_id")
    transport.args.provider = "openai"
    transport.args.responses_state_mode = "full"
    model_runner.apply_provider_defaults(transport.args)
    assert transport.args.responses_state_mode == "full"

    captured = []
    call = {"type": "function_call", "call_id": "call1", "name": "search", "arguments": "{}"}

    def completion(**kwargs):
        captured.append({**kwargs, "input_items": list(kwargs["input_items"])})
        return {"id": "response1", "output": [call], "status": "completed"}

    monkeypatch.setattr(model_runner, "responses_completion", completion)
    for _ in range(2):
        asyncio.run(transport.complete([], tools=[], tool_choice="auto", max_output_tokens=32768))
    assert captured[1]["input_items"] == [call]
    assert all(row["previous_response_id"] is None and row["store"] is False for row in captured)


def test_chat_budget_counts_reasoning_and_tools_but_not_trace_metadata():
    transport = _transport()
    messages = [{"role": "assistant", "content": "route", "reasoning_content": "C" * 900}]
    tools = [{"type": "function", "function": {"name": "search", "description": "D" * 300}}]
    snapshot = transport.budget_snapshot(messages, tools)
    assert (
        snapshot.input_tokens
        > transport.budget_snapshot([{"role": "assistant", "content": "route"}], []).input_tokens
    )
    messages[0]["response_usage"] = {"ignored": "x" * 10000}
    messages[0]["token_usage"] = {"total_tokens": 999999}
    assert transport.budget_snapshot(messages, tools) == snapshot
    assert snapshot.max_output_tokens == 32768


def test_responses_continuation_rebases_usage_and_restores_checkpoint(monkeypatch):
    initial = [{"role": "user", "content": "plan"}]
    transport = _transport(
        "responses",
        "previous_id",
        responses_pending_input=list(initial),
        responses_input_history=list(initial),
    )
    monkeypatch.setattr(
        model_runner,
        "responses_completion",
        lambda **kwargs: {
            "id": "response1",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 5000,
                "output_tokens": 1000,
                "output_tokens_details": {"reasoning_tokens": 800},
            },
        },
    )
    asyncio.run(transport.complete(initial, tools=[], tool_choice="auto", max_output_tokens=32768))
    assert transport.budget_estimator.continuation_tokens == 6000
    transport.append_user({"role": "user", "content": "continue"})
    snapshot = transport.budget_snapshot(initial, [])
    assert snapshot.input_tokens > 6000
    assert snapshot.source == "provider_usage_plus_new_input"
    fields = transport.checkpoint_fields()
    resumed = _transport(
        "responses",
        "previous_id",
        responses_previous_id=fields["responses_previous_id"],
        responses_pending_input=fields["responses_pending_input"],
        responses_input_history=fields["responses_input_history"],
        budget_estimator=fields["token_budget_estimator"],
    )
    assert resumed.budget_snapshot(initial, []) == snapshot


def test_server_managed_history_without_usage_is_unknown():
    transport = _transport("responses", "previous_id", responses_previous_id="legacy")
    snapshot = transport.budget_snapshot([], [])
    assert snapshot.input_tokens is None
    assert snapshot.remaining_tokens is None
    assert snapshot.max_output_tokens == 32768


def test_full_replay_keeps_feedback_after_encrypted_reasoning_and_checkpoint(monkeypatch):
    initial = [{"role": "user", "content": "plan"}]
    transport = _transport(
        "responses",
        "full",
        responses_pending_input=list(initial),
        responses_input_history=list(initial),
    )
    responses = iter(
        [
            {
                "id": "r1",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs1",
                        "encrypted_content": "ciphertext" * 1000,
                    }
                ],
                "usage": {"input_tokens": 5000, "output_tokens": 1000},
            },
            {"id": "r2", "output": [], "usage": {"input_tokens": 7000, "output_tokens": 100}},
        ]
    )
    monkeypatch.setattr(model_runner, "responses_completion", lambda **kwargs: next(responses))
    asyncio.run(transport.complete(initial, tools=[], tool_choice="auto", max_output_tokens=32768))
    transport.append_tool(argparse.Namespace(call_id="call1"), '{"result": "CCO"}')
    before = copy.deepcopy(transport.checkpoint_fields())

    pending = [{"role": "tool", "tool_call_id": "call1", "content": '{"result": "CCO"}'}]
    content, snapshot, status = prepare_budget_feedback(
        transport.budget_snapshot,
        pending,
        [],
        BudgetFeedbackState(),
        turn_index=1,
    )
    assert status == "emitted"
    assert "~" in content and "context tokens." in content
    assert snapshot.source == "provider_usage_plus_replay_delta"
    assert snapshot.input_tokens > 6000
    assert transport.checkpoint_fields() == before  # Preview never mutates history.
    transport.replace_last_input(with_harness_feedback(pending, content)[-1])
    attached = transport.budget_snapshot([], [])
    fields = json.loads(json.dumps(transport.checkpoint_fields(), sort_keys=True))
    resumed = _transport(
        "responses",
        "full",
        responses_input_history=fields["responses_input_history"],
        responses_pending_input=fields["responses_pending_input"],
        budget_estimator=fields["token_budget_estimator"],
    )
    assert resumed.budget_snapshot([], []) == attached
    asyncio.run(resumed.complete([], tools=[], tool_choice="auto", max_output_tokens=32768))
    assert resumed.budget_snapshot([], []).input_tokens == 7100


def test_anthropic_preview_includes_pending_tools_and_observes_cache_usage(monkeypatch):
    transport = _transport(
        "messages",
        anthropic_system=[{"type": "text", "text": "plan"}],
        anthropic_tool_results=[{"type": "tool_result", "tool_use_id": "a", "content": "C" * 600}],
    )
    preview = transport._budget_payload([], [])
    assert preview["messages"][0]["content"] == transport._anthropic_tool_results
    assert transport.anthropic_messages == []
    assert transport._anthropic_tool_results
    captured = {}

    def complete(**kwargs):
        captured.update(kwargs)
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 300,
                "output_tokens": 50,
            },
        }

    monkeypatch.setattr(model_runner, "anthropic_completion", complete)
    asyncio.run(transport.complete([], tools=[], tool_choice="auto", max_output_tokens=32768))
    # The actual input before appending the response matches the side-effect-free preview.
    assert captured["messages"][:-1] == preview["messages"]
    assert transport.budget_estimator.continuation_tokens == 650
    assert (
        ContextTokenEstimator.from_checkpoint(
            transport.checkpoint_fields()["token_budget_estimator"]
        )
        == transport.budget_estimator
    )


def test_budget_cli_and_checkpoint_settings():
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    args = parser.parse_args(
        ["--model", "fixture", "--max-model-len", "262144", "--no-budget-feedback"]
    )
    assert args.max_model_len == 262144
    assert args.budget_feedback is False
    assert args.context_safety_margin_fraction == 1 / 32
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "fixture", "--max-model-len", "0"])
    args.responses_state_mode = "full"
    task = {"task_id": "t", "target_smiles": "CCO"}
    config = RolloutPolicyConfig(max_model_len=262144, budget_feedback=False)
    checkpoint = {
        "model": args.model,
        "provider": args.provider,
        "api_mode": args.api_mode,
        "responses_state_mode": args.responses_state_mode,
        **task,
        **config.budget_fields,
        "stop_condition": args.stop_condition,
    }
    assert model_runner.checkpoint_matches_run(checkpoint, args, task, config)
    # The restored prompt states these budgets, so a resume under others is refused.
    for field, value in (
        ("max_model_len", 100000),
        ("budget_feedback", True),
        ("max_turns", 40),
        ("max_tool_calls", 75),
        ("max_tokens", 16384),
        ("context_safety_margin_fraction", 0.0),
    ):
        assert not model_runner.checkpoint_matches_run(
            {**checkpoint, field: value}, args, task, config
        )
    checkpoint["stop_condition"] = 1.0
    assert not model_runner.checkpoint_matches_run(checkpoint, args, task, config)


@pytest.mark.parametrize("api_mode", ["chat", "responses", "messages"])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("max_tool_calls", [0, 150])
def test_hosted_initial_budget_is_system_content_and_resume_preserves_it(
    tmp_path, monkeypatch, api_mode, enabled, max_tool_calls
):
    taskset = tmp_path / "tasks.jsonl"
    taskset.write_text(
        json.dumps(
            {
                "task_id": "fixture",
                "target_smiles": "CCO",
                "system_prompt": "Plan syntheses.",
                "question": "Make CCO.",
                "prompt_fingerprint": PROMPT_FINGERPRINT,
            }
        )
        + "\n"
    )
    db.init_db(tmp_path / "empty.sqlite")
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    argv = [
        "--model",
        "fixture",
        "--api-mode",
        api_mode,
        "--responses-state-mode",
        "full",
        "--base-url",
        "https://example.test/v1",
        "--responses-url",
        "https://example.test/v1/responses",
        "--messages-url",
        "https://example.test/v1/messages",
        "--taskset",
        str(taskset),
        "--db",
        str(tmp_path / "empty.sqlite"),
        "--trace",
        str(tmp_path / "trace.jsonl"),
        "--checkpoint",
        str(tmp_path / "checkpoint.json"),
        "--max-model-len",
        "50000",
        "--max-tokens",
        "2000",
        "--max-turns",
        "37",
        "--max-tool-calls",
        str(max_tool_calls),
    ]
    if not enabled:
        argv.append("--no-budget-feedback")
    args = parser.parse_args(argv)
    monkeypatch.setattr(model_runner, "resolve_api_key", lambda *args, **kwargs: "fixture-key")
    monkeypatch.setattr(db, "validate_model_run_database", lambda *args, **kwargs: None)
    requests = []

    def fail_request(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        raise RuntimeError("fixture interruption")

    function = {
        "chat": "chat_completion",
        "responses": "responses_completion",
        "messages": "anthropic_completion",
    }[api_mode]
    monkeypatch.setattr(model_runner, function, fail_request)
    # The second invocation resumes the first failed request from its checkpoint.
    for _ in range(2):
        with pytest.raises(RuntimeError, match="fixture interruption"):
            model_runner.run_model_from_args(args)
    assert requests[0] == requests[1]
    if api_mode == "messages":
        system = json.dumps(requests[0]["system"])
        following = json.dumps(requests[0]["messages"])
    else:
        history = requests[0]["input_items" if api_mode == "responses" else "messages"]
        system = history[0]["content"]
        following = json.dumps(history[1:])
    # The system prompt states the limits whether or not progress feedback is on.
    assert system.count("context-window budget is 50000") == 1
    assert system.count("Each response is capped at 2000") == 1
    if max_tool_calls:
        assert "Use no more than 150 total tool calls or 37 assistant turns." in system
    else:
        assert "Use no more than 37 assistant turns." in system
        assert "total tool calls" not in system
    assert "context-window budget" not in following and "Each response is capped" not in following
    assert ("context tokens." in following) == enabled
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert sum(e.get("event_type") == "token_budget" for e in events) == int(enabled)


def test_task_budgets_set_the_prompt_request_and_checkpoint(tmp_path, monkeypatch):
    taskset = tmp_path / "tasks.jsonl"
    taskset.write_text(
        json.dumps(
            {
                "task_id": "fixture",
                "target_smiles": "CCO",
                "system_prompt": "Plan syntheses.",
                "question": "Make CCO.",
                "prompt_fingerprint": PROMPT_FINGERPRINT,
                "max_turns": 5,
                "max_tool_calls": 7,
                "max_model_len": 40000,
                "max_tokens": 1000,
                "budget_feedback": False,
            }
        )
        + "\n"
    )
    db.init_db(tmp_path / "empty.sqlite")
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    args = parser.parse_args(
        [
            "--model",
            "fixture",
            "--api-mode",
            "chat",
            "--taskset",
            str(taskset),
            "--db",
            str(tmp_path / "empty.sqlite"),
            "--trace",
            str(tmp_path / "trace.jsonl"),
            "--checkpoint",
            str(tmp_path / "checkpoint.json"),
            "--max-model-len",
            "50000",
            "--max-tokens",
            "2000",
        ]
    )
    monkeypatch.setattr(model_runner, "resolve_api_key", lambda *args, **kwargs: "fixture-key")
    monkeypatch.setattr(db, "validate_model_run_database", lambda *args, **kwargs: None)
    requests = []

    def fail_request(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        raise RuntimeError("fixture interruption")

    monkeypatch.setattr(model_runner, "chat_completion", fail_request)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        model_runner.run_model_from_args(args)

    system = requests[0]["messages"][0]["content"]
    assert "Use no more than 7 total tool calls or 5 assistant turns." in system
    assert "context-window budget is 40000" in system
    assert "Each response is capped at 1000" in system
    assert requests[0]["max_tokens"] == 1000
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert {
        name: checkpoint[name]
        for name in ("max_turns", "max_tool_calls", "max_model_len", "max_tokens")
    } == {
        "max_turns": 5,
        "max_tool_calls": 7,
        "max_model_len": 40000,
        "max_tokens": 1000,
    }
    assert checkpoint["budget_feedback"] is False
    # A task may not ask for more context than the runner allows.
    row = json.loads(taskset.read_text())
    taskset.write_text(json.dumps({**row, "max_model_len": 60000}) + "\n")
    (tmp_path / "checkpoint.json").unlink()
    with pytest.raises(RuntimeError, match="exceeds the runner's context limit 50000"):
        model_runner.run_model_from_args(args)
