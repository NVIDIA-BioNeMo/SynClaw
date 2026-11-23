# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from synclaw.core import chem, db
from synclaw.rollout.budget import ContextBudgetSnapshot
from synclaw.rollout.loop import (
    AssistantTurn,
    ControlledTermination,
    LocalToolExecutor,
    RolloutState,
    RolloutTask,
    ToolExecution,
    call_local_tool,
    run_rollout,
)
from synclaw.rollout.policy import (
    RolloutPolicyConfig,
    RolloutPolicyState,
    prepare_tool_call,
    tool_argument_error,
)
from synclaw.token_usage import aggregate_token_usage_from_events
from synclaw.tools.contracts import filter_chat_tool_schemas
from synclaw.tools.synthesis import SynthesisTools, score_pathway_against_target


class FakeTransport:
    def __init__(self, turns: list[AssistantTurn | Exception]) -> None:
        self.turns = turns
        self.requests: list[list[dict[str, Any]]] = []
        self.tool_choices: list[Any] = []

    async def complete(self, messages, *, tools, tool_choice, max_output_tokens=None):
        self.requests.append([dict(message) for message in messages])
        self.tool_choices.append(tool_choice)
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    def append_user(self, message):
        pass

    def append_tool(self, call, result_json):
        pass

    def replace_last_input(self, message):
        pass


class FakeTools:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls: list[str] = []

    async def execute(self, calls, *, target_smiles, assistant_turn):
        del target_smiles, assistant_turn
        self.calls.extend(call.name for call in calls)
        return [
            ToolExecution(
                tool_argument_error(call.name, call.argument_error, call.arguments)
                if call.argument_error is not None
                else self.results.pop(0)
            )
            for call in calls
        ]


def config(**overrides: Any) -> RolloutPolicyConfig:
    values = {
        "max_turns": 4,
        "max_tool_calls": 10,
        "stop_condition": 1.0,
        "turn_warning_remaining": 0,
        "same_pathway_finish_after": 2,
        "max_low_similarity_submissions": None,
    }
    values.update(overrides)
    return RolloutPolicyConfig(**values)


def task() -> RolloutTask:
    return RolloutTask(
        task_id="target-1",
        target_smiles="CCO",
        initial_messages=(
            {"role": "system", "content": "system"},
            {"role": "user", "content": "target"},
        ),
    )


TOOL_SCHEMAS = filter_chat_tool_schemas(set())


def _run_shared_rollout(
    transport: FakeTransport,
    tools: FakeTools | None = None,
    *,
    rollout_config: RolloutPolicyConfig | None = None,
    **kwargs: Any,
):
    return asyncio.run(
        run_rollout(
            task=task(),
            config=rollout_config or config(),
            transport=transport,
            tool_executor=tools or FakeTools([]),
            tool_schemas=TOOL_SCHEMAS,
            **kwargs,
        )
    )


def tool_turn(name: str, arguments: dict[str, Any]) -> AssistantTurn:
    return AssistantTurn(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
            "response_usage": {"output_tokens": 7},
        }
    )


@pytest.mark.parametrize(
    ("name", "arguments", "expected_issue"),
    [
        ("get_reaction_template", {"reaction_id": "R001"}, "Unknown tool name"),
        ("search_building_blocks", {"query": ""}, "missing required argument 'query'"),
    ],
)
def test_local_tool_executor_rejects_undeclared_or_invalid_calls(
    name: str, arguments: dict[str, Any], expected_issue: str
) -> None:
    class GuardedTools:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_reaction_template(self, **_kwargs: Any) -> dict[str, Any]:
            self.calls.append("get_reaction_template")
            return {"found": True}

        def search_building_blocks(self, **_kwargs: Any) -> dict[str, Any]:
            self.calls.append("search_building_blocks")
            return {"valid": True}

    tools = GuardedTools()
    call = prepare_tool_call(
        raw_name=name,
        raw_arguments=arguments,
        raw_tool_call={"id": "call-1", "type": "function"},
        event_index=0,
    )

    execution = asyncio.run(
        LocalToolExecutor(tools).execute([call], target_smiles="CCO", assistant_turn=1)
    )[0]

    assert execution.result["error"] == "invalid_tool_call"
    assert execution.result["retryable"] is True
    assert expected_issue in execution.result["issues"][0]
    assert tools.calls == []


def test_local_tool_dispatcher_defaults_the_template_search_limit() -> None:
    class SearchTools:
        def search_reaction_templates(self, **arguments: Any) -> dict[str, Any]:
            return arguments

    result = call_local_tool(SearchTools(), "search_reaction_templates", {})

    assert result["limit"] == 10


def test_local_tool_dispatcher_does_not_hide_harness_errors() -> None:
    class BrokenTools:
        def mces(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
            raise RuntimeError("index missing")

    with pytest.raises(RuntimeError, match="index missing"):
        call_local_tool(BrokenTools(), "mces", {"smiles_a": "C", "smiles_b": "CC"})


def test_local_tool_dispatcher_returns_an_error_for_a_non_finite_limit(tmp_path: Path) -> None:
    (tmp_path / "bbs.txt").write_text("SMILES\nCCO\n", encoding="utf-8")
    database = tmp_path / "test.sqlite"
    db.ingest_building_blocks(tmp_path / "bbs.txt", database)
    # json.loads("1e999") is how a model's oversized number arrives.
    arguments = {"query": "ester", "limit": json.loads("1e999")}

    result = call_local_tool(SynthesisTools(database), "search_reaction_templates", arguments)

    assert result["error"] == "invalid_tool_call"
    assert "must be a whole number" in result["issues"][0]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        (
            "simulate_reaction",
            {"reaction_id": "R001", "reactants": ["BB:99999999999999999999"]},
        ),
        (
            "submit_pathway",
            {
                "steps": [
                    {
                        "reaction_id": "R001",
                        "reactants": ["BB:99999999999999999999"],
                        "product_smiles": "CC=O",
                    }
                ]
            },
        ),
    ],
)
def test_local_tool_dispatcher_reports_an_out_of_range_building_block_id(
    tmp_path: Path, name: str, arguments: dict[str, Any]
) -> None:
    (tmp_path / "bbs.txt").write_text("SMILES\nCCO\n", encoding="utf-8")
    database = tmp_path / "test.sqlite"
    db.ingest_building_blocks(tmp_path / "bbs.txt", database)

    result = call_local_tool(SynthesisTools(database), name, arguments, target_smiles="CC=O")

    assert "unknown building block reference: BB:99999999999999999999" in " ".join(result["issues"])


@pytest.mark.parametrize("step", ["BB:1", 1, None])
def test_local_tool_dispatcher_rejects_non_object_steps(step: Any) -> None:
    result = call_local_tool(object(), "submit_pathway", {"steps": [step], "pathway_text": "x"})

    assert result["error"] == "invalid_tool_call"
    assert "array of objects" in result["issues"][0]


@pytest.mark.parametrize("max_tool_calls", [2, 10])
@pytest.mark.parametrize("floor", ["exact", 1.0])
@pytest.mark.parametrize(
    "products",
    [
        [("Oc1ccccn1", True), ("O=c1ccccn1C", True)],
        [("O=c1ccccn1C", True), ("Oc1ccccn1", True)],
        [("Oc1ccccn1", True), ("O=c1ccccn1C", False)],
        [("Oc1ccccn1", True), ("O=c1cccc[nH]1", True)],
    ],
)
def test_exact_submission_ends_after_its_batch_without_losing_identity(
    max_tool_calls,
    floor,
    products,
) -> None:
    target = "O=c1cccc[nH]1"
    results = [
        {
            "accepted": accepted,
            **score_pathway_against_target(
                [product],
                target_smiles=target,
                target_validation=chem.validate_smiles(target),
                preferred_smiles=None,
                accepted=accepted,
            ),
        }
        for product, accepted in products
    ]
    calls = [
        {
            "id": f"call-{i}",
            "type": "function",
            "function": {
                "name": "submit_pathway",
                "arguments": json.dumps({"steps": [], "building_block": f"BB:{i}"}),
            },
        }
        for i in (1, 2)
    ]
    transport = FakeTransport([AssistantTurn({"role": "assistant", "tool_calls": calls})])
    tools = FakeTools(results.copy())
    result = asyncio.run(
        run_rollout(
            task=RolloutTask("tautomer", target, task().initial_messages),
            config=config(stop_condition=floor, max_tool_calls=max_tool_calls),
            transport=transport,
            tool_executor=tools,
            tool_schemas=TOOL_SCHEMAS,
        )
    )

    assert len(transport.requests) == 1
    assert result.reason == "terminal_submission"
    # Among identical products the policy keeps the higher-similarity one.
    exact = max(
        (r for r in results if r["identical_to_target"] is True),
        key=lambda r: r["similarity_to_target"],
    )
    assert result.submission["actual_final_smiles"] == exact["actual_final_smiles"]
    assert result.submission["identical_to_target"] is True
    assert result.state.policy.best_accepted_similarity == exact["similarity_to_target"]
    assert result.state.policy.total_tool_calls == 2
    assert sum(e["event_type"] == "tool_result" for e in result.state.events) == 2
    assert result.state.events[-1]["payload"]["identical_to_target"] is True


def test_with_a_model_stop_condition_the_model_answers_after_an_exact_submission() -> None:
    exact = {
        "accepted": True,
        "actual_final_smiles": "CCO",
        "similarity_to_target": 1.0,
    }
    transport = FakeTransport(
        [
            tool_turn("submit_pathway", {"steps": []}),
            AssistantTurn({"content": "The submitted route makes the target."}, "stop"),
        ]
    )

    result = _run_shared_rollout(
        transport, FakeTools([exact]), rollout_config=config(stop_condition="model")
    )

    assert result.reason == "model_stop_best_accepted_submission"
    assert result.submission is not None and result.state.policy.best_accepted_identical
    assert result.state.messages[-1] == {
        "role": "assistant",
        "content": "The submitted route makes the target.",
    }


def test_shared_rollout_selects_accepted_submission_and_records_trace() -> None:
    transport = FakeTransport([tool_turn("submit_pathway", {"steps": []})])
    tools = FakeTools(
        [
            {
                "accepted": True,
                "actual_final_smiles": "CCO",
                "similarity_to_target": 1.0,
                "constitutional_similarity_to_target": 1.0,
            }
        ]
    )

    result = _run_shared_rollout(transport, tools)

    assert result.reason == "terminal_submission"
    assert result.submission is not None and result.submission["accepted"] is True
    assert tools.calls == ["submit_pathway"]
    assert result.state.assistant_turns == 1
    # Token usage is derived from the recorded events by one shared accumulator
    # rather than tallied a second time on the state.
    assert aggregate_token_usage_from_events(result.state.events)["output_tokens"] == 7
    assert [event["event_type"] for event in result.state.events] == [
        "message",
        "message",
        "request_budget",
        "message",
        "tool_call",
        "tool_result",
        "selected_submission",
    ]
    selected_payload = result.state.events[-1]["payload"]
    assert set(selected_payload) == {
        "reason",
        "similarity_to_target",
        "identical_to_target",
        "constitutional_similarity_to_target",
        "submission_arguments",
        "result",
        "model_result",
        "model_result_compaction",
    }
    assert selected_payload["result"]["accepted"] is True


def test_shared_rollout_handles_excluded_tool_without_dispatching_it() -> None:
    transport = FakeTransport([tool_turn("similarity", {"smiles_a": "CC", "smiles_b": "CO"})])
    tools = FakeTools([])

    result = _run_shared_rollout(
        transport,
        tools,
        rollout_config=config(max_turns=1, excluded_tools=frozenset({"similarity"})),
    )

    assert result.reason == "no_accepted_submission"
    assert tools.calls == []
    tool_result = next(
        event for event in result.state.events if event["event_type"] == "tool_result"
    )
    assert tool_result["payload"]["result"]["error"] == "invalid_tool_call"


def test_shared_rollout_keeps_rejected_submission_but_does_not_score_it() -> None:
    transport = FakeTransport([tool_turn("submit_pathway", {"steps": []})])
    tools = FakeTools(
        [
            {
                "accepted": False,
                "actual_final_smiles": "CCO",
                "final_smiles": "CCO",
                "constitutional_similarity_to_target": 1.0,
                "similarity_to_target": 1.0,
            }
        ]
    )

    result = _run_shared_rollout(
        transport,
        tools,
        rollout_config=config(max_turns=1),
    )

    assert result.reason == "no_accepted_submission"
    assert result.submission is not None and result.submission["accepted"] is False
    assert result.submission_arguments == {"steps": []}
    selected = next(
        event for event in result.state.events if event["event_type"] == "selected_submission"
    )
    assert selected["payload"]["similarity_to_target"] is None


def test_shared_rollout_replays_malformed_tool_arguments_as_canonical_json() -> None:
    malformed = AssistantTurn(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "bad-call",
                    "type": "function",
                    "function": {"name": "similarity", "arguments": "{\n"},
                }
            ],
        }
    )
    transport = FakeTransport([malformed, ControlledTermination("context_window_exceeded", "done")])

    result = _run_shared_rollout(transport)

    replayed = next(
        message for message in transport.requests[1] if message.get("role") == "assistant"
    )
    assert json.loads(replayed["tool_calls"][0]["function"]["arguments"]) == {}
    raw_event = next(event for event in result.state.events if event["event_type"] == "tool_call")
    assert raw_event["payload"]["raw_tool_call"]["function"]["arguments"] == "{\n"


def test_shared_rollout_returns_partial_result_for_controlled_termination() -> None:
    transport = FakeTransport(
        [ControlledTermination("context_window_exceeded", "context exhausted")]
    )

    result = _run_shared_rollout(transport)

    assert result.partial is True
    assert result.reason == "context_window_exceeded"
    assert result.termination_message == "context exhausted"
    assert result.state.events[-1]["event_type"] == "controlled_termination"


def test_shared_rollout_marks_responses_output_limit_as_partial() -> None:
    transport = FakeTransport(
        [
            AssistantTurn(
                {"role": "assistant", "content": "Truncated reasoning"},
                finish_reason="max_output_tokens",
            )
        ]
    )

    result = _run_shared_rollout(transport, rollout_config=config(max_turns=1))

    assert result.reason == "max_tokens"
    assert result.partial is True
    assert result.termination_kind == "max_tokens"
    assert result.state.events[-1]["event_type"] == "controlled_termination"


@pytest.mark.parametrize("estimated_input_tokens", [40000, 220000])
def test_output_limit_recovers_once_without_resetting_budgets(estimated_input_tokens):
    finish_reason = "length"
    truncated = AssistantTurn(
        {
            "content": "Partial answer",
            "reasoning_content": "Partial plan",
            "response_usage": {"output_tokens": 32768},
        },
        finish_reason=finish_reason,
    )
    transport = FakeTransport([truncated, tool_turn("submit_pathway", {"steps": []})])
    tools = FakeTools([{"accepted": True, "similarity_to_target": 1.0}])
    result = _run_shared_rollout(
        transport,
        tools,
        rollout_config=config(max_turns=2),
        budget_provider=lambda *_: ContextBudgetSnapshot(262144, estimated_input_tokens, 32768),
    )
    assert result.reason == "terminal_submission" and not result.partial
    assert result.state.assistant_turns == 2
    assert result.state.policy.total_tool_calls == 1
    assert result.state.policy.output_limit_recovery_used
    assert aggregate_token_usage_from_events(result.state.events)["output_tokens"] == 32775
    replay = transport.requests[1]
    assert replay[-1]["role"] == "user"
    assert any("only output-limit recovery attempt" in m.get("content", "") for m in replay)
    assistant = next(m for m in replay if m["role"] == "assistant")
    assert "Partial plan" in assistant["content"]
    assert "Partial answer" in assistant["content"]
    assert not assistant["reasoning_content"]
    original = next(e["payload"] for e in result.state.events if e.get("role") == "assistant")
    assert original["reasoning_content"] == "Partial plan"
    assert original["content"] == "Partial answer"
    budgets = [e["payload"] for e in result.state.events if e["event_type"] == "token_budget"]
    assert budgets[-1]["turns_remaining"] == 1
    assert budgets[-1]["tool_calls_remaining"] == 10


def test_without_progress_feedback_an_output_limit_ends_the_rollout():
    truncated = AssistantTurn({"content": "Partial answer"}, finish_reason="length")
    transport = FakeTransport([truncated, tool_turn("submit_pathway", {"steps": []})])
    result = _run_shared_rollout(
        transport,
        FakeTools([]),
        rollout_config=config(max_turns=4, budget_feedback=False),
    )

    assert result.reason == "max_tokens" and result.partial
    assert len(transport.requests) == 1
    assert not result.state.policy.output_limit_recovery_used


def test_truncated_tool_batch_is_not_executed_and_keeps_reasoning_boundary():
    calls = [
        {
            "id": "complete",
            "type": "function",
            "function": {"name": "submit_pathway", "arguments": '{"steps": []}'},
        },
        {
            "id": "partial",
            "type": "function",
            "function": {"name": "simulate_reaction", "arguments": '{"reaction_id":'},
        },
    ]
    transport = FakeTransport(
        [
            AssistantTurn({"tool_calls": calls, "reasoning_content": "Keep plan"}, "length"),
            tool_turn("submit_pathway", {"steps": []}),
        ]
    )
    tools = FakeTools([{"accepted": True, "similarity_to_target": 1.0}])
    result = _run_shared_rollout(transport, tools)
    assert tools.calls == ["submit_pathway"]
    assert result.state.policy.total_tool_calls == 1
    replay = transport.requests[1]
    assert [m["role"] for m in replay] == ["system", "user", "assistant", "tool", "tool"]
    assert replay[2]["reasoning_content"] == "Keep plan"
    assert "reaction_id" not in json.loads(replay[2]["tool_calls"][1]["function"]["arguments"])
    for reply, call in zip(replay[-2:], calls, strict=True):
        assert reply["tool_call_id"] == call["id"]
        assert json.loads(reply["content"])["executed"] is False
    assert "only output-limit recovery attempt" in replay[-1]["content"]
    assert json.loads(replay[-1]["content"])["truncated_arguments"] == '{"reaction_id":'
    assert sum(e["event_type"] == "tool_result" for e in result.state.events) == 1
    original = next(e["payload"] for e in result.state.events if e.get("role") == "assistant")
    assert original["tool_calls"] == calls


def test_reasoning_only_recovery_keeps_work_from_earlier_tool_turns():
    search = tool_turn("search_building_blocks", {"query": "CCO"})
    search.message["reasoning_content"] = "First plan before search"
    transport = FakeTransport(
        [
            search,
            AssistantTurn({"reasoning_content": "Continue plan after search"}, "length"),
            tool_turn("submit_pathway", {"steps": []}),
        ]
    )
    tools = FakeTools([{"matches": []}, {"accepted": True, "similarity_to_target": 1.0}])
    result = _run_shared_rollout(transport, tools)
    assistants = [m for m in transport.requests[2] if m["role"] == "assistant"]
    assert "First plan before search" in assistants[0]["content"]
    assert "Continue plan after search" in assistants[1]["content"]
    assert all(not m["reasoning_content"] for m in assistants)
    assert result.state.policy.total_tool_calls == 2
    assert result.reason == "terminal_submission"


@pytest.mark.parametrize("blocked_by", ["turns", "tool_finalization", "missing_call_id"])
def test_output_limit_does_not_recover_without_budget_or_valid_protocol(blocked_by):
    policy = RolloutPolicyState(tool_limit_finalization_requested=blocked_by == "tool_finalization")
    message = {"content": "partial"}
    if blocked_by == "missing_call_id":
        message["tool_calls"] = [{"function": {"name": "submit_pathway", "arguments": "{"}}]
    transport = FakeTransport([AssistantTurn(message, "length")])
    result = _run_shared_rollout(
        transport,
        state=RolloutState(messages=list(task().initial_messages), events=[], policy=policy),
        rollout_config=config(max_turns=1 if blocked_by == "turns" else 4),
    )
    assert result.partial and result.reason == "max_tokens"
    assert len(transport.requests) == 1
    assert not result.state.policy.output_limit_recovery_used


def test_output_limit_retry_is_bounded_across_checkpoint_resume_and_keeps_best_route():
    analogue = {"accepted": True, "similarity_to_target": 0.6}
    state = RolloutState(messages=list(task().initial_messages), events=[])
    saved = {}

    def checkpoint(_event, next_turn):
        if next_turn == 2:
            saved.update(
                copy.deepcopy(
                    {
                        "messages": state.messages,
                        "events": state.events,
                        "policy": state.policy.checkpoint_fields(),
                        "assistant_turns": state.assistant_turns,
                    }
                )
            )

    transport = FakeTransport(
        [
            tool_turn("submit_pathway", {"steps": []}),
            AssistantTurn({"reasoning_content": "Improve analogue"}, "length"),
            RuntimeError("network interrupted"),
        ]
    )
    with pytest.raises(RuntimeError, match="network interrupted"):
        _run_shared_rollout(
            transport, FakeTools([analogue]), state=state, event_callback=checkpoint
        )
    assert saved["policy"]["output_limit_recovery_used"]
    saved["policy"] = RolloutPolicyState.from_checkpoint(saved["policy"])
    restored = RolloutState(**saved)
    retry = FakeTransport([AssistantTurn({"content": "Still truncated"}, "length")])
    result = _run_shared_rollout(retry, state=restored, start_turn_index=2)
    assert result.partial and result.reason == "max_tokens"
    assert result.submission == analogue
    assert result.state.assistant_turns == 3
    assert len(retry.requests) == 1
    assert sum(e["event_type"] == "output_limit_recovery" for e in result.state.events) == 1
    assert not RolloutPolicyState.from_checkpoint({}).output_limit_recovery_used


def test_shared_rollout_returns_partial_result_when_tool_execution_times_out() -> None:
    class TimeoutTools(FakeTools):
        async def execute(self, calls, *, target_smiles, assistant_turn):
            raise ControlledTermination("tool_timeout", "chemistry call timed out")

    transport = FakeTransport([tool_turn("similarity", {"smiles_a": "CC", "smiles_b": "CO"})])

    result = _run_shared_rollout(transport, TimeoutTools([]))

    assert result.partial is True
    assert result.reason == "tool_timeout"
    assert result.termination_message == "chemistry call timed out"


def test_shared_rollout_checkpoints_completed_non_tool_turn_before_next_request() -> None:
    transport = FakeTransport(
        [
            tool_turn("search_building_blocks", {"query": "CCO"}),
            AssistantTurn({"role": "assistant", "content": "Still planning."}),
            RuntimeError("request failed"),
        ]
    )
    checkpoint_turns: list[int] = []

    with pytest.raises(RuntimeError, match="request failed"):
        _run_shared_rollout(
            transport,
            FakeTools([{"valid": True, "results": []}]),
            event_callback=lambda _event, turn: (
                checkpoint_turns.append(turn) if turn is not None else None
            ),
        )

    assert checkpoint_turns[-1] == 2


def test_shared_rollout_requests_and_accepts_a_tool_limit_final_submission() -> None:
    transport = FakeTransport(
        [
            tool_turn("search_building_blocks", {"query": "CCO"}),
            tool_turn("submit_pathway", {"steps": [], "building_block": "BB:1"}),
        ]
    )
    tools = FakeTools(
        [
            {"valid": True, "results": []},
            {
                "accepted": True,
                "actual_final_smiles": "CC",
                "similarity_to_target": 0.4,
                "constitutional_similarity_to_target": 0.4,
            },
        ]
    )

    result = _run_shared_rollout(
        transport,
        tools,
        rollout_config=config(max_tool_calls=1),
    )

    assert result.reason == "tool_limit_final_submission"
    assert any(
        "tool-call budget has been reached" in str(message.get("content"))
        for message in transport.requests[1]
    )
    assert transport.tool_choices[1] == {
        "type": "function",
        "function": {"name": "submit_pathway"},
    }


def test_benchmark_empty_replies_still_obey_turn_budget() -> None:
    transport = FakeTransport([AssistantTurn({"content": ""}, "stop") for _ in range(3)])

    result = _run_shared_rollout(
        transport, rollout_config=config(stop_condition="exact", max_turns=3)
    )

    assert len(transport.requests) == result.state.assistant_turns == 3
    assert result.state.policy.total_tool_calls == 0
    assert result.submission is None
    assert result.reason == "no_accepted_submission"


def test_unlimited_turns_require_a_budget_provider():
    with pytest.raises(ValueError, match="budget_provider"):
        _run_shared_rollout(
            FakeTransport([]), rollout_config=config(max_turns=None, max_model_len=50000)
        )
