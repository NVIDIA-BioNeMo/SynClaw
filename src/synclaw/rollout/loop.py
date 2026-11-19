# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral execution of one BioNeMo-SynClaw tool-use rollout."""

from __future__ import annotations

import inspect
import itertools
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from synclaw.core.libraries import LibrarySelection
from synclaw.rollout.budget import (
    BudgetFeedbackState,
    BudgetProvider,
    ContextBudgetSnapshot,
    prepare_budget_feedback,
    with_harness_feedback,
)
from synclaw.rollout.policy import (
    DEFAULT_REACTION_LIBRARY,
    PreparedToolCall,
    RolloutPolicyConfig,
    RolloutPolicyState,
    handle_no_tool_response,
    handle_tool_result,
    prepare_tool_call,
    response_or_chat_tool_choice,
    selected_submission_event,
    tool_argument_error,
    tool_limit_finish_reason,
    turn_reminders,
)
from synclaw.tools.compaction import compact_json_size, compact_tool_result
from synclaw.tools.contracts import TOOL_ARGUMENT_DEFAULTS, TOOL_NAMES

EventCallback = Callable[[dict[str, Any], int | None], Awaitable[None] | None]
_TOKEN_LIMITS = frozenset({"length", "max_tokens", "max_output_tokens"})


@dataclass(frozen=True, slots=True)
class RolloutTask:
    task_id: str
    target_smiles: str
    initial_messages: tuple[dict[str, Any], ...]


# Finish reasons live with the loop that emits them, so a rename here cannot leave a
# consumer's vocabulary stale.
TURN_LIMIT_REASONS = frozenset({"best_accepted_submission", "no_accepted_submission"})
INTENTIONAL_REASONS = frozenset({"terminal_submission", "model_stop_best_accepted_submission"})
# The model chose to stop with no accepted route.
STOPPED_WITHOUT_ROUTE_REASONS = frozenset(
    {"model_stop_no_accepted_submission", "assistant_stopped_without_tools"}
)


def stop_category(reason: object) -> str:
    """Map a recorded rollout finish reason to a stable high-level category."""

    normalized = str(reason or "unknown")
    if normalized.startswith("tool_limit_"):
        return "tool_call_limit"
    if normalized in TURN_LIMIT_REASONS:
        return "turn_limit"
    if normalized in INTENTIONAL_REASONS or normalized.startswith("finish_token_"):
        return "intentional"
    if normalized in STOPPED_WITHOUT_ROUTE_REASONS:
        return "stopped_without_route"
    return "other"


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    message: dict[str, Any]
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """One executed tool call.

    Only `result` reaches the model; it is compacted into the tool message. The
    remaining fields are recorded on the trace event and are never shown.
    """

    result: dict[str, Any]
    event_metadata: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    outcome: str = "unknown"


class RolloutTransport(Protocol):
    async def complete(
        self, messages: Sequence[dict[str, Any]], *, tools: Sequence[dict[str, Any]], tool_choice: Any,
        max_output_tokens: int | None,
    ) -> AssistantTurn: ...
    def append_user(self, message: dict[str, Any]) -> None: ...
    def append_tool(self, call: PreparedToolCall, result_json: str) -> None: ...
    def replace_last_input(self, message: dict[str, Any]) -> None: ...


class ToolExecutor(Protocol):
    """Execute prepared calls, returning structured errors for malformed arguments."""

    async def execute(
        self,
        calls: Sequence[PreparedToolCall],
        *,
        target_smiles: str,
        assistant_turn: int,
    ) -> Sequence[ToolExecution]: ...


class LocalToolExecutor:
    """Execute normalized calls through BioNeMo-SynClaw's in-process synthesis tools."""

    def __init__(
        self,
        tools: Any,
        *,
        reaction_library: LibrarySelection = DEFAULT_REACTION_LIBRARY,
    ) -> None:
        self.tools = tools
        self.reaction_library = reaction_library

    async def execute(
        self,
        calls: Sequence[PreparedToolCall],
        *,
        target_smiles: str,
        assistant_turn: int,
    ) -> Sequence[ToolExecution]:
        del assistant_turn
        return [self._execute_one(call, target_smiles=target_smiles) for call in calls]

    def _execute_one(
        self, call: PreparedToolCall, *, target_smiles: str
    ) -> ToolExecution:
        started_at = time.perf_counter()
        if call.argument_error is not None:
            return ToolExecution(
                tool_argument_error(call.name, call.argument_error, call.arguments),
                duration_seconds=time.perf_counter() - started_at,
                outcome="argument_error",
            )
        return ToolExecution(
            call_local_tool(
                self.tools,
                call.name,
                call.dispatch_arguments,
                reaction_library=self.reaction_library,
                target_smiles=target_smiles,
            ),
            duration_seconds=time.perf_counter() - started_at,
            outcome="ok",
        )


def call_local_tool(
    tools: Any,
    name: str,
    arguments: dict[str, Any],
    *,
    reaction_library: LibrarySelection = DEFAULT_REACTION_LIBRARY,
    target_smiles: str | None = None,
) -> dict[str, Any]:
    """Validate and dispatch one call to an allowlisted in-process tool."""
    arguments = {**TOOL_ARGUMENT_DEFAULTS.get(name, {}), **arguments}
    try:
        if name not in TOOL_NAMES:
            return tool_argument_error(name, f"Unknown tool name: {name}", arguments)
        if name == "search_building_blocks":
            return tools.search_building_blocks(
                _required_str(arguments, "query", name),
                arguments["limit"],
                include_near=arguments["include_near"],
                min_similarity=arguments["min_similarity"],
            )
        if name == "similarity":
            return tools.similarity(
                _required_str(arguments, "smiles_a", name),
                _required_str(arguments, "smiles_b", name),
            )
        if name == "mces":
            return tools.mces(
                _required_str(arguments, "smiles_a", name),
                _required_str(arguments, "smiles_b", name),
            )
        if name == "search_reaction_templates":
            return tools.search_reaction_templates(
                query=_optional_str(arguments, "query"),
                reactant_smiles=_optional_str_list(arguments, "reactant_smiles", name),
                product_smiles=_optional_str_or_str_list(
                    arguments, "product_smiles", name
                ),
                limit=arguments["limit"],
                library=reaction_library or None,
                run_forward=arguments["run_forward"],
            )
        if name == "simulate_reaction":
            return tools.simulate_reaction(
                _required_str(arguments, "reaction_id", name),
                _required_str_list(arguments, "reactants", name),
                product_smiles=_optional_str(arguments, "product_smiles"),
                max_products=arguments["max_products"],
            )
        if name == "simulate_reverse_reaction":
            return tools.simulate_reverse_reaction(
                _required_str(arguments, "reaction_id", name),
                _required_str_or_str_list(arguments, "product_smiles", name),
                max_outcomes=arguments["max_outcomes"],
                include_near=arguments["include_near"],
                min_similarity=arguments["min_similarity"],
                top_near_matches=arguments["top_near_matches"],
            )
        if name == "submit_pathway":
            steps = _optional_array(arguments, "steps", name) or []
            if not all(isinstance(step, dict) for step in steps):
                raise ValueError(f"{name} argument 'steps' must be an array of objects")
            return tools.submit_pathway(
                steps=steps,
                building_block=_optional_str(arguments, "building_block"),
                pathway_text=arguments.get("pathway_text"),
                target_smiles=target_smiles,
            )
        raise AssertionError(f"Missing local dispatcher for allowlisted tool: {name}")
    # Malformed model arguments surface as these; anything else is a harness bug and must propagate.
    except (TypeError, ValueError) as exc:
        return tool_argument_error(name, str(exc), arguments)


def _required_str(arguments: dict[str, Any], key: str, tool_name: str) -> str:
    value = arguments.get(key)
    if value in (None, ""):
        raise ValueError(f"{tool_name} is missing required argument '{key}'")
    return str(value)


def _optional_str(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _required_array(
    arguments: dict[str, Any], key: str, tool_name: str
) -> list[Any]:
    value = arguments.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{tool_name} argument '{key}' must be a JSON array")
    return value


def _optional_array(
    arguments: dict[str, Any], key: str, tool_name: str
) -> list[Any] | None:
    value = arguments.get(key)
    if value in (None, ""):
        return None
    if not isinstance(value, list):
        raise ValueError(f"{tool_name} argument '{key}' must be a JSON array")
    return value


def _required_str_list(
    arguments: dict[str, Any], key: str, tool_name: str
) -> list[str]:
    return [str(item) for item in _required_array(arguments, key, tool_name)]


def _optional_str_list(
    arguments: dict[str, Any], key: str, tool_name: str
) -> list[str] | None:
    value = arguments.get(key)
    if value in (None, ""):
        return None
    if not isinstance(value, list):
        raise ValueError(f"{tool_name} argument '{key}' must be a JSON array")
    return [str(item) for item in value]


def _required_str_or_str_list(
    arguments: dict[str, Any], key: str, tool_name: str
) -> str | list[str]:
    value = arguments.get(key)
    if value in (None, ""):
        raise ValueError(f"{tool_name} is missing required argument '{key}'")
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{tool_name} argument '{key}' must not be empty")
        return [str(item) for item in value]
    return str(value)


def _optional_str_or_str_list(
    arguments: dict[str, Any], key: str, tool_name: str
) -> str | list[str] | None:
    value = arguments.get(key)
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return value
    raise ValueError(
        f"{tool_name} argument '{key}' must be a string or JSON array of strings"
    )


class ControlledTermination(RuntimeError):
    def __init__(self, kind: str, message: str, *, provider_finish_reason: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.provider_finish_reason = provider_finish_reason


def request_output_budget(
    snapshot: ContextBudgetSnapshot, *, safety_margin_tokens: int,
) -> ContextBudgetSnapshot:
    """Clip one request's output cap to the context its input leaves, less a safety margin.

    The input is an estimate, so the margin absorbs estimation error. An unknown
    context limit or input leaves the per-response cap unchanged. When no output
    fits, the rollout ends before the request rather than truncating history.
    """

    remaining = snapshot.remaining_tokens
    if remaining is None:
        if snapshot.max_output_tokens is None and snapshot.context_window_tokens is not None:
            # With no cap of its own and an unknown input, only the whole context bounds the output.
            return replace(
                snapshot, max_output_tokens=snapshot.context_window_tokens - safety_margin_tokens
            )
        return snapshot
    cap = snapshot.max_output_tokens
    allowed = remaining - safety_margin_tokens if cap is None else min(cap, remaining - safety_margin_tokens)
    if allowed <= 0:
        raise ControlledTermination(
            "context_window_exceeded",
            f"About {snapshot.input_tokens} input tokens leave no output within the "
            f"{snapshot.context_window_tokens}-token context limit after a "
            f"{safety_margin_tokens}-token safety margin.",
        )
    return replace(snapshot, max_output_tokens=allowed)


@dataclass(slots=True)
class RolloutState:
    messages: list[dict[str, Any]]
    events: list[dict[str, Any]]
    policy: RolloutPolicyState = field(default_factory=RolloutPolicyState)
    assistant_turns: int = 0
    budget_feedback: BudgetFeedbackState = field(default_factory=BudgetFeedbackState)


@dataclass(frozen=True, slots=True)
class RolloutResult:
    state: RolloutState
    reason: str
    submission: dict[str, Any] | None
    submission_arguments: dict[str, Any] | None
    partial: bool = False
    termination_kind: str | None = None
    termination_message: str | None = None


async def run_rollout(
    *,
    task: RolloutTask,
    config: RolloutPolicyConfig,
    transport: RolloutTransport,
    tool_executor: ToolExecutor,
    tool_schemas: Sequence[dict[str, Any]],
    state: RolloutState | None = None,
    start_turn_index: int = 0,
    event_callback: EventCallback | None = None,
    budget_provider: BudgetProvider | None = None,
) -> RolloutResult:
    """Run the canonical BioNeMo-SynClaw loop independently of model and tool transports."""
    if config.max_turns is None and budget_provider is None:
        raise ValueError("Unlimited turns need a budget_provider to end the rollout")
    state = state or RolloutState(
        messages=[dict(row) for row in task.initial_messages], events=[]
    )

    async def record(event: dict[str, Any] | None, turn: int | None = None) -> None:
        if event is not None:
            state.events.append(event)
        if event_callback is not None:
            outcome = event_callback(event or {}, turn)
            if inspect.isawaitable(outcome):
                await outcome

    if not state.events:
        for message in state.messages:
            await record(_message_event(message, task.target_smiles), 0)

    reason, partial, termination_kind, termination_message = (
        "no_submission", False, None, None
    )
    termination: ControlledTermination | None = None
    turn_indices = (
        itertools.count(start_turn_index)
        if config.max_turns is None
        else range(start_turn_index, config.max_turns)
    )
    for turn_index in turn_indices:
        for reminder in turn_reminders(config, state.policy, turn_index=turn_index):
            await _append_harness_feedback(state, transport, task, reminder, record)
        if config.token_budget_feedback and budget_provider is not None:
            await _emit_budget_feedback(
                state, transport, task, config, tool_schemas, budget_provider, turn_index, record,
            )
        tool_choice = response_or_chat_tool_choice(
            api_mode="chat",
            forced_submit_requested=state.policy.forced_submit_requested,
            has_submission=state.policy.has_submission,
            tools=list(tool_schemas),
        )
        # Clip and record each request's output cap before sending it, so a failed request keeps its budget.
        snapshot = (
            budget_provider(tuple(state.messages), tool_schemas)
            if budget_provider is not None
            else ContextBudgetSnapshot(config.max_model_len, None, config.max_tokens)
        )
        try:
            try:
                request_budget = request_output_budget(
                    snapshot, safety_margin_tokens=config.context_safety_margin_tokens,
                )
            except ControlledTermination:
                await record(_request_budget_event(task, turn_index, snapshot, config, sent=False), turn_index)
                raise
            await record(_request_budget_event(task, turn_index, request_budget, config), turn_index)
            turn = await transport.complete(
                tuple(state.messages), tools=tool_schemas, tool_choice=tool_choice,
                max_output_tokens=request_budget.max_output_tokens,
            )
        except ControlledTermination as exc:
            termination = exc
            break

        message = {**turn.message, "role": "assistant"}
        state.messages.append(message)
        state.assistant_turns += 1
        await record(_message_event(message, task.target_smiles), turn_index)
        finish_reason = str(turn.finish_reason or "").strip().lower() or None
        if finish_reason in _TOKEN_LIMITS:
            # Recovery is a harness message about the output limit, so it is progress feedback too.
            if (
                config.token_budget_feedback
                and not state.policy.output_limit_recovery_used
                and not state.policy.tool_limit_finalization_requested
                and (config.max_turns is None or turn_index + 1 < config.max_turns)
                and await _recover_output_limit(state, transport, task, finish_reason, record)
            ):
                await record(None, turn_index + 1)
                continue
            cap = request_budget.max_output_tokens
            if cap is not None and (config.max_tokens is None or cap < config.max_tokens):
                # The cap had been clipped to the context left, so the context ran out.
                partial, termination_kind, reason = True, "context_window_exceeded", "context_window_exceeded"
                termination_message = (
                    f"The response reached the {cap}-token output cap that the remaining context allowed."
                )
            else:
                partial, termination_kind, reason = True, "max_tokens", "max_tokens"
                termination_message = f"The provider stopped generation with finish_reason={finish_reason!r}."
            await record(
                _termination_event(task, turn_index, reason, termination_message, finish_reason),
                turn_index,
            )
            break

        raw_calls = list(message.get("tool_calls") or [])
        if not raw_calls:
            decision = handle_no_tool_response(config, state.policy, content=str(message.get("content") or ""))
            if decision.finish_reason is not None:
                reason = decision.finish_reason
                break
            if decision.reminder is not None:
                await _append_harness_feedback(state, transport, task, decision.reminder, record)
                await record(None, turn_index + 1)
                continue
            reason = "assistant_stopped_without_tools"
            break

        prepared = [_prepare_call(call, len(state.events) + index) for index, call in enumerate(raw_calls)]
        # Trace events retain the raw call, but providers only see canonical JSON on later turns.
        # This keeps a recoverable argument error from poisoning the conversation history.
        message["tool_calls"] = [_canonical_tool_call(item) for item in prepared]
        for item in prepared:
            await record(_tool_event("tool_call", task, item.name, item.event_payload), turn_index)
        try:
            executions = await _execute_tools(
                prepared,
                tool_executor,
                config,
                call_limit=(
                    config.max_tool_calls - state.policy.total_tool_calls
                    if config.max_tool_calls > 0 else None
                ),
                target_smiles=task.target_smiles,
                assistant_turn=turn_index + 1,
            )
        except ControlledTermination as exc:
            termination = exc
            break
        repairs: list[str] = []
        terminal = None
        terminal_due_to_limit = False
        terminal_meets_floor = False
        for item, execution in zip(prepared, executions, strict=True):
            result = execution.result
            model_result = compact_tool_result(item.name, result)
            result_json = json.dumps(model_result, sort_keys=True, separators=(",", ":"))
            state.messages.append({"role": "tool", "tool_call_id": item.call_id, "content": result_json})
            transport.append_tool(item, result_json)
            payload = {
                "result": result,
                "model_result": model_result,
                "model_result_compaction": {
                    "full_json_bytes": compact_json_size(result),
                    "model_json_bytes": compact_json_size(model_result),
                },
                "duration_seconds": round(execution.duration_seconds, 6),
                "outcome": execution.outcome,
                **execution.event_metadata,
            }
            await record(_tool_event("tool_result", task, item.name, payload), turn_index)
            if execution.outcome == "over_budget":
                continue
            decision = handle_tool_result(
                config, state.policy, name=item.name, arguments=item.dispatch_arguments,
                result=result, target_smiles=task.target_smiles,
            )
            if decision.repair is not None:
                repairs.append(decision.repair)
            # Record all already-executed calls, but the first submission that meets
            # the floor ends the search even if a later result in this batch reaches
            # a budget limit. The policy state keeps the best route on its own.
            if not terminal_meets_floor:
                if decision.terminal_due_to_tool_limit:
                    terminal_due_to_limit = True
                if decision.terminal_submission is not None:
                    terminal = decision.terminal_submission
                    terminal_meets_floor = decision.meets_floor
        if terminal is None and (
            terminal_due_to_limit or state.policy.tool_limit_finalization_requested
        ):
            reason = tool_limit_finish_reason(state.policy)
            break
        if terminal is not None:
            reason = "tool_limit_final_submission" if not terminal_meets_floor and (
                state.policy.tool_limit_finalization_requested or terminal_due_to_limit
            ) else "terminal_submission"
            break
        for repair in repairs:
            await _append_harness_feedback(state, transport, task, repair, record)
        await record(None, turn_index + 1)
    else:
        reason = (
            "best_accepted_submission"
            if state.policy.best_accepted_submission
            else "no_accepted_submission"
        )
    if termination is not None:
        partial, termination_kind, termination_message, reason = (
            True, termination.kind, str(termination), termination.kind
        )
        await record(
            _termination_event(
                task, turn_index, termination.kind, str(termination), termination.provider_finish_reason,
            ),
            turn_index,
        )

    submission, arguments = state.policy.selected_submission()
    selected_event = selected_submission_event(
        task={"task_id": task.task_id, "target_smiles": task.target_smiles},
        submission=submission,
        submission_arguments=arguments,
        reason=reason,
    )
    if selected_event is not None:
        await record(selected_event, config.max_turns)
    return RolloutResult(
        state, reason, submission, arguments, partial,
        termination_kind, termination_message,
    )


async def _recover_output_limit(
    state: RolloutState, transport: RolloutTransport, task: RolloutTask,
    finish_reason: str, record: Callable[..., Awaitable[None]],
) -> bool:
    """Resolve unexecuted calls and request one shorter continuation, without extra budget."""
    message = state.messages[-1]
    raw_calls = list(message.get("tool_calls") or [])
    if any(not call.get("id") or not (call.get("function") or {}).get("name") for call in raw_calls):
        return False  # Cannot safely pair a partial call without its provider ID/name.
    content = (
        "Your previous response hit its output-token limit and was truncated. "
        "None of the tool calls from that response were executed. "
        "Continue from the work already shown with a shorter response; reissue any needed "
        "tool call with complete arguments. Submit your best verified route, even if it is "
        "an analogue, as soon as possible so it can be scored. This is your only output-limit "
        "recovery attempt."
    )
    state.policy.output_limit_recovery_used = True
    if raw_calls:
        prepared = [_prepare_call(call, len(state.events) + i) for i, call in enumerate(raw_calls)]
        # As with ordinary argument repair, replay valid JSON while retaining raw trace data.
        message["tool_calls"] = [_canonical_tool_call(call) for call in prepared]
        for raw, call in zip(raw_calls, prepared, strict=True):
            result_json = json.dumps({
                "error": "output_limit_truncation", "executed": False,
                "message": "Not executed: the assistant response was truncated. Reissue the complete call.",
                # Canonical replay replaces malformed JSON; retain its useful prefix as data.
                **({"truncated_arguments": raw["function"].get("arguments")}
                   if call.argument_error else {}),
            })
            state.messages.append({"role": "tool", "tool_call_id": call.call_id, "content": result_json})
            transport.append_tool(call, result_json)
        # These are protocol error replies, not executed chemistry calls or scored submissions.
    else:
        message.pop("tool_calls", None)
        # A new user turn can make chat templates discard earlier reasoning_content.
        # Keep exposed working notes as ordinary assistant text for replay. Native
        # Responses/Anthropic histories retain their own opaque/signed reasoning blocks.
        for index, previous in enumerate(state.messages):
            reasoning = previous.get("reasoning_content")
            if previous["role"] == "assistant" and reasoning:
                state.messages[index] = {
                    **previous, "reasoning_content": "",
                    "content": f"[Prior reasoning]\n{reasoning}\n[/Prior reasoning]\n\n"
                    + (previous.get("content") or ""),
                }
    await _append_harness_feedback(state, transport, task, content, record)
    await record({
        "event_type": "output_limit_recovery", "target_smiles": task.target_smiles,
        "payload": {"assistant_turn": state.assistant_turns, "provider_finish_reason": finish_reason,
                    "discarded_tool_calls": len(raw_calls)},
    })
    return True


async def _execute_tools(
    calls: list[PreparedToolCall], executor: ToolExecutor, config: RolloutPolicyConfig,
    *, call_limit: int | None, target_smiles: str, assistant_turn: int,
) -> list[ToolExecution]:
    results: list[ToolExecution | None] = [None] * len(calls)
    pending: list[tuple[int, PreparedToolCall]] = []
    # Calls in one response share the budget, so a batch cannot exceed it. Once the
    # budget is spent, the final turn may still make one submission.
    if call_limit is None:
        allowed = range(len(calls))
    elif call_limit > 0:
        allowed = range(min(call_limit, len(calls)))
    else:
        allowed = [index for index, item in enumerate(calls) if item.name == "submit_pathway"][:1]
    for index, item in enumerate(calls):
        over_budget = index not in allowed
        error, outcome = (
            ("The tool-call budget was reached; this call was not run.", "over_budget")
            if over_budget
            else (f"Tool is excluded from this run: {item.name}", "excluded_tool")
        )
        if not over_budget and item.name not in config.excluded_tools:
            pending.append((index, item))
        else:
            arguments = item.arguments if item.argument_error else item.dispatch_arguments
            results[index] = ToolExecution(tool_argument_error(item.name, error, arguments), outcome=outcome)
    if pending:
        executed = list(
            await executor.execute(
                [item for _, item in pending],
                target_smiles=target_smiles,
                assistant_turn=assistant_turn,
            )
        )
        if len(executed) != len(pending):
            raise RuntimeError(f"Tool executor returned {len(executed)} results for {len(pending)} calls")
        for (index, _), result in zip(pending, executed, strict=True):
            results[index] = result
    return [result for result in results if result is not None]


def _commit_input_update(
    state: RolloutState, transport: RolloutTransport, message: dict[str, Any],
) -> None:
    transport.replace_last_input(message)
    state.messages[-1] = message


async def _append_harness_feedback(state: RolloutState, transport: RolloutTransport, task: RolloutTask,
                       content: str, record: Callable[..., Awaitable[None]]) -> None:
    # Policy warnings and repair guidance also belong to the current tool turn.
    # Preserve that boundary for models which clear thinking on a new user turn.
    if state.messages and state.messages[-1]["role"] == "tool":
        updated = with_harness_feedback(state.messages, content, field="notice")
        assert updated is not None
        _commit_input_update(state, transport, updated[-1])
        await record({
            "event_type": "harness_feedback", "target_smiles": task.target_smiles,
            "payload": {"content": content, "message_index": len(state.messages) - 1,
                        "message_role": "tool"},
        })
        return
    message = {"role": "user", "content": content}
    state.messages.append(message)
    transport.append_user(message)
    await record(_message_event(message, task.target_smiles))


async def _emit_budget_feedback(
    state: RolloutState,
    transport: RolloutTransport,
    task: RolloutTask,
    config: RolloutPolicyConfig,
    tool_schemas: Sequence[dict[str, Any]],
    budget_provider: BudgetProvider,
    turn_index: int,
    record: Callable[..., Awaitable[None]],
) -> None:
    feedback = state.budget_feedback
    turns_remaining = (
        1 if state.policy.tool_limit_finalization_requested
        else None if config.max_turns is None
        else config.max_turns - turn_index
    )
    if not config.context_limit_stated:
        # The context stays enforced, but feedback must not reveal a limit the prompt leaves out.
        provider = budget_provider

        def budget_provider(messages: Any, tools: Any) -> ContextBudgetSnapshot:
            return replace(provider(messages, tools), context_window_tokens=None)
    tool_calls_remaining = (
        max(0, config.max_tool_calls - state.policy.total_tool_calls)
        if config.max_tool_calls > 0 else None
    )
    content, snapshot, status = prepare_budget_feedback(
        budget_provider, state.messages, tool_schemas, feedback, turn_index=turn_index,
        turns_remaining=turns_remaining, tool_calls_remaining=tool_calls_remaining,
        context_warning_remaining_fraction=config.context_warning_remaining_fraction,
        context_critical_remaining_fraction=config.context_critical_remaining_fraction,
    )
    if snapshot is None:
        return
    feedback.last_turn = turn_index
    if content:
        feedback.announced_context_limit = snapshot.context_window_tokens
        feedback.announced_output_limit = snapshot.max_output_tokens
        updated = with_harness_feedback(state.messages, content)
        assert updated is not None
        if len(updated) > len(state.messages):
            await _append_harness_feedback(state, transport, task, content, record)
        else:
            _commit_input_update(state, transport, updated[-1])
        remaining_tokens = snapshot.remaining_tokens
        if (
            remaining_tokens is not None
            and snapshot.context_window_tokens is not None
            and remaining_tokens <= snapshot.context_window_tokens * config.context_warning_remaining_fraction
        ):
            feedback.context_warning_sent = True
    await record({
        "event_type": "token_budget", "target_smiles": task.target_smiles,
        "payload": {
            **snapshot.trace_fields(), "turn_index": turn_index, "status": status,
            "turns_remaining": turns_remaining, "tool_calls_remaining": tool_calls_remaining,
            "content": content,
            "message_index": len(state.messages) - 1 if content else None,
            "message_role": state.messages[-1]["role"] if content else None,
        },
    })


def _prepare_call(raw: dict[str, Any], index: int) -> PreparedToolCall:
    function = raw.get("function") or {}
    return prepare_tool_call(
        raw_name=function.get("name", raw.get("name")),
        raw_arguments=function.get("arguments", raw.get("arguments")),
        raw_tool_call=raw, event_index=index,
    )


def _canonical_tool_call(call: PreparedToolCall) -> dict[str, Any]:
    return {
        "id": call.call_id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(
                call.dispatch_arguments,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    }


def _message_event(message: dict[str, Any], target: str) -> dict[str, Any]:
    return {"event_type": "message", "role": message["role"], "target_smiles": target, "payload": dict(message)}


def _tool_event(kind: str, task: RolloutTask, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"event_type": kind, "tool_name": name, "target_smiles": task.target_smiles, "payload": payload}


def _request_budget_event(
    task: RolloutTask, turn: int, budget: ContextBudgetSnapshot, config: RolloutPolicyConfig,
    *, sent: bool = True,
) -> dict[str, Any]:
    payload = {
        **budget.trace_fields(), "turn_index": turn, "configured_output_tokens": config.max_tokens, "sent": sent,
    }
    if not sent:
        # No output fit, so the rollout ended before this request.
        payload["request_output_tokens"] = 0
    return {"event_type": "request_budget", "target_smiles": task.target_smiles, "payload": payload}


def _termination_event(task: RolloutTask, turn: int, kind: str, message: str,
                       provider_reason: str | None) -> dict[str, Any]:
    return {"event_type": "controlled_termination", "target_smiles": task.target_smiles,
            "payload": {"kind": kind, "message": message, "assistant_turn": turn + 1,
                        "provider_finish_reason": provider_reason}}


__all__ = [
    "AssistantTurn", "ControlledTermination", "LocalToolExecutor",
    "RolloutResult", "RolloutState", "RolloutTask", "RolloutTransport", "ToolExecution",
    "ToolExecutor", "call_local_tool", "request_output_budget", "run_rollout",
    "INTENTIONAL_REASONS", "STOPPED_WITHOUT_ROUTE_REASONS", "TURN_LIMIT_REASONS", "stop_category",
]
