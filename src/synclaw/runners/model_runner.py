# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one tool-use synthesis-planning task against a hosted model."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from synclaw import providers, token_usage
from synclaw.core import db, toxicity
from synclaw.rollout import policy, prompts
from synclaw.rollout.budget import (
    BudgetFeedbackState,
    ContextBudgetSnapshot,
    ContextTokenEstimator,
    budget_summary_fields,
    initialize_budget_feedback,
)
from synclaw.rollout.loop import (
    AssistantTurn,
    LocalToolExecutor,
    RolloutState,
    RolloutTask,
    run_rollout,
)
from synclaw.tasks.datasets import build_adhoc_task
from synclaw.tools import contracts
from synclaw.tools.synthesis import (
    SynthesisTools,
    hidden_target_event,
    hidden_target_summary_fields,
)

DEFAULT_MODEL_RUN_DB = Path("var/synclaw.sqlite")
DEFAULT_TASKSET = "artifacts/tasksets/chembl_llm_tasks.jsonl"
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
ANTHROPIC_VERSION = "2023-06-01"


def attach_response_usage(message: dict[str, Any], response: dict[str, Any]) -> None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return
    message["response_usage"] = usage
    normalized = token_usage.normalize_token_usage(usage)
    if normalized:
        message["token_usage"] = normalized
        max_output_tokens = token_usage.int_or_none(
            response.get("max_output_tokens", response.get("max_tokens"))
        )
        output_tokens = normalized.get("output_tokens")
        if max_output_tokens is not None:
            message["max_output_tokens"] = max_output_tokens
            message["hit_max_output_tokens"] = (
                isinstance(output_tokens, int) and output_tokens >= max_output_tokens
            )


def retry_delay_seconds(attempt: int, body: str = "", headers: Any = None) -> float:
    retry_after = None
    if headers is not None:
        try:
            retry_after = headers.get("Retry-After")
        except AttributeError:
            retry_after = None
    parsed_retry_after = token_usage.float_or_none(retry_after)
    if parsed_retry_after is None:
        match = re.search(r"wait\s+(\d+(?:\.\d+)?)\s+seconds?", body, flags=re.IGNORECASE)
        if match:
            parsed_retry_after = token_usage.float_or_none(match.group(1))
    exponential = 10.0 * (2**attempt)
    return min(max(exponential, parsed_retry_after or 0.0), 120.0)


def _hide_target(args: argparse.Namespace) -> bool:
    return (
        bool(getattr(args, "hide_target_building_block", False))
        or getattr(args, "hide_building_block_similarity", None) is not None
    )


def _hidden_ids(args: argparse.Namespace) -> frozenset[int]:
    return frozenset(getattr(args, "hide_building_block_id", None) or [])


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=providers.provider_names(),
        default=providers.default_provider(),
        help="Provider preset for API transport and endpoint defaults",
    )
    parser.add_argument("--taskset", help=f"Task set JSONL. Defaults to {DEFAULT_TASKSET}.")
    parser.add_argument(
        "--task-index", type=int, help="Zero-based task to run from the task set. Defaults to 0."
    )
    parser.add_argument(
        "--smiles",
        help="Plan this one target SMILES instead of reading a task set "
        "(cannot be combined with --taskset or --task-index)",
    )
    parser.add_argument("--db", help="SQLite database. Defaults to var/synclaw.sqlite.")
    parser.add_argument("--model", required=True, help="Model ID as the provider names it")
    parser.add_argument(
        "--api-mode",
        choices=("chat", "responses", "messages"),
        help="Override the provider preset's API: Chat Completions, Responses, or Messages",
    )
    parser.add_argument("--base-url", help="Chat-completions API base URL")
    parser.add_argument(
        "--responses-url",
        help="Responses API URL",
    )
    parser.add_argument(
        "--messages-url",
        help="Messages API URL",
    )
    parser.add_argument(
        "--responses-state-mode",
        choices=("previous_id", "full"),
        help=(
            "Use previous_response_id continuations or resend full Responses input history; "
            "defaults to the provider preset"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Environment variable containing the API key; defaults to the provider preset's variables",
    )
    parser.add_argument(
        "--auth-header",
        choices=providers.AUTH_HEADERS,
        help=(
            "How the API key is sent: an Authorization bearer token or an x-api-key header. "
            "Defaults to x-api-key for the Messages API and bearer otherwise."
        ),
    )
    parser.add_argument("--trace", default="artifacts/traces/model_run.jsonl")
    parser.add_argument(
        "--checkpoint",
        help=(
            "Ephemeral checkpoint JSON path. The runner updates it after completed turns and "
            "removes it on success."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        help="Resume from a checkpoint JSON file. Defaults to --checkpoint when that file already exists.",
    )
    parser.add_argument(
        "--summary-output", help="Optional one-row evaluated summary CSV to write after the run"
    )
    parser.add_argument(
        "--prompt-version", help="Optional prompt or harness label stored in --summary-output"
    )
    parser.add_argument("--max-turns", type=int, default=policy.DEFAULT_MAX_TURNS)
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=policy.DEFAULT_MAX_TOOL_CALLS,
        help=(
            "Soft total tool-call budget; when reached, request one final submit_pathway turn "
            "and then fall back to the best accepted route if needed; set 0 to disable"
        ),
    )
    parser.add_argument(
        "--turn-warning-remaining",
        type=int,
        default=policy.DEFAULT_TURN_WARNING_REMAINING,
        help="Warn the model when this many assistant turns remain; set 0 to disable",
    )
    parser.add_argument(
        "--force-submit-after-non-tool-turns",
        type=int,
        default=policy.DEFAULT_FORCE_SUBMIT_AFTER_NON_TOOL_TURNS,
        help="Request final submission after this many assistant turns without tool calls after prior tool use; set 0 to disable",
    )
    parser.add_argument(
        "--same-pathway-finish-after",
        type=int,
        default=policy.DEFAULT_SAME_PATHWAY_FINISH_AFTER,
        help=(
            "After this many consecutive accepted submit_pathway calls with the same route, "
            "tell the model it may finish with <FINISH>; set 0 to disable"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_token_limit,
        default=policy.DEFAULT_MAX_TOKENS,
        help="Output cap per response; a task's max_tokens overrides it",
    )
    parser.add_argument(
        "--max-model-len",
        type=_positive_token_limit,
        help=(
            "Context limit; each response's output is clipped to the context left. A task's "
            "max_model_len may lower it. Omitted when unknown, which disables clipping"
        ),
    )
    policy.add_rollout_flag_arguments(parser)
    parser.add_argument(
        "--context-warning-remaining-fraction",
        type=float,
        default=policy.DEFAULT_CONTEXT_WARNING_REMAINING_FRACTION,
        help="Warn once when estimated remaining context falls to this fraction of the window",
    )
    parser.add_argument(
        "--context-critical-remaining-fraction",
        type=float,
        default=policy.DEFAULT_CONTEXT_CRITICAL_REMAINING_FRACTION,
        help="Push for immediate submission when estimated remaining context falls to this fraction",
    )
    parser.add_argument(
        "--token-param",
        choices=("auto", "max_tokens", "max_completion_tokens"),
        default="auto",
        help="Chat completions token budget parameter name",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        help=(
            "Optional reasoning effort hint. max is sent as-is to the Messages API and "
            "as xhigh to Chat Completions and Responses endpoints."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional temperature; omitted by default to use provider default",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Optional nucleus-sampling probability; omitted by default to use provider default",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        help="Optional model sampling seed for reproducible requests when supported",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-chat-completion request timeout in seconds",
    )
    parser.add_argument(
        "--max-low-similarity-submissions",
        type=int,
        default=policy.DEFAULT_MAX_LOW_SIMILARITY_SUBMISSIONS,
        help=(
            "Optional cap on accepted submissions below the --stop-condition similarity before ending. "
            "Omit to keep retrying until the turn/tool budget is exhausted."
        ),
    )
    toxicity.add_toxicity_argument(parser)
    parser.add_argument(
        "--hide-target-building-block",
        action="store_true",
        help=(
            "Treat the target as out of stock, so it cannot be bought directly or used as a "
            "starting reactant and an exact match needs a real synthesis route. Building blocks "
            "that Standard InChI identity counts as the target are hidden too; analogues can "
            "still be bought."
        ),
    )
    parser.add_argument(
        "--hide-building-block-id",
        type=int,
        action="append",
        default=[],
        help="Hide this catalog BB ID for every input; repeat to exclude multiple IDs.",
    )
    parser.add_argument(
        "--hide-building-block-similarity",
        type=float,
        help="Hide stock with constitutional Morgan/Tanimoto similarity >= this cutoff [0, 1].",
    )
    parser.add_argument(
        "--reaction-library",
        nargs="+",
        default=policy.DEFAULT_REACTION_LIBRARY,
        help="Reaction-template libraries the tools may use; others are excluded from search and overview",
    )
    parser.add_argument(
        "--exclude-tool",
        action="append",
        default=[],
        help=(
            "Tool name to hide from the model and block at dispatch. May be repeated or comma-separated. "
            "Added to any excluded_tools the task lists. submit_pathway cannot be excluded."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = run_model_from_args(args)
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_model_from_args(args: argparse.Namespace) -> dict[str, Any]:
    session_started_at = time.time()
    session_started_monotonic = time.monotonic()
    apply_provider_defaults(args)
    require_endpoint(args)
    # Resolved before the key check so a bad target or task set is reported on the first try.
    task = resolve_run_task(args)
    use_preset_keys = uses_preset_endpoint(args)
    api_key = resolve_api_key(args.provider, args.api_key_env, use_preset_keys=use_preset_keys)
    if not api_key and providers.api_key_required(
        args.provider, args.api_key_env, use_preset_keys=use_preset_keys
    ):
        raise RuntimeError(providers.missing_api_key_message(args.provider, args.api_key_env))

    args.db = str(resolve_task_db_path(getattr(args, "db", None)))
    prompts.validate_model_run_database(args.db, library=args.reaction_library or None)
    excluded_tools = run_excluded_tools(args, task)
    chat_tool_schemas = contracts.filter_chat_tool_schemas(excluded_tools)
    responses_tool_schemas = contracts.filter_responses_tool_schemas(excluded_tools)
    messages_tool_schemas = contracts.filter_messages_tool_schemas(excluded_tools)
    tools = SynthesisTools(
        args.db,
        reaction_libraries=args.reaction_library,
        toxicity=toxicity.config_from_args(args),
        hidden_target_smiles=task["target_smiles"] if _hide_target(args) else None,
        hidden_building_block_ids=_hidden_ids(args),
        hidden_similarity=getattr(args, "hide_building_block_similarity", None),
    )
    # Refuse a flagged target before any tool call or model request is spent on it.
    target_screen_event = tools.screen.screen_target(task["target_smiles"])
    if target_screen_event and target_screen_event["event_type"] == "target_refused":
        return refuse_target(args, task, target_screen_event, tools.screen.config.mode)
    hidden_ids = tools.hidden_building_block_ids  # Resolve once up front, including on resume.
    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if getattr(args, "checkpoint", None) else None
    resume_checkpoint_path = (
        Path(args.resume_checkpoint)
        if getattr(args, "resume_checkpoint", None)
        else checkpoint_path
        if checkpoint_path is not None and checkpoint_path.exists()
        else None
    )

    budget_checkpoint: dict[str, Any] = {}
    policy_config = policy.RolloutPolicyConfig(
        max_turns=args.max_turns,
        max_tool_calls=int(args.max_tool_calls or 0),
        max_tokens=args.max_tokens,
        max_model_len=getattr(args, "max_model_len", None),
        context_safety_margin_fraction=getattr(
            args, "context_safety_margin_fraction", policy.DEFAULT_CONTEXT_SAFETY_MARGIN_FRACTION
        ),
        budget_feedback=getattr(args, "budget_feedback", True),
        stop_condition=getattr(args, "stop_condition", policy.DEFAULT_STOP_CONDITION),
        turn_warning_remaining=args.turn_warning_remaining,
        force_submit_after_non_tool_turns=args.force_submit_after_non_tool_turns,
        same_pathway_finish_after=int(args.same_pathway_finish_after or 0),
        max_low_similarity_submissions=args.max_low_similarity_submissions,
        excluded_tools=frozenset(excluded_tools),
        context_warning_remaining_fraction=args.context_warning_remaining_fraction,
        context_critical_remaining_fraction=args.context_critical_remaining_fraction,
    )
    try:
        policy_config = policy.task_budget_config(task, policy_config)
    except ValueError as exc:
        raise RuntimeError(f"Task {task['task_id']!r}: {exc}") from exc
    if resume_checkpoint_path is not None and resume_checkpoint_path.exists():
        checkpoint = load_checkpoint(resume_checkpoint_path)
        budget_checkpoint = checkpoint
        budget_feedback = BudgetFeedbackState.from_checkpoint(
            checkpoint.get("budget_feedback_state")
        )
        if not checkpoint_matches_run(checkpoint, args, task, policy_config):
            raise RuntimeError(
                f"Checkpoint {resume_checkpoint_path} does not match this model/provider/task run"
            )
        checkpoint_path = resume_checkpoint_path
        checkpoint_trace_path = Path(str(checkpoint.get("trace_path") or trace_path))
        if not checkpoint_trace_path.exists():
            raise RuntimeError(
                f"Checkpoint {resume_checkpoint_path} expects trace {checkpoint_trace_path}, which does not exist"
            )
        trace_events = list(db.iter_jsonl(checkpoint_trace_path))
        messages = checkpoint.get("messages") or []
        responses_pending_input = checkpoint.get("responses_pending_input") or []
        responses_input_history = checkpoint.get("responses_input_history") or []
        anthropic_system = normalize_anthropic_system(checkpoint.get("anthropic_system"))
        anthropic_messages = checkpoint.get("anthropic_messages") or []
        anthropic_tool_results = checkpoint.get("anthropic_tool_results") or []
        if (
            not isinstance(messages, list)
            or not isinstance(responses_pending_input, list)
            or not isinstance(responses_input_history, list)
            or not isinstance(anthropic_system, list)
            or not isinstance(anthropic_messages, list)
            or not isinstance(anthropic_tool_results, list)
        ):
            raise RuntimeError(
                f"Checkpoint conversation state is malformed: {resume_checkpoint_path}"
            )
        responses_previous_id = checkpoint.get("responses_previous_id")
        policy_state = policy.RolloutPolicyState.from_checkpoint(checkpoint)
        start_turn_index = int(checkpoint.get("next_turn_index") or 0)
        session_started_at = float(checkpoint.get("session_started_at") or session_started_at)
        session_started_monotonic = time.monotonic() - float(
            checkpoint.get("runtime_seconds") or 0.0
        )
        write_trace(trace_path, trace_events)
    else:
        trace_events: list[dict[str, Any]] = []
        messages = prompts.build_initial_messages(task=task, config=policy_config)
        budget_feedback = initialize_budget_feedback(policy_config)
        responses_previous_id: str | None = None
        responses_pending_input: list[dict[str, Any]] = [dict(item) for item in messages]
        responses_input_history: list[dict[str, Any]] = [dict(item) for item in messages]
        anthropic_system, anthropic_messages = build_anthropic_initial_state(messages)
        anthropic_tool_results: list[dict[str, Any]] = []
        policy_state = policy.RolloutPolicyState()
        start_turn_index = 0

    transport = _HostedTransport(
        args=args,
        config=policy_config,
        api_key=api_key,
        responses_previous_id=responses_previous_id,
        responses_pending_input=responses_pending_input,
        responses_input_history=responses_input_history,
        anthropic_system=anthropic_system,
        anthropic_messages=anthropic_messages,
        anthropic_tool_results=anthropic_tool_results,
        chat_tool_schemas=chat_tool_schemas,
        responses_tool_schemas=responses_tool_schemas,
        messages_tool_schemas=messages_tool_schemas,
        budget_estimator=budget_checkpoint.get("token_budget_estimator"),
    )
    rollout_state = RolloutState(
        messages=messages,
        events=trace_events,
        policy=policy_state,
        assistant_turns=sum(message.get("role") == "assistant" for message in messages),
        budget_feedback=budget_feedback,
    )
    checkpoint_next_turn_index = start_turn_index

    def checkpoint_payload(next_turn_index: int) -> dict[str, Any]:
        return {
            "version": 1,
            "model": args.model,
            "provider": args.provider,
            "api_mode": args.api_mode,
            "responses_state_mode": args.responses_state_mode,
            **{field: getattr(args, field) for field in providers.ENDPOINT_FIELDS},
            "auth_header": args.auth_header,
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
            "top_p": getattr(args, "top_p", None),
            "sampling_seed": getattr(args, "sampling_seed", None),
            **policy_config.budget_fields,
            "budget_feedback_state": rollout_state.budget_feedback.checkpoint_fields(),
            "task_id": task["task_id"],
            "target_smiles": task["target_smiles"],
            "hide_target_building_block": _hide_target(args),
            "hide_building_block_ids": sorted(_hidden_ids(args)),
            "hide_building_block_similarity": tools.hidden_similarity,
            "toxicity_screen": tools.screen.config.mode,
            "toxicity_watchlist": str(tools.screen.config.watchlist.resolve()),
            "toxicity_threshold": tools.screen.config.threshold,
            "trace_path": str(trace_path),
            "session_started_at": session_started_at,
            "runtime_seconds": time.monotonic() - session_started_monotonic,
            "next_turn_index": next_turn_index,
            "messages": messages,
            **transport.checkpoint_fields(),
            **policy_state.checkpoint_fields(),
        }

    def persist_state(next_turn_index: int | None = None) -> None:
        nonlocal checkpoint_next_turn_index
        if next_turn_index is not None:
            checkpoint_next_turn_index = next_turn_index
        write_trace(trace_path, trace_events)
        if checkpoint_path is not None:
            write_checkpoint(checkpoint_path, checkpoint_payload(checkpoint_next_turn_index))

    def finish_run(
        submission: dict[str, Any] | None,
        submission_arguments: dict[str, Any] | None,
        *,
        reason: str,
    ) -> dict[str, Any]:
        runtime_seconds = time.monotonic() - session_started_monotonic
        append_run_summary_event(
            trace_events,
            args=args,
            config=policy_config,
            task=task,
            submission=submission,
            session_started_at=session_started_at,
            runtime_seconds=runtime_seconds,
            invalid_submissions=policy_state.invalid_submissions,
            low_similarity_submissions=policy_state.low_similarity_submissions,
        )
        write_trace(trace_path, trace_events)
        delete_checkpoint(checkpoint_path)
        return summarize_run(
            args,
            policy_config,
            task,
            trace_path,
            submission,
            invalid_submissions=policy_state.invalid_submissions,
            low_similarity_submissions=policy_state.low_similarity_submissions,
            runtime_seconds=runtime_seconds,
        )

    if not trace_events:
        trace_events.extend(
            {
                "event_type": "message",
                "role": message["role"],
                "target_smiles": task["target_smiles"],
                "payload": dict(message),
            }
            for message in messages
        )
        if _hide_target(args) or _hidden_ids(args):
            trace_events.insert(
                0,
                hidden_target_event(
                    task["target_smiles"],
                    hidden_ids,
                    hide_target=_hide_target(args),
                    similarity_threshold=tools.hidden_similarity,
                ),
            )
        if target_screen_event:
            trace_events.insert(0, target_screen_event)
        persist_state(0)
    else:
        persist_state(start_turn_index)

    try:
        result = asyncio.run(
            run_rollout(
                task=RolloutTask(
                    task_id=str(task["task_id"]),
                    target_smiles=str(task["target_smiles"]),
                    initial_messages=tuple(messages[:2]),
                ),
                config=policy_config,
                transport=transport,
                budget_provider=transport.budget_snapshot,
                tool_executor=LocalToolExecutor(tools, reaction_library=args.reaction_library),
                tool_schemas=transport.tool_schemas,
                state=rollout_state,
                start_turn_index=start_turn_index,
                event_callback=lambda _event, turn: persist_state(turn),
            )
        )
        return finish_run(
            result.submission,
            result.submission_arguments,
            reason=result.reason,
        )
    except RuntimeError as exc:
        runtime_seconds = time.monotonic() - session_started_monotonic
        trace_events.append(
            {
                "event_type": "run_error",
                "target_smiles": task["target_smiles"],
                "payload": {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
                    "runtime_seconds": runtime_seconds,
                    "next_turn_index": checkpoint_next_turn_index,
                },
            },
        )
        persist_state(checkpoint_next_turn_index)
        if checkpoint_path is not None:
            raise RuntimeError(f"{exc} Checkpoint saved to {checkpoint_path}") from exc
        raise


class _HostedTransport:
    """Adapt BioNeMo-SynClaw's hosted transports to the shared rollout engine."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        config: policy.RolloutPolicyConfig,
        api_key: str | None,
        responses_previous_id: str | None,
        responses_pending_input: list[dict[str, Any]],
        responses_input_history: list[dict[str, Any]],
        anthropic_system: list[dict[str, Any]],
        anthropic_messages: list[dict[str, Any]],
        anthropic_tool_results: list[dict[str, Any]],
        chat_tool_schemas: list[dict[str, Any]],
        responses_tool_schemas: list[dict[str, Any]],
        messages_tool_schemas: list[dict[str, Any]],
        budget_estimator: dict[str, Any] | None = None,
    ) -> None:
        self.args = args
        self.config = config
        self.api_key = api_key
        self.responses_previous_id = responses_previous_id
        self.responses_pending_input = responses_pending_input
        self.responses_input_history = responses_input_history
        self.anthropic_system = anthropic_system
        self.anthropic_messages = anthropic_messages
        self._schemas = {
            "chat": chat_tool_schemas,
            "responses": responses_tool_schemas,
            "messages": messages_tool_schemas,
        }
        self._anthropic_tool_results = anthropic_tool_results
        self.budget_estimator = ContextTokenEstimator.from_checkpoint(budget_estimator)

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return self._schemas[self.args.api_mode]

    @property
    def _server_managed_context(self) -> bool:
        return (
            self.args.api_mode == "responses"
            and self.args.responses_state_mode == "previous_id"
            and bool(self.responses_previous_id)
        )

    def _responses_history(self) -> list[dict[str, Any]]:
        return (
            self.responses_input_history
            if self.args.responses_state_mode == "full"
            else self.responses_pending_input
        )

    def _budget_payload(self, messages: Any, tools: Any) -> dict[str, Any]:
        if self.args.api_mode == "responses":
            payload = {"input": list(self._responses_history())}
            # The server-state anchor already includes the unchanged tool schemas.
            if not self._server_managed_context:
                payload["tools"] = list(tools)
            return payload
        if self.args.api_mode == "messages":
            history = list(self.anthropic_messages)
            if self._anthropic_tool_results:
                history.append(anthropic_tool_result_message(self._anthropic_tool_results))
            return {"system": self.anthropic_system, "messages": history, "tools": list(tools)}
        allowed = {"role", "content", "name", "tool_calls", "tool_call_id", "reasoning_content"}
        history = [{k: v for k, v in row.items() if k in allowed} for row in messages]
        return {"messages": history, "tools": list(tools)}

    def _native_last_input(
        self, native: dict[str, Any], message: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Replace only pending text/output, leaving reasoning and signatures intact."""
        if self.args.api_mode == "responses":
            if (
                message["role"] == "tool"
                and native.get("type") == "function_call_output"
                and native.get("call_id") == message.get("tool_call_id")
            ):
                return {**native, "output": message["content"]}
            if message["role"] == native.get("role") == "user":
                return {**native, "content": message["content"]}
        elif self.args.api_mode == "messages" and native.get("role") == "user":
            blocks = native.get("content") or []
            if (
                blocks
                and message["role"] == "tool"
                and blocks[-1].get("type") == "tool_result"
                and blocks[-1].get("tool_use_id") == message.get("tool_call_id")
            ):
                return {
                    **native,
                    "content": [
                        *blocks[:-1],
                        {**blocks[-1], "content": message["content"]},
                    ],
                }
            if blocks and message["role"] == "user" and blocks[-1].get("type") == "text":
                return {
                    **native,
                    "content": [*blocks[:-1], {**blocks[-1], "text": message["content"]}],
                }
        return None

    def replace_last_input(self, message: dict[str, Any]) -> None:
        if self.args.api_mode == "chat":
            return  # Chat uses the shared messages directly.
        if self.args.api_mode == "responses":
            histories = [self.responses_pending_input, self.responses_input_history]
        else:
            self._flush_anthropic_tool_results()
            histories = [self.anthropic_messages]
        for history in histories:
            updated = self._native_last_input(history[-1], message) if history else None
            if updated is None:
                raise ValueError("Budget feedback must update the last unsent input")
            history[-1] = updated

    def budget_snapshot(self, messages: Any, tools: Any) -> ContextBudgetSnapshot:
        return self.budget_estimator.snapshot(
            self._budget_payload(messages, tools),
            context_window_tokens=self.config.max_model_len,
            max_output_tokens=self.config.max_tokens,
            server_managed=self._server_managed_context,
        )

    async def complete(
        self,
        messages: Any,
        *,
        tools: Any,
        tool_choice: Any,
        max_output_tokens: int | None,
    ) -> AssistantTurn:
        budget_payload = self._budget_payload(messages, tools)
        max_tokens = max_output_tokens
        server_managed = self._server_managed_context
        if self.args.api_mode == "responses":
            if isinstance(tool_choice, dict):
                function = tool_choice.get("function") or {}
                tool_choice = {"type": "function", "name": function.get("name")}
            response = responses_completion(
                responses_url=self.args.responses_url,
                api_key=self.api_key,
                auth_header=getattr(self.args, "auth_header", None),
                model=self.args.model,
                input_items=self._responses_history(),
                previous_response_id=(
                    self.responses_previous_id
                    if self.args.responses_state_mode == "previous_id"
                    else None
                ),
                store=self.args.responses_state_mode == "previous_id",
                max_output_tokens=max_tokens,
                temperature=self.args.temperature,
                top_p=getattr(self.args, "top_p", None),
                sampling_seed=getattr(self.args, "sampling_seed", None),
                reasoning_effort=self.args.reasoning_effort,
                retries=self.args.retries,
                request_timeout=self.args.request_timeout,
                tool_choice=tool_choice,
                tools=list(tools),
            )
            self.budget_estimator.observe(
                budget_payload,
                response.get("usage"),
                server_managed=server_managed,
                replay_output=(
                    response.get("output") or []
                    if self.args.responses_state_mode == "full"
                    else None
                ),
            )
            if self.args.responses_state_mode == "previous_id":
                self.responses_previous_id = response.get("id")
            else:
                self.responses_input_history.extend(response.get("output") or [])
            self.responses_pending_input.clear()
            message = responses_to_chat_message(response)
            return AssistantTurn(
                message=message,
                finish_reason=responses_finish_reason(response),
            )

        if self.args.api_mode == "messages":
            self._flush_anthropic_tool_results()
            response = anthropic_completion(
                messages_url=self.args.messages_url,
                api_key=self.api_key,
                auth_header=getattr(self.args, "auth_header", None),
                model=self.args.model,
                system=self.anthropic_system,
                messages=self.anthropic_messages,
                max_tokens=max_tokens,
                temperature=self.args.temperature,
                top_p=getattr(self.args, "top_p", None),
                reasoning_effort=self.args.reasoning_effort,
                retries=self.args.retries,
                request_timeout=self.args.request_timeout,
                tool_choice=None,
                tools=list(tools),
            )
            self.budget_estimator.observe(
                budget_payload,
                response.get("usage"),
                separate_cache_usage=True,
            )
            self.anthropic_messages.append(
                {"role": "assistant", "content": response.get("content") or []}
            )
            message = anthropic_response_to_chat_message(response)
            return AssistantTurn(message=message, finish_reason=response.get("stop_reason"))

        response = chat_completion(
            base_url=self.args.base_url,
            api_key=self.api_key,
            auth_header=getattr(self.args, "auth_header", None),
            model=self.args.model,
            messages=list(messages),
            max_tokens=max_tokens,
            token_param=resolve_token_param(self.args.base_url, self.args.token_param),
            temperature=self.args.temperature,
            top_p=getattr(self.args, "top_p", None),
            sampling_seed=getattr(self.args, "sampling_seed", None),
            reasoning_effort=self.args.reasoning_effort,
            retries=self.args.retries,
            request_timeout=self.args.request_timeout,
            tool_choice=tool_choice,
            tools=list(tools),
        )
        choice = (response.get("choices") or [{}])[0]
        self.budget_estimator.observe(budget_payload, response.get("usage"))
        message = dict(choice.get("message") or {})
        if response.get("model") is not None:
            message["response_model"] = response.get("model")
        if response.get("system_fingerprint") is not None:
            message["system_fingerprint"] = response.get("system_fingerprint")
        attach_response_usage(message, response)
        return AssistantTurn(message=message, finish_reason=choice.get("finish_reason"))

    def append_user(self, message: dict[str, Any]) -> None:
        if self.args.api_mode == "responses":
            self.responses_pending_input.append(message)
            self.responses_input_history.append(message)
        elif self.args.api_mode == "messages":
            self._flush_anthropic_tool_results()
            self.anthropic_messages.append(
                anthropic_text_message(str(message.get("content") or ""))
            )

    def append_tool(self, call: Any, result_json: str) -> None:
        if self.args.api_mode == "responses":
            output = {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": result_json,
            }
            self.responses_pending_input.append(output)
            self.responses_input_history.append(output)
        elif self.args.api_mode == "messages":
            self._anthropic_tool_results.append(
                {"type": "tool_result", "tool_use_id": call.call_id, "content": result_json}
            )

    def _flush_anthropic_tool_results(self) -> None:
        if not self._anthropic_tool_results:
            return
        self.anthropic_messages.append(anthropic_tool_result_message(self._anthropic_tool_results))
        self._anthropic_tool_results = []

    def checkpoint_fields(self) -> dict[str, Any]:
        return {
            "token_budget_estimator": self.budget_estimator.checkpoint_fields(),
            "responses_previous_id": self.responses_previous_id,
            "responses_pending_input": self.responses_pending_input,
            "responses_input_history": self.responses_input_history,
            "anthropic_system": self.anthropic_system,
            "anthropic_messages": self.anthropic_messages,
            "anthropic_tool_results": self._anthropic_tool_results,
        }


def _positive_token_limit(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("token limit must be positive")
    return result


def apply_provider_defaults(args: argparse.Namespace) -> None:
    """Fill unset transport options from the provider preset; flags always win."""

    provider = getattr(args, "provider", None) or "custom"
    preset = providers.load_provider_config().preset(provider)
    args.provider = provider
    if not getattr(args, "api_mode", None):
        args.api_mode = preset.api_mode
    for name in providers.ENDPOINT_FIELDS:
        if not getattr(args, name, None):
            setattr(args, name, getattr(preset, name))
    if not getattr(args, "responses_state_mode", None):
        args.responses_state_mode = preset.responses_state_mode
    if not getattr(args, "auth_header", None):
        args.auth_header = providers.default_auth_header(args.api_mode)
    normalize_reasoning_effort(args)


_ENDPOINT_FOR_API_MODE = {
    "chat": ("base_url", "--base-url"),
    "responses": ("responses_url", "--responses-url"),
    "messages": ("messages_url", "--messages-url"),
}


def require_endpoint(args: argparse.Namespace) -> None:
    """Fail before any work when the selected API mode has no endpoint.

    A preset may leave an endpoint empty (``custom`` leaves all of them empty);
    without this check the first request fails with an opaque URL error.
    """

    field, flag = _ENDPOINT_FOR_API_MODE.get(args.api_mode, ("base_url", "--base-url"))
    if getattr(args, field, None):
        return
    raise RuntimeError(
        f"No {field} is configured for provider {args.provider!r} ({args.api_mode} API); "
        f"pass {flag}"
    )


def uses_preset_endpoint(args: argparse.Namespace) -> bool:
    """Whether the selected API mode still targets the provider preset's own endpoint."""

    field, _ = _ENDPOINT_FOR_API_MODE.get(args.api_mode, ("base_url", "--base-url"))
    preset = providers.load_provider_config().preset(args.provider)
    return getattr(args, field, None) == getattr(preset, field)


def normalize_reasoning_effort(args: argparse.Namespace) -> None:
    """Map ``max`` onto the strongest value each API accepts."""

    if (
        getattr(args, "reasoning_effort", None) == "max"
        and getattr(args, "api_mode", None) != "messages"
    ):
        args.reasoning_effort = "xhigh"


def resolve_run_task(args: argparse.Namespace) -> dict[str, Any]:
    """The task a run plans: the --smiles target, else the selected task-set row."""

    taskset = getattr(args, "taskset", None)
    task_index = getattr(args, "task_index", None)
    smiles = getattr(args, "smiles", None)
    if smiles is None:
        return load_task(
            DEFAULT_TASKSET if taskset is None else taskset, 0 if task_index is None else task_index
        )
    if taskset is not None or task_index is not None:
        raise RuntimeError("--smiles cannot be combined with --taskset or --task-index")
    try:
        return build_adhoc_task(smiles)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def load_task(path: str, index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return prompts.require_current_prompt_text(json.loads(line))
    raise ValueError(f"No task at index {index} in {path}")


def resolve_task_db_path(db_arg: str | Path | None) -> Path:
    return Path(db_arg) if db_arg else DEFAULT_MODEL_RUN_DB


def chat_completion(
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    token_param: str,
    temperature: float | None,
    top_p: float | None,
    sampling_seed: int | None,
    reasoning_effort: str | None,
    retries: int,
    request_timeout: float,
    tool_choice: Any,
    tools: list[dict[str, Any]],
    auth_header: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        token_param: max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if sampling_seed is not None:
        payload["seed"] = sampling_seed
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            **providers.auth_headers(api_key, auth_header or "bearer"),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRYABLE_HTTP_CODES and attempt < retries:
                time.sleep(retry_delay_seconds(attempt, body=body, headers=exc.headers))
                continue
            raise RuntimeError(f"Chat Completions API HTTP {exc.code}: {body}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < retries:
                time.sleep(retry_delay_seconds(attempt))
                continue
            raise RuntimeError(f"Chat Completions API request timed out or failed: {exc}") from exc
    raise RuntimeError("Chat Completions API request failed after retries")


def anthropic_system_block(content: str) -> list[dict[str, Any]]:
    if not content:
        return []
    return [
        {
            "type": "text",
            "text": content,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def normalize_anthropic_system(system: Any) -> list[dict[str, Any]]:
    if isinstance(system, str):
        return anthropic_system_block(system)
    if isinstance(system, list):
        return system
    return []


def build_anthropic_initial_state(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "system" and not system:
            system = anthropic_system_block(content)
        elif role in {"user", "assistant"}:
            conversation.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )
    return system, conversation


def anthropic_text_message(content: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": content}]}


def anthropic_tool_result_message(tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "user", "content": tool_results}


def anthropic_completion(
    messages_url: str,
    api_key: str | None,
    model: str,
    system: str | list[dict[str, Any]],
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    reasoning_effort: str | None,
    retries: int,
    request_timeout: float,
    tool_choice: Any | None,
    tools: list[dict[str, Any]],
    auth_header: str | None = None,
) -> dict[str, Any]:
    system_blocks = normalize_anthropic_system(system)
    payload: dict[str, Any] = {
        "model": model,
        "system": system_blocks,
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
        "cache_control": {"type": "ephemeral"},
    }
    if reasoning_effort:
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": reasoning_effort}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    request = urllib.request.Request(
        messages_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            **providers.auth_headers(api_key, auth_header or "x-api-key"),
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRYABLE_HTTP_CODES and attempt < retries:
                time.sleep(retry_delay_seconds(attempt, body=body, headers=exc.headers))
                continue
            raise RuntimeError(f"Messages API HTTP {exc.code}: {body}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < retries:
                time.sleep(retry_delay_seconds(attempt))
                continue
            raise RuntimeError(f"Messages API request timed out or failed: {exc}") from exc
    raise RuntimeError("Messages API request failed after retries")


def responses_completion(
    responses_url: str,
    api_key: str | None,
    model: str,
    input_items: list[dict[str, Any]],
    previous_response_id: str | None,
    store: bool,
    max_output_tokens: int,
    temperature: float | None,
    top_p: float | None,
    sampling_seed: int | None,
    reasoning_effort: str | None,
    retries: int,
    request_timeout: float,
    tool_choice: Any,
    tools: list[dict[str, Any]],
    auth_header: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if sampling_seed is not None:
        payload["seed"] = sampling_seed
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if store:
        payload["store"] = True
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    request = urllib.request.Request(
        responses_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            **providers.auth_headers(api_key, auth_header or "bearer"),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRYABLE_HTTP_CODES and attempt < retries:
                time.sleep(retry_delay_seconds(attempt, body=body, headers=exc.headers))
                continue
            raise RuntimeError(f"Responses API HTTP {exc.code}: {body}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < retries:
                time.sleep(retry_delay_seconds(attempt))
                continue
            raise RuntimeError(f"Responses API request timed out or failed: {exc}") from exc
    raise RuntimeError("Responses API request failed after retries")


def responses_finish_reason(response: dict[str, Any]) -> str | None:
    if str(response.get("status") or "").strip().lower() != "incomplete":
        return None
    details = response.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    normalized = str(reason or "incomplete").strip().lower()
    return normalized or "incomplete"


def responses_to_chat_message(response: dict[str, Any]) -> dict[str, Any]:
    output = response.get("output") or []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"function_call", "tool_call"}:
            call_id = item.get("call_id") or item.get("id")
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "{}",
                    },
                    "raw_response_output": item,
                }
            )
        elif item_type == "message":
            for content in item.get("content") or []:
                content_type = content.get("type")
                if content_type in {"output_text", "text"}:
                    content_parts.append(content.get("text") or "")
                elif content_type in {"reasoning_text", "summary_text"}:
                    reasoning_parts.append(content.get("text") or "")
        elif item_type in {"reasoning", "reasoning_summary"}:
            reasoning = _extract_response_text(item)
            if reasoning:
                reasoning_parts.append(reasoning)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(part for part in content_parts if part),
        "response_model": response.get("model"),
        "system_fingerprint": response.get("system_fingerprint"),
        "responses_id": response.get("id"),
        "responses_status": response.get("status"),
        "responses_output_types": [item.get("type") for item in output if isinstance(item, dict)],
        "raw_response_output": output,
    }
    attach_response_usage(message, response)
    if reasoning_parts:
        message["reasoning_text"] = "\n".join(part for part in reasoning_parts if part)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def anthropic_response_to_chat_message(response: dict[str, Any]) -> dict[str, Any]:
    content = response.get("content") or []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            content_parts.append(block.get("text") or "")
        elif block_type == "thinking":
            thinking_text = _extract_response_text(block)
            if thinking_text:
                reasoning_parts.append(thinking_text)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input") or {}, sort_keys=True),
                    },
                    "raw_response_output": block,
                }
            )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(part for part in content_parts if part),
        "response_model": response.get("model"),
        "anthropic_id": response.get("id"),
        "anthropic_stop_reason": response.get("stop_reason"),
        "anthropic_content_types": [
            block.get("type") for block in content if isinstance(block, dict)
        ],
        "raw_response_output": content,
    }
    attach_response_usage(message, response)
    if reasoning_parts:
        message["reasoning_text"] = "\n".join(part for part in reasoning_parts if part)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _extract_response_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_extract_response_text(item) for item in value]
        return "\n".join(part for part in parts if part) or None
    if not isinstance(value, dict):
        return None
    parts: list[str] = []
    for key in ("text", "summary", "content"):
        text = _extract_response_text(value.get(key))
        if text:
            parts.append(text)
    return "\n".join(parts) or None


def resolve_api_key(provider: str, api_key_env: str, *, use_preset_keys: bool = True) -> str | None:
    return providers.resolve_api_key(provider, api_key_env, use_preset_keys=use_preset_keys)


def resolve_token_param(base_url: str, token_param: str) -> str:
    if token_param != "auto":
        return token_param
    if "api.openai.com" in base_url:
        return "max_completion_tokens"
    return "max_tokens"


def write_trace(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a JSON object: {path}")
    return checkpoint


def delete_checkpoint(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    try:
        path.with_name(f"{path.name}.tmp").unlink()
    except FileNotFoundError:
        pass


def checkpoint_matches_run(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    task: dict[str, Any],
    config: policy.RolloutPolicyConfig,
) -> bool:
    if checkpoint.get("responses_state_mode") != getattr(args, "responses_state_mode", None):
        return False
    # A different endpoint cannot continue the server-side state (``responses_previous_id``)
    # or the conversation the checkpoint recorded.
    for field, default in (
        *((name, None) for name in providers.ENDPOINT_FIELDS),
        ("auth_header", None),
        ("reasoning_effort", None),
        ("temperature", None),
        ("top_p", None),
        ("sampling_seed", None),
    ):
        if field in checkpoint and checkpoint.get(field) != getattr(args, field, default):
            return False
    # The restored system prompt states these budgets, and enforcement, including the stop
    # condition, must keep them.
    if any(
        field in checkpoint and checkpoint[field] != value
        for field, value in config.budget_fields.items()
    ):
        return False
    # Restored submissions were screened under the recorded settings only; checkpoints that
    # predate the screen ran without it.
    screen = toxicity.config_from_args(args)
    if checkpoint.get("toxicity_screen", toxicity.MODE_OFF) != screen.mode:
        return False
    if screen.mode == toxicity.MODE_ON and (
        checkpoint.get("toxicity_watchlist") != str(screen.watchlist.resolve())
        or checkpoint.get("toxicity_threshold") != screen.threshold
    ):
        return False
    return (
        checkpoint.get("model") == args.model
        and checkpoint.get("provider") == args.provider
        and checkpoint.get("api_mode") == args.api_mode
        and checkpoint.get("task_id") == task.get("task_id")
        and checkpoint.get("target_smiles") == task.get("target_smiles")
        and bool(checkpoint.get("hide_target_building_block", False)) == _hide_target(args)
        and frozenset(checkpoint.get("hide_building_block_ids", [])) == _hidden_ids(args)
        and checkpoint.get("hide_building_block_similarity")
        == getattr(args, "hide_building_block_similarity", None)
    )


def run_excluded_tools(args: argparse.Namespace, task: dict[str, Any]) -> set[str]:
    return contracts.task_excluded_tools(
        task, contracts.normalize_excluded_tools(getattr(args, "exclude_tool", []) or [])
    )


def refuse_target(
    args: argparse.Namespace, task: dict[str, Any], event: dict[str, Any], mode: str
) -> dict[str, Any]:
    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_trace(trace_path, [event])
    if summary_output := getattr(args, "summary_output", None):
        from synclaw.evaluation.evaluate import evaluate_trace_file, write_summary_csv

        evaluated = evaluate_trace_file(
            trace_path,
            args.db,
            model=args.model,
            target_smiles=task["target_smiles"],
            tool_config={"toxicity_screen": mode},
        )
        write_summary_csv([evaluated], summary_output)
    return {
        "refused": True,
        "task_id": task["task_id"],
        "target_smiles": task["target_smiles"],
        "error_message": event["payload"]["error_message"],
        "trace": str(trace_path),
        **toxicity.toxicity_summary_fields([event], mode),
    }


def summarize_run(
    args: argparse.Namespace,
    config: policy.RolloutPolicyConfig,
    task: dict[str, Any],
    trace_path: Path,
    submission: dict[str, Any] | None,
    invalid_submissions: int = 0,
    low_similarity_submissions: int = 0,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    summary = summarize(task, args.model, trace_path, submission)
    summary["provider"] = args.provider
    summary["api_mode"] = args.api_mode
    summary["invalid_submissions"] = invalid_submissions
    summary["low_similarity_submissions"] = low_similarity_submissions
    summary["runtime_seconds"] = runtime_seconds
    summary.update(policy.request_summary_fields(args, config))
    summary["excluded_tools"] = sorted(config.excluded_tools)
    trace_events = list(db.iter_jsonl(trace_path))
    summary.update(latest_response_metadata(trace_events))
    summary.update(token_usage.aggregate_token_usage_from_events(trace_events))
    summary.update(budget_summary_fields(trace_events))
    summary.update(hidden_target_summary_fields(trace_events))
    summary.update(toxicity_summary(args, trace_events))
    summary_output = getattr(args, "summary_output", None)
    if summary_output:
        from synclaw.evaluation.evaluate import evaluate_trace_file, write_summary_csv

        evaluated = evaluate_trace_file(
            trace_path,
            db_path=args.db,
            model=args.model,
            prompt_version=getattr(args, "prompt_version", None),
            target_smiles=task["target_smiles"],
            tool_config={
                "trace_file": str(trace_path),
                "stop_condition": config.stop_condition,
                "toxicity_screen": summary["toxicity_screen"],
            },
        )
        write_summary_csv([evaluated], summary_output)
        summary["summary_output"] = summary_output
        summary["evaluation_summary"] = evaluated
    return summary


def toxicity_summary(args: argparse.Namespace, events: list[dict[str, Any]]) -> dict[str, Any]:
    mode = toxicity.config_from_args(args).mode
    return toxicity.toxicity_summary_fields(events, mode)


def append_run_summary_event(
    trace_events: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    config: policy.RolloutPolicyConfig,
    task: dict[str, Any],
    submission: dict[str, Any] | None,
    session_started_at: float,
    runtime_seconds: float,
    invalid_submissions: int,
    low_similarity_submissions: int,
) -> None:
    usage = token_usage.aggregate_token_usage_from_events(trace_events)
    response_metadata = latest_response_metadata(trace_events)
    trace_events.append(
        {
            "event_type": "run_summary",
            "target_smiles": task["target_smiles"],
            "payload": {
                "model": args.model,
                "provider": args.provider,
                "api_mode": args.api_mode,
                **policy.request_summary_fields(args, config),
                "excluded_tools": sorted(config.excluded_tools),
                **response_metadata,
                "task_id": task["task_id"],
                "runtime_seconds": runtime_seconds,
                "started_at_unix": session_started_at,
                "completed_at_unix": session_started_at + runtime_seconds,
                **policy.submission_summary_fields(submission, task["target_smiles"]),
                "invalid_submissions": invalid_submissions,
                "low_similarity_submissions": low_similarity_submissions,
                **usage,
                **budget_summary_fields(trace_events),
                **toxicity_summary(args, trace_events),
            },
        }
    )


def latest_response_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return deployment metadata from the latest assistant response that provides it."""

    result: dict[str, Any] = {}
    for event in events:
        if event.get("event_type") != "message" or event.get("role") != "assistant":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("response_model") is not None:
            result["response_model"] = payload["response_model"]
        if payload.get("system_fingerprint") is not None:
            result["system_fingerprint"] = payload["system_fingerprint"]
    return result


def summarize(
    task: dict[str, Any],
    model: str,
    trace_path: Path,
    submission: dict[str, Any] | None,
) -> dict[str, Any]:
    reaction_valid = submission.get("reaction_validation") == "valid" if submission else None
    submission_fields = policy.submission_summary_fields(submission, task["target_smiles"])
    return {
        "model": model,
        "task_id": task["task_id"],
        "target_smiles": task["target_smiles"],
        "trace": str(trace_path),
        **submission_fields,
        "reaction_valid": reaction_valid,
        "issues": submission.get("issues") if submission else None,
        "direct_building_block_ref": submission.get("direct_building_block_ref")
        if submission
        else None,
        "referenced_building_block_ids": submission.get("referenced_building_block_ids")
        if submission
        else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
