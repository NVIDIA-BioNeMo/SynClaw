# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral state and decisions for SynClaw tool-use rollouts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Literal

from synclaw.core import chem
from synclaw.core.libraries import DEFAULT_REACTION_LIBRARIES
from synclaw.tools.compaction import compact_json_size, compact_tool_result
from synclaw.tools.contracts import (
    TASK_OVERRIDE_FIELDS,
    UNLIMITED_BUDGET,
    normalize_dispatch_arguments,
    task_budget_overrides,
)

PERSISTENCE_GUARANTEE_SENTENCE = (
    "A better reconstruction is guaranteed to be possible for this task."
)
NON_TOOL_PERSISTENCE_GUARANTEE_SENTENCE = (
    "A better reconstruction is guaranteed to be possible for this task; "
    "do not conclude the current pathway is the maximum achievable route."
)

DEFAULT_MAX_TURNS = 80
DEFAULT_MAX_TOOL_CALLS = 150
DEFAULT_MAX_TOKENS = 32768
# Share of the context kept free of output when each response is clipped to the context left,
# absorbing input-estimation error: 1/32 is 8192 tokens of a 262144-token context.
DEFAULT_CONTEXT_SAFETY_MARGIN_FRACTION = 1 / 32
MIN_CONTEXT_SAFETY_MARGIN_TOKENS = 512
DEFAULT_TURN_WARNING_REMAINING = 3
DEFAULT_FORCE_SUBMIT_AFTER_NON_TOOL_TURNS = 2
DEFAULT_SAME_PATHWAY_FINISH_AFTER = 5
DEFAULT_MAX_LOW_SIMILARITY_SUBMISSIONS: int | None = None
DEFAULT_REACTION_LIBRARY = DEFAULT_REACTION_LIBRARIES
DEFAULT_CONTEXT_WARNING_REMAINING_FRACTION = 0.5
DEFAULT_CONTEXT_CRITICAL_REMAINING_FRACTION = 0.25
# The context length a deployment serves and enforces when it names none of its own.
# RolloutPolicyConfig itself still defaults to no context limit.
DEFAULT_MAX_MODEL_LEN = 262_144
# What an "inf" task budget becomes; an unlimited max_model_len is handled separately.
_UNLIMITED_BUDGET_VALUES: dict[str, int | None] = {"max_turns": None, "max_tool_calls": 0, "max_tokens": None}

# What ends a rollout: an exact match, an accepted route at or above a similarity, or the model.
StopCondition = float | Literal["exact", "model"]
DEFAULT_STOP_CONDITION: StopCondition = "exact"


def parse_stop_condition(value: str | float) -> StopCondition:
    """Parse ``exact``, ``model`` or a similarity between 0 and 1."""

    text = value.strip().lower() if isinstance(value, str) else None
    if text == "exact":
        return "exact"
    if text == "model":
        return "model"
    try:
        condition = math.nan if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        condition = math.nan
    if not math.isfinite(condition) or not 0 <= condition <= 1:
        raise ValueError(f"Stop condition must be 'exact', 'model' or a similarity between 0 and 1, not {value!r}")
    return condition

def meets_submission_floor(
    similarity: float | None,
    stop_condition: StopCondition,
    *,
    identical_to_target: bool | None,
) -> bool:
    """Apply the same success criterion during rollouts and trace evaluation.

    Identity must come from ``compare_molecular_identity`` in Standard InChI
    mode. Callers must separately require validator acceptance. Numeric floors
    allow an identity match even when normalized tautomers have low similarity;
    ``exact`` requires identity and never falls back to fingerprint similarity.
    ``model`` is scored like ``exact``, though no submission ends its rollouts.
    """

    if stop_condition in ("exact", "model"):
        return identical_to_target is True
    return (
        identical_to_target is True
        or stop_condition <= 0
        or similarity is not None and float(similarity) >= stop_condition
    )


@dataclass(frozen=True, slots=True)
class RolloutPolicyConfig:
    """Behavioral settings shared by every rollout transport."""

    # None: no turn cap. Zero tool calls: no tool-call cap.
    max_turns: int | None = DEFAULT_MAX_TURNS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    # None: no cap of its own; each response is clipped only to the context left.
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    max_model_len: int | None = None
    # False keeps the context limit enforced but out of the prompt and feedback.
    context_limit_stated: bool = True
    context_safety_margin_fraction: float = DEFAULT_CONTEXT_SAFETY_MARGIN_FRACTION
    # Progress messages only: remaining budgets, context warnings and the turn warning.
    token_budget_feedback: bool = True
    # "exact" or a similarity: that submission ends the rollout, with coaching below it.
    # "model": no submission ends it and nothing coaches; a reply without a tool call ends it.
    stop_condition: StopCondition = DEFAULT_STOP_CONDITION
    turn_warning_remaining: int = DEFAULT_TURN_WARNING_REMAINING
    force_submit_after_non_tool_turns: int = DEFAULT_FORCE_SUBMIT_AFTER_NON_TOOL_TURNS
    same_pathway_finish_after: int = DEFAULT_SAME_PATHWAY_FINISH_AFTER
    max_low_similarity_submissions: int | None = DEFAULT_MAX_LOW_SIMILARITY_SUBMISSIONS
    excluded_tools: frozenset[str] = frozenset()
    context_warning_remaining_fraction: float = DEFAULT_CONTEXT_WARNING_REMAINING_FRACTION
    context_critical_remaining_fraction: float = DEFAULT_CONTEXT_CRITICAL_REMAINING_FRACTION

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_condition", parse_stop_condition(self.stop_condition))
        if self.max_tokens is None and self.max_model_len is None:
            raise ValueError("An unlimited output cap needs a known context limit to clip it")
        if self.max_turns is None and self.max_model_len is None:
            raise ValueError("Unlimited turns need a known context limit to end the rollout")
        if not 0 <= self.context_safety_margin_fraction < 1:
            raise ValueError("context_safety_margin_fraction must be at least 0 and below 1")
        if self.max_model_len is not None and self.max_model_len <= self.context_safety_margin_tokens:
            raise ValueError(
                f"A context limit of {self.max_model_len} tokens leaves no room for output after "
                f"the {self.context_safety_margin_tokens}-token safety margin"
            )
        if not (0 <= self.context_critical_remaining_fraction <= self.context_warning_remaining_fraction <= 1):
            raise ValueError(
                "context_critical_remaining_fraction and context_warning_remaining_fraction must "
                "satisfy 0 <= critical <= warning <= 1"
            )

    @property
    def budget_fields(self) -> dict[str, Any]:
        """The effective budgets and their enforcement settings, as checkpoints record them."""

        names = (*TASK_OVERRIDE_FIELDS, "context_limit_stated", "context_safety_margin_fraction", "stop_condition")
        return {name: getattr(self, name) for name in names}

    @property
    def model_stops(self) -> bool:
        """Whether the model, not a submission, ends the rollout."""

        return self.stop_condition == "model"

    @property
    def context_safety_margin_tokens(self) -> int:
        """The margin for this context limit: its fraction, but at least a small floor."""

        if self.max_model_len is None or self.context_safety_margin_fraction == 0:
            return 0
        return max(
            MIN_CONTEXT_SAFETY_MARGIN_TOKENS, round(self.max_model_len * self.context_safety_margin_fraction)
        )

    @property
    def stated_max_model_len(self) -> int | None:
        """The context limit as the model is told it, or None when it is unstated."""

        return self.max_model_len if self.context_limit_stated else None


def task_budget_config(task: dict[str, Any], config: RolloutPolicyConfig) -> RolloutPolicyConfig:
    """Apply a task row's budget overrides to the runner's settings.

    Missing or null fields keep the runner's value. A task may lower the runner's
    context limit but not raise it, since the runner's limit is what the server allows.
    An ``"inf"`` budget is unlimited and left out of the prompt, except the context,
    which stays enforced at the runner's limit but is no longer stated.
    """

    overrides: dict[str, Any] = dict(task_budget_overrides(task))
    for name in [name for name, value in overrides.items() if value == UNLIMITED_BUDGET]:
        if name == "max_model_len":
            del overrides[name]
            overrides["context_limit_stated"] = False
        else:
            overrides[name] = _UNLIMITED_BUDGET_VALUES[name]
    context = overrides.get("max_model_len")
    if context is not None and config.max_model_len is not None and context > config.max_model_len:
        raise ValueError(
            f"Task max_model_len {context} exceeds the runner's context limit {config.max_model_len}"
        )
    return replace(config, **overrides)


@dataclass(slots=True)
class RolloutPolicyState:
    """Mutable policy state with a stable version-one checkpoint representation."""

    total_tool_calls: int = 0
    consecutive_non_tool_turns_after_tools: int = 0
    forced_submit_requested: bool = False
    # The one request to submit when the model stops without an accepted route; unlike
    # forced_submit_requested, a later submission does not clear it.
    model_stop_submit_requested: bool = False
    tool_limit_finalization_requested: bool = False
    turn_warning_sent: bool = False
    output_limit_recovery_used: bool = False
    invalid_submissions: int = 0
    low_similarity_submissions: int = 0
    repeated_submission_signature: str | None = None
    repeated_submission_count: int = 0
    last_submission: dict[str, Any] | None = None
    last_submission_arguments: dict[str, Any] | None = None
    best_accepted_submission: dict[str, Any] | None = None
    best_accepted_submission_arguments: dict[str, Any] | None = None
    best_accepted_similarity: float | None = None
    best_accepted_identical: bool | None = None

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> RolloutPolicyState:
        return cls(
            total_tool_calls=int(checkpoint.get("total_tool_calls") or 0),
            consecutive_non_tool_turns_after_tools=int(
                checkpoint.get("consecutive_non_tool_turns_after_tools")
                or checkpoint.get("non_tool_turns_after_search")
                or 0
            ),
            forced_submit_requested=bool(checkpoint.get("forced_submit_requested")),
            model_stop_submit_requested=bool(checkpoint.get("model_stop_submit_requested")),
            tool_limit_finalization_requested=bool(
                checkpoint.get("tool_limit_finalization_requested")
            ),
            turn_warning_sent=bool(checkpoint.get("turn_warning_sent")),
            output_limit_recovery_used=bool(checkpoint.get("output_limit_recovery_used")),
            invalid_submissions=int(checkpoint.get("invalid_submissions") or 0),
            low_similarity_submissions=int(
                checkpoint.get("low_similarity_submissions") or 0
            ),
            repeated_submission_signature=checkpoint.get("repeated_submission_signature"),
            repeated_submission_count=int(checkpoint.get("repeated_submission_count") or 0),
            last_submission=checkpoint.get("last_submission"),
            last_submission_arguments=checkpoint.get("last_submission_arguments"),
            best_accepted_submission=checkpoint.get("best_accepted_submission"),
            best_accepted_submission_arguments=checkpoint.get(
                "best_accepted_submission_arguments"
            ),
            best_accepted_similarity=checkpoint.get("best_accepted_similarity"),
            best_accepted_identical=checkpoint.get("best_accepted_identical"),
        )

    def checkpoint_fields(self) -> dict[str, Any]:
        return {
            "total_tool_calls": self.total_tool_calls,
            "consecutive_non_tool_turns_after_tools": (
                self.consecutive_non_tool_turns_after_tools
            ),
            "forced_submit_requested": self.forced_submit_requested,
            "model_stop_submit_requested": self.model_stop_submit_requested,
            "tool_limit_finalization_requested": self.tool_limit_finalization_requested,
            "turn_warning_sent": self.turn_warning_sent,
            "output_limit_recovery_used": self.output_limit_recovery_used,
            "invalid_submissions": self.invalid_submissions,
            "low_similarity_submissions": self.low_similarity_submissions,
            "repeated_submission_signature": self.repeated_submission_signature,
            "repeated_submission_count": self.repeated_submission_count,
            "last_submission": self.last_submission,
            "last_submission_arguments": self.last_submission_arguments,
            "best_accepted_submission": self.best_accepted_submission,
            "best_accepted_submission_arguments": self.best_accepted_submission_arguments,
            "best_accepted_similarity": self.best_accepted_similarity,
            "best_accepted_identical": self.best_accepted_identical,
        }

    @property
    def has_submission(self) -> bool:
        return self.last_submission is not None

    def selected_submission(
        self,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return the best accepted submission, else the last one attempted."""

        if self.best_accepted_submission is not None:
            return self.best_accepted_submission, self.best_accepted_submission_arguments
        return self.last_submission, self.last_submission_arguments


@dataclass(frozen=True, slots=True)
class NoToolDecision:
    finish_reason: str | None = None
    reminder: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    dispatch_arguments: dict[str, Any]
    argument_error: str | None
    event_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultDecision:
    repair: str | None = None
    terminal_submission: dict[str, Any] | None = None
    terminal_due_to_tool_limit: bool = False
    meets_floor: bool = False


def add_rollout_flag_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the context margin, progress feedback and stop condition flags shared by every rollout host."""

    parser.add_argument(
        "--context-safety-margin-fraction", type=float,
        default=DEFAULT_CONTEXT_SAFETY_MARGIN_FRACTION,
        help=(
            "Share of the context limit kept free of output when clipping it to the context left, "
            "at least 512 tokens; 0 disables the margin (default: 1/32)"
        ),
    )
    parser.add_argument(
        "--no-token-budget-feedback", dest="token_budget_feedback", action="store_false",
        help=(
            "Disable progress messages: remaining budgets, context warnings and the turn warning. "
            "The system prompt still states the limits. A task's token_budget_feedback overrides it"
        ),
    )
    parser.add_argument(
        "--stop-condition", type=parse_stop_condition, default=DEFAULT_STOP_CONDITION,
        help=(
            "What ends a rollout. 'exact' (default): an accepted exact match, with coaching after "
            "weaker submissions. A similarity such as 0.8: an accepted route at or above it; 0 "
            "ends on any accepted route. 'model': the model's reply without a tool call, keeping "
            "its best accepted route, with no coaching; a model with none is asked once to submit"
        ),
    )


def turn_reminders(
    config: RolloutPolicyConfig,
    state: RolloutPolicyState,
    *,
    turn_index: int,
) -> list[str]:
    """Return and record reminders that must precede the next model turn."""

    reminders: list[str] = []
    turns_remaining = None if config.max_turns is None else config.max_turns - turn_index
    if (
        turns_remaining is not None
        and config.token_budget_feedback
        and config.turn_warning_remaining > 0
        and turns_remaining == config.turn_warning_remaining
        and turns_remaining > 1
        and not state.turn_warning_sent
    ):
        finish_clause = ""
        if state.best_accepted_submission is not None and not config.model_stops:
            finish_clause = (
                " If you absolutely cannot improve on a previously submitted pathway, "
                "finish by responding with <FINISH>."
            )
        reminders.append(
            f"You have {turns_remaining} assistant turns remaining, including this turn. "
            "If you have already identified a verified route that improves on prior submissions, "
            "call submit_pathway for it now so it can be scored. Use any remaining tools only "
            "to verify a clearly better candidate; otherwise submit the best verified route you have."
            f"{finish_clause}"
        )
        state.turn_warning_sent = True

    if turns_remaining == 1 and not state.forced_submit_requested:
        reminders.append(
            "This is the final allowed turn. Call submit_pathway now with your best route."
        )
        state.forced_submit_requested = True

    if (
        config.max_tool_calls > 0
        and state.total_tool_calls >= config.max_tool_calls
        and not state.forced_submit_requested
    ):
        reminders.append(
            "The tool-call budget has been reached, so this is the final allowed turn. "
            "Call submit_pathway now with your best route."
        )
        state.forced_submit_requested = True
        state.tool_limit_finalization_requested = True
    return reminders


def handle_no_tool_response(
    config: RolloutPolicyConfig,
    state: RolloutPolicyState,
    *,
    content: str,
) -> NoToolDecision:
    if state.best_accepted_submission is not None and "<FINISH>" in content:
        return NoToolDecision(finish_reason="finish_token_best_accepted_submission")
    if state.forced_submit_requested and state.tool_limit_finalization_requested:
        # Checked before an accepted stop, so a stop at the tool-call limit is reported as one.
        return NoToolDecision(finish_reason=tool_limit_finish_reason(state))
    if config.model_stops:
        if state.best_accepted_submission is not None:
            return NoToolDecision(finish_reason="model_stop_best_accepted_submission")
        if state.forced_submit_requested or state.model_stop_submit_requested:
            return NoToolDecision(finish_reason="model_stop_no_accepted_submission")
        # One request to submit, even before any tool call, so a model that stops without an
        # accepted route still answers.
        state.forced_submit_requested = True
        state.model_stop_submit_requested = True
        return NoToolDecision(reminder="Call submit_pathway now with your best route.")
    if state.forced_submit_requested:
        return NoToolDecision(
            reminder=(
                "Your previous response did not call submit_pathway. "
                "Call submit_pathway now with your best route."
            )
        )
    if state.total_tool_calls <= 0:
        # Stopping is shared policy; adapters retain their historical reason labels.
        return NoToolDecision()

    state.consecutive_non_tool_turns_after_tools += 1
    persistence_suffix = ""
    if is_below_persistence_floor(
        state.best_accepted_similarity,
        config.stop_condition,
    ):
        persistence_suffix = f" {NON_TOOL_PERSISTENCE_GUARANTEE_SENTENCE}"
    if (
        config.force_submit_after_non_tool_turns > 0
        and state.consecutive_non_tool_turns_after_tools
        >= config.force_submit_after_non_tool_turns
    ):
        state.forced_submit_requested = True
        reminder = (
            f"Call submit_pathway now with your best route.{persistence_suffix}"
        )
    else:
        reminder = (
            "Continue. Use additional tools if they are useful, and finish by calling "
            f"submit_pathway when the route is ready.{persistence_suffix}"
        )
    return NoToolDecision(reminder=reminder)


def prepare_tool_call(
    *,
    raw_name: Any,
    raw_arguments: Any,
    raw_tool_call: dict[str, Any],
    event_index: int,
) -> PreparedToolCall:
    name = clean_tool_name(raw_name)
    argument_error: str | None = None
    try:
        arguments = parse_arguments(raw_arguments)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        arguments = {}
        argument_error = f"Could not parse tool arguments as a JSON object: {exc}"
    dispatch_arguments = dict(arguments)
    ignored_target_smiles = None
    ignored_final_smiles = None
    if name == "submit_pathway":
        ignored_target_smiles = dispatch_arguments.pop("target_smiles", None)
        ignored_final_smiles = dispatch_arguments.pop("final_smiles", None)
    dispatch_arguments, argument_coercions = coerce_tool_arguments(
        name, dispatch_arguments
    )
    dispatch_arguments, defaults_applied, ignored_arguments = (
        normalize_dispatch_arguments(name, dispatch_arguments)
    )
    call_id = str(raw_tool_call.get("id") or f"tool-call-{event_index}")
    return PreparedToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        dispatch_arguments=dispatch_arguments,
        argument_error=argument_error,
        event_payload={
            "arguments": arguments,
            "dispatch_arguments": dispatch_arguments,
            "ignored_target_smiles": ignored_target_smiles,
            "ignored_final_smiles": ignored_final_smiles,
            "ignored_arguments": ignored_arguments,
            "defaults_applied": defaults_applied,
            "argument_coercions": argument_coercions,
            "argument_error": argument_error,
            "raw_tool_call": raw_tool_call,
        },
    )


def handle_tool_result(
    config: RolloutPolicyConfig,
    state: RolloutPolicyState,
    *,
    name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    target_smiles: str,
) -> ToolResultDecision:
    """Apply one result in model-emitted order and decide repair or termination."""

    state.consecutive_non_tool_turns_after_tools = 0
    state.total_tool_calls += 1
    tool_budget_exhausted = (
        config.max_tool_calls > 0 and state.total_tool_calls >= config.max_tool_calls
    )
    if name != "submit_pathway":
        # When the model ends the rollout, the tool's own error is the whole response.
        if is_retryable_tool_error(result) and not config.model_stops:
            issues = "; ".join(str(issue) for issue in result.get("issues", []))
            return ToolResultDecision(
                repair=(
                    f"The {name or 'unknown'} tool call failed: {issues}. "
                    "Correct the arguments and call an appropriate tool again."
                )
            )
        return ToolResultDecision()

    state.last_submission = result
    state.last_submission_arguments = dict(arguments)
    accepted = bool(result.get("accepted"))
    signature = pathway_submission_signature(arguments)
    if accepted and signature:
        if signature == state.repeated_submission_signature:
            state.repeated_submission_count += 1
        else:
            state.repeated_submission_signature = signature
            state.repeated_submission_count = 1
    elif not accepted:
        state.repeated_submission_signature = None
        state.repeated_submission_count = 0

    similarity = submission_similarity_to_target(result, target_smiles)
    identical = submission_identical_to_target(result, target_smiles) is True
    if accepted and (
        state.best_accepted_submission is None
        or is_better_accepted_submission(
            identical, similarity, state.best_accepted_identical, state.best_accepted_similarity
        )
    ):
        state.best_accepted_submission = result
        # Record the template each step was grounded to, so a reaction SMARTS keeps its ID.
        steps = [
            {**step, "reaction_id": checked.get("reaction_id", step.get("reaction_id"))}
            for step, checked in zip(arguments.get("steps") or [], result.get("steps") or [])
        ]
        state.best_accepted_submission_arguments = {**arguments, **({"steps": steps} if steps else {})}
        state.best_accepted_similarity = similarity
        state.best_accepted_identical = identical
    meets_floor = meets_submission_floor(
        similarity,
        config.stop_condition,
        identical_to_target=identical,
    )
    if accepted and meets_floor and not config.model_stops:
        return ToolResultDecision(
            terminal_submission=result,
            meets_floor=True,
        )
    if accepted:
        if not meets_floor:
            state.low_similarity_submissions += 1
        if state.tool_limit_finalization_requested or tool_budget_exhausted:
            return ToolResultDecision(
                terminal_submission=state.best_accepted_submission,
                terminal_due_to_tool_limit=True,
            )
        if (
            config.max_low_similarity_submissions is not None
            and state.low_similarity_submissions
            >= max(1, config.max_low_similarity_submissions)
        ):
            return ToolResultDecision(
                terminal_submission=state.best_accepted_submission,
            )
        if config.model_stops:
            # The validator's result reports acceptance, similarity and exact identity.
            state.forced_submit_requested = False
            state.consecutive_non_tool_turns_after_tools = 0
            return ToolResultDecision()
        finish_clause = ""
        if (
            config.same_pathway_finish_after > 0
            and state.repeated_submission_count >= config.same_pathway_finish_after
            and state.best_accepted_submission is not None
        ):
            finish_clause = (
                " If you absolutely cannot improve on a previously submitted pathway, "
                "finish by responding with <FINISH>."
            )
        if config.stop_condition == "exact" and similarity == 1.0:
            lead = (
                "The validator accepted that pathway, but its product did not establish an "
                "exact match to the target under Standard InChI identity. "
                f"Its similarity_to_target was {similarity}; even 1.0 does not establish "
                f"molecular identity. {PERSISTENCE_GUARANTEE_SENTENCE} "
                "Try to identify an exact synthesis."
            )
            goal = "an exact synthesis route"
        else:
            lead = (
                "The validator accepted that pathway, but its similarity_to_target was "
                f"{similarity}. {PERSISTENCE_GUARANTEE_SENTENCE} "
                "Try to improve the analogue or identify an exact synthesis."
            )
            goal = "a higher-similarity route"
        repair = (
            f"{lead} Consider near building-block analogues. Compare the similarity of candidate products to the target, "
            f"and verify {goal} before submitting again.{finish_clause}"
        )
    else:
        state.invalid_submissions += 1
        if state.tool_limit_finalization_requested or tool_budget_exhausted:
            state.tool_limit_finalization_requested = True
            return ToolResultDecision(
                terminal_submission=state.best_accepted_submission,
                terminal_due_to_tool_limit=True,
            )
        error_message = result.get("error_message") or json.dumps(
            result.get("issues", []), sort_keys=True
        )
        # When the model ends the rollout, the rejected result's error_message is the whole response.
        repair = (
            f"Invalid submission #{state.invalid_submissions}: {error_message} "
            "Repair it using the tools. Empty steps are valid only with a grounded building_block. "
            "Verify each step's claimed product, then submit a verified pathway or direct-purchase "
            "building block."
        ) if not config.model_stops else None
    state.forced_submit_requested = False
    state.consecutive_non_tool_turns_after_tools = 0
    return ToolResultDecision(repair=repair)


def tool_limit_finish_reason(state: RolloutPolicyState) -> str:
    return (
        "tool_limit_best_accepted_submission"
        if state.best_accepted_submission is not None
        else "tool_limit_no_submission"
    )


def response_or_chat_tool_choice(
    *,
    api_mode: str,
    forced_submit_requested: bool,
    has_submission: bool,
    tools: list[dict[str, Any]],
) -> Any:
    if (
        not forced_submit_requested
        or has_submission
        or not tool_schema_contains_name(tools, "submit_pathway")
    ):
        return "auto"
    if api_mode == "responses":
        return {"type": "function", "name": "submit_pathway"}
    if api_mode == "chat":
        return {"type": "function", "function": {"name": "submit_pathway"}}
    return "auto"


def tool_schema_contains_name(tools: list[dict[str, Any]], name: str) -> bool:
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            return True
        if tool.get("name") == name:
            return True
    return False


def pathway_submission_signature(arguments: dict[str, Any] | None) -> str | None:
    if not isinstance(arguments, dict):
        return None
    building_block = arguments.get("building_block")
    steps = arguments.get("steps")
    if building_block:
        normalized = {"building_block": str(building_block).strip(), "steps": []}
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    if not isinstance(steps, list):
        return None
    normalized_steps: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            return None
        reactants = step.get("reactants")
        if not isinstance(reactants, list):
            return None
        normalized_steps.append(
            {
                "product_smiles": str(step.get("product_smiles") or "").strip(),
                "reactants": [str(reactant).strip() for reactant in reactants],
                "reaction_id": str(step.get("reaction_id") or "").strip(),
            }
        )
    return json.dumps({"steps": normalized_steps}, sort_keys=True, separators=(",", ":"))


def parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("tool arguments must decode to a JSON object")
    return decoded


def clean_tool_name(name: Any) -> str:
    text = "" if name is None else str(name)
    return text.split("<|", 1)[0]


def coerce_tool_arguments(
    name: str, arguments: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coerced = dict(arguments)
    coercions: list[dict[str, Any]] = []
    if name == "simulate_reaction":
        _coerce_json_string_array_field(coerced, "reactants", "reactants", coercions)
    elif name == "simulate_reverse_reaction":
        _coerce_json_string_array_field(
            coerced, "product_smiles", "product_smiles", coercions
        )
    elif name == "search_reaction_templates":
        _coerce_json_string_array_field(
            coerced, "reactant_smiles", "reactant_smiles", coercions
        )
        _coerce_json_string_array_field(
            coerced, "product_smiles", "product_smiles", coercions
        )
    elif name == "submit_pathway":
        _coerce_json_string_object_array_field(coerced, "steps", "steps", coercions)
        steps = coerced.get("steps")
        if isinstance(steps, list):
            copied_steps: list[Any] = []
            changed = False
            for index, step in enumerate(steps):
                if isinstance(step, dict):
                    copied_step = dict(step)
                    before = copied_step.get("reactants")
                    _coerce_json_string_array_field(
                        copied_step,
                        "reactants",
                        f"steps[{index}].reactants",
                        coercions,
                    )
                    changed = changed or copied_step.get("reactants") is not before
                    copied_steps.append(copied_step)
                else:
                    copied_steps.append(step)
            if changed:
                coerced["steps"] = copied_steps
    return coerced, coercions


def _coerce_json_string_object_array_field(
    arguments: dict[str, Any],
    key: str,
    field_path: str,
    coercions: list[dict[str, Any]],
) -> None:
    value = arguments.get(key)
    if not isinstance(value, str):
        return
    text = value.strip()
    if not text.startswith("["):
        return
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        return
    arguments[key] = parsed
    coercions.append(
        {
            "field": field_path,
            "coercion": "json_string_to_object_array",
            "raw_value": value,
            "coerced_value": parsed,
        }
    )


def _coerce_json_string_array_field(
    arguments: dict[str, Any],
    key: str,
    field_path: str,
    coercions: list[dict[str, Any]],
) -> None:
    value = arguments.get(key)
    if not isinstance(value, str):
        return
    text = value.strip()
    if not text.startswith("["):
        return
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return
    arguments[key] = parsed
    coercions.append(
        {
            "field": field_path,
            "coercion": "json_string_to_array",
            "raw_value": value,
            "coerced_value": parsed,
        }
    )


def tool_argument_error(
    name: str, message: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "error": "invalid_tool_call",
        "tool_name": name,
        "issues": [message],
        "received_arguments": arguments,
        "retryable": True,
    }
    if name == "submit_pathway":
        result["accepted"] = False
        result["error_message"] = (
            f"The submit_pathway tool call was rejected before validation: {message}."
        )
    return result


def is_retryable_tool_error(result: dict[str, Any]) -> bool:
    return bool(result.get("retryable") and result.get("error"))


def is_better_submission_similarity(
    candidate: float | None, incumbent: float | None
) -> bool:
    if incumbent is None:
        return True
    if candidate is None:
        return False
    return float(candidate) > float(incumbent)


def is_better_accepted_submission(
    identical: bool,
    similarity: float | None,
    best_identical: bool | None,
    best_similarity: float | None,
) -> bool:
    """Whether an accepted submission beats the best so far; an exact match beats any analogue."""

    best_identical = best_identical is True
    return (identical and not best_identical) or (
        identical == best_identical and is_better_submission_similarity(similarity, best_similarity)
    )


def is_below_persistence_floor(
    similarity: float | None,
    stop_condition: StopCondition,
) -> bool:
    # An identity match already terminates the rollout. A continuing exact-mode
    # rollout therefore still needs improvement, even at fingerprint similarity 1.
    if stop_condition == "exact":
        return similarity is not None
    return (
        float(stop_condition) > 0
        and similarity is not None
        and float(similarity) < float(stop_condition)
    )


def submission_identical_to_target(
    submission: dict[str, Any] | None,
    target_smiles: str | None,
) -> bool | None:
    """Standard InChI identity between an accepted submission's product and the target."""

    if not submission or submission.get("accepted") is not True or not target_smiles:
        return None
    # Recompute against this target; cached booleans may describe a different
    # target or identity policy, or have been deserialized as truthy strings.
    final_smiles = submission.get("actual_final_smiles") or submission.get("final_smiles")
    if not final_smiles:
        return None
    return chem.molecules_identical(str(target_smiles), str(final_smiles))


def _submission_fingerprint_similarity(
    submission: dict[str, Any] | None,
    target_smiles: str | None,
    score: str,
) -> float | None:
    """Return a stored or recomputed fingerprint score (``constitutional`` or ``stereo_aware``)."""

    if not submission or submission.get("accepted") is not True or not target_smiles:
        return None
    value = submission.get(f"{score}_similarity_to_target")
    if value is not None:
        return float(value)
    final_smiles = submission.get("actual_final_smiles") or submission.get("final_smiles")
    if not final_smiles:
        return None
    result = chem.similarity(str(target_smiles), str(final_smiles))
    if not result.get("valid"):
        return None
    return result.get(f"{score}_similarity")


def submission_constitutional_similarity_to_target(
    submission: dict[str, Any] | None,
    target_smiles: str | None,
) -> float | None:
    return _submission_fingerprint_similarity(submission, target_smiles, "constitutional")


def submission_similarity_to_target(
    submission: dict[str, Any] | None,
    target_smiles: str | None,
) -> float | None:
    """The active similarity: stereo-aware Morgan/Tanimoto."""

    return _submission_fingerprint_similarity(submission, target_smiles, "stereo_aware")


def submission_summary_fields(
    submission: dict[str, Any] | None,
    target_smiles: str,
) -> dict[str, Any]:
    """Build the submission fields shared by every rollout summary transport."""

    similarity = submission_similarity_to_target(submission, target_smiles)
    return {
        "submitted": submission is not None,
        "accepted": bool(submission and submission.get("accepted") is True),
        "similarity_to_target": similarity,
        "identical_to_target": submission_identical_to_target(submission, target_smiles),
        "constitutional_similarity_to_target": submission_constitutional_similarity_to_target(
            submission, target_smiles
        ),
        "stereo_aware_similarity_to_target": similarity,
        "pathway_type": submission.get("pathway_type") if submission else None,
        "final_smiles": (
            submission.get("final_smiles")
            if submission and submission.get("accepted") is True
            else None
        ),
        "reaction_validation": (
            submission.get("reaction_validation") if submission else None
        ),
        "building_block_grounded": (
            submission.get("building_block_grounded") if submission else None
        ),
    }


def append_selected_submission_event(
    trace_events: list[dict[str, Any]],
    *,
    task: dict[str, Any],
    submission: dict[str, Any] | None,
    submission_arguments: dict[str, Any] | None = None,
    reason: str,
) -> None:
    if submission is None:
        return
    model_result = compact_tool_result("submit_pathway", submission)
    summary = submission_summary_fields(submission, task["target_smiles"])
    trace_events.append(
        {
            "event_type": "selected_submission",
            "tool_name": "submit_pathway",
            "target_smiles": task["target_smiles"],
            "payload": {
                "reason": reason,
                "similarity_to_target": summary["similarity_to_target"],
                "identical_to_target": summary["identical_to_target"],
                "constitutional_similarity_to_target": summary[
                    "constitutional_similarity_to_target"
                ],
                "stereo_aware_similarity_to_target": summary[
                    "stereo_aware_similarity_to_target"
                ],
                "submission_arguments": submission_arguments,
                "result": submission,
                "model_result": model_result,
                "model_result_compaction": {
                    "full_json_bytes": compact_json_size(submission),
                    "model_json_bytes": compact_json_size(model_result),
                },
            },
        }
    )


def request_summary_fields(args: Any, config: RolloutPolicyConfig) -> dict[str, Any]:
    """Return the generation and harness settings needed to reproduce a run.

    Budgets come from ``config``, the task's effective settings, not the runner's defaults.
    """

    return {
        "responses_state_mode": getattr(args, "responses_state_mode", None),
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "temperature": getattr(args, "temperature", None),
        "top_p": getattr(args, "top_p", None),
        "sampling_seed": getattr(args, "sampling_seed", None),
        "max_output_tokens": config.max_tokens,
        "max_model_len": config.max_model_len,
        "context_limit_stated": config.context_limit_stated,
        "context_safety_margin_fraction": config.context_safety_margin_fraction,
        "context_safety_margin_tokens": config.context_safety_margin_tokens,
        "token_budget_feedback": config.token_budget_feedback,
        "stop_condition": config.stop_condition,
        "max_turns": config.max_turns,
        "max_tool_calls": config.max_tool_calls,
        "reaction_library": getattr(args, "reaction_library", None),
    }
