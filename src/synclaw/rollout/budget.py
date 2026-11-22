# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context accounting and rollout feedback, independent of model transports.

``_feedback_text`` below emits BioNeMo-SynClaw-chemistry-specific guidance (``submit_pathway``,
``building_block``) once context runs low; it is not domain-independent.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

from synclaw.token_usage import int_or_none, normalize_token_usage

if TYPE_CHECKING:
    from synclaw.rollout.policy import RolloutPolicyConfig

# Reserved for the reminder itself. The longest variant is about 130 tokens, or about
# 215 at the uncalibrated rate; this bound covers both.
FEEDBACK_ALLOWANCE_TOKENS = 256


@dataclass(frozen=True, slots=True)
class ContextBudgetSnapshot:
    """An adapter's assessment of the next request, never a billing total.

    ``input_tokens`` includes tools and retained conversation state. ``None``
    means unknown. An adapter with a real tokenizer may set ``estimated=False``;
    output capacity is a per-response cap, not a decrementing session balance.
    """

    context_window_tokens: int | None = None
    input_tokens: int | None = None
    max_output_tokens: int | None = None
    source: str = "unavailable"
    estimated: bool = True

    def __post_init__(self) -> None:
        for name in ("context_window_tokens", "input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            minimum = 0 if name == "input_tokens" else 1
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}, or None")

    @property
    def remaining_tokens(self) -> int | None:
        if self.context_window_tokens is None or self.input_tokens is None:
            return None
        return max(0, self.context_window_tokens - self.input_tokens)

    def trace_fields(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "remaining_tokens": self.remaining_tokens,
            "request_output_tokens": self.max_output_tokens,
        }


BudgetProvider = Callable[
    [Sequence[dict[str, Any]], Sequence[dict[str, Any]]], ContextBudgetSnapshot
]


def serialized_input_bytes(payload: Any) -> int:
    """Measure model-facing input only, using one stable JSON representation."""

    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


_OPAQUE_INPUT_TYPES = frozenset(
    {
        "image_url",
        "input_image",
        "input_audio",
        "image",
        "document",
        "item_reference",
        "redacted_thinking",
    }
)


def _opaque_input(value: Any) -> bool:
    """Whether input holds content whose token cost cannot be estimated from its JSON bytes."""

    if isinstance(value, dict):
        kind = value.get("type")
        if value.get("encrypted_content") or (
            isinstance(kind, str) and kind in _OPAQUE_INPUT_TYPES
        ):
            return True
        # A Responses reasoning summary is not the hidden reasoning it references.
        if kind == "reasoning" and value.get("id"):
            return True
        return any(_opaque_input(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_opaque_input(item) for item in value)
    return False


def _replay_prefix_digest(payload: dict[str, Any], length: int) -> str:
    """Identify retained Responses history and schemas across JSON checkpoints."""

    prefix = {**payload, "input": payload["input"][:length]}
    serialized = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ContextTokenEstimator:
    """Usage-calibrated UTF-8 estimate, with anchors for retained opaque state.

    No tokenizer or provider round trip is required. Calibration is approximate:
    text composition and chat-template overhead can change between requests.
    Usage anchors assume reported output remains in context until the next usage
    report; provider-side compaction can make that overestimate. Full Responses
    replay verifies the retained prefix and estimates only appended readable input.
    """

    tokens_per_byte: float = 1 / 3
    calibrated: bool = False
    continuation_tokens: int | None = None
    replay_prefix_length: int = 0
    replay_prefix_digest: str | None = None

    def observe(
        self,
        payload: Any,
        usage: dict[str, Any] | None,
        *,
        server_managed: bool = False,
        separate_cache_usage: bool = False,
        replay_output: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        normalized = normalize_token_usage(usage)
        input_tokens = normalized.get("input_tokens")
        output_tokens = normalized.get("output_tokens")
        if input_tokens is not None and separate_cache_usage:
            # Anthropic's input_tokens excludes both cache categories. OpenAI's
            # cached_tokens is already a subset of input_tokens and is not added.
            input_tokens += normalized.get("cache_creation_input_tokens") or 0
            input_tokens += normalized.get("cache_read_input_tokens") or 0
        if not server_managed and input_tokens and not _opaque_input(payload):
            observed_rate = input_tokens / max(1, serialized_input_bytes(payload))
            # Equal-weight the new observation against the running rate so one
            # atypical turn (e.g. a SMILES-dense tool result) nudges the estimate
            # rather than fully overwriting it for the rest of the rollout.
            self.tokens_per_byte = (
                observed_rate if not self.calibrated else (self.tokens_per_byte + observed_rate) / 2
            )
            self.calibrated = True
        self.continuation_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        self.replay_prefix_length = 0
        self.replay_prefix_digest = None
        if (
            not server_managed
            and self.continuation_tokens is not None
            and replay_output is not None
            and isinstance(payload, dict)
            and isinstance(payload.get("input"), list)
        ):
            retained = {**payload, "input": [*payload["input"], *replay_output]}
            self.replay_prefix_length = len(retained["input"])
            self.replay_prefix_digest = _replay_prefix_digest(retained, self.replay_prefix_length)

    def snapshot(
        self,
        payload: Any,
        *,
        context_window_tokens: int | None,
        max_output_tokens: int | None,
        server_managed: bool = False,
    ) -> ContextBudgetSnapshot:
        occupied = None
        source = "unavailable"
        opaque = _opaque_input(payload)
        if not opaque and (not server_managed or self.continuation_tokens is not None):
            occupied = math.ceil(serialized_input_bytes(payload) * self.tokens_per_byte)
            source = "provider_calibrated_utf8" if self.calibrated else "utf8_estimate"
            if server_managed:
                occupied += self.continuation_tokens or 0
                source = "provider_usage_plus_new_input"
        elif opaque and not server_managed:
            occupied = self._replay_input_tokens(payload)
            if occupied is not None:
                source = "provider_usage_plus_replay_delta"
        return ContextBudgetSnapshot(
            context_window_tokens=context_window_tokens,
            input_tokens=occupied,
            max_output_tokens=max_output_tokens,
            source=source,
        )

    def _replay_input_tokens(self, payload: Any) -> int | None:
        if (
            self.continuation_tokens is None
            or self.replay_prefix_digest is None
            or not isinstance(payload, dict)
            or not isinstance(payload.get("input"), list)
            or len(payload["input"]) < self.replay_prefix_length
            or _replay_prefix_digest(payload, self.replay_prefix_length)
            != self.replay_prefix_digest
        ):
            return None
        appended = payload["input"][self.replay_prefix_length :]
        if _opaque_input(appended):
            return None
        # Count only the appended JSON list contents and its separator from the
        # retained prefix. Ciphertext, tool schemas and prior output are covered
        # by usage; reasoning tokens are already included in output_tokens.
        new_bytes = serialized_input_bytes(appended) - 2
        if appended and self.replay_prefix_length:
            new_bytes += 1
        return self.continuation_tokens + math.ceil(new_bytes * self.tokens_per_byte)

    def checkpoint_fields(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, value: dict[str, Any] | None) -> ContextTokenEstimator:
        if not value:
            return cls()
        rate = float(value.get("tokens_per_byte", 1 / 3))
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Invalid checkpoint token estimation rate")
        return cls(
            tokens_per_byte=rate,
            calibrated=bool(value.get("calibrated")),
            continuation_tokens=int_or_none(value.get("continuation_tokens")),
            replay_prefix_length=int(value.get("replay_prefix_length") or 0),
            replay_prefix_digest=value.get("replay_prefix_digest"),
        )


@dataclass(slots=True)
class BudgetFeedbackState:
    announced_context_limit: int | None = None
    announced_output_limit: int | None = None
    last_turn: int | None = None
    context_warning_sent: bool = False

    def mark_reported(
        self,
        snapshot: ContextBudgetSnapshot,
        content: str | None,
        *,
        turn_index: int,
        context_warning_remaining_fraction: float,
    ) -> None:
        """Record a turn's assessment and, if ``content`` was sent, the limits and warning it carried."""

        self.last_turn = turn_index
        if content:
            self.announced_context_limit = snapshot.context_window_tokens
            self.announced_output_limit = snapshot.max_output_tokens
            if _remaining_at_most(snapshot, context_warning_remaining_fraction):
                self.context_warning_sent = True

    def checkpoint_fields(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, value: dict[str, Any] | None) -> BudgetFeedbackState:
        return cls(**(value or {}))


def _remaining_at_most(snapshot: ContextBudgetSnapshot, fraction: float) -> bool:
    remaining = snapshot.remaining_tokens
    limit = snapshot.context_window_tokens
    return remaining is not None and limit is not None and remaining <= limit * fraction


def limit_sentences(context_window_tokens: int | None, max_output_tokens: int | None) -> list[str]:
    """State the context and per-response output limits; an unstated limit is left out."""

    sentences = []
    if context_window_tokens is not None:
        sentences.append(
            f"Your context-window budget is {context_window_tokens} tokens, shared by "
            "the conversation, tool results and subsequent output."
        )
    if max_output_tokens is not None:
        sentences.append(
            f"Each response is capped at {max_output_tokens} output tokens, including "
            "reasoning. This cap resets for each response."
        )
    return sentences


def _feedback_text(
    snapshot: ContextBudgetSnapshot,
    state: BudgetFeedbackState,
    *,
    turns_remaining: int | None = None,
    tool_calls_remaining: int | None = None,
    context_warning_remaining_fraction: float = 0.5,
    context_critical_remaining_fraction: float = 0.25,
) -> str:
    # A limit is restated only if it differs from the one the model was told.
    parts = limit_sentences(
        snapshot.context_window_tokens
        if snapshot.context_window_tokens != state.announced_context_limit
        else None,
        snapshot.max_output_tokens
        if snapshot.max_output_tokens != state.announced_output_limit
        else None,
    )
    remaining = [
        f"{count} {noun}{'' if count == 1 else 's'}"
        for count, noun in ((turns_remaining, "turn"), (tool_calls_remaining, "tool call"))
        if count is not None
    ]
    remaining_tokens = snapshot.remaining_tokens
    if remaining_tokens is not None:
        count = (
            f"~{_rounded_tokens(remaining_tokens)}" if snapshot.estimated else str(remaining_tokens)
        )
        remaining.append(f"{count} context tokens")
    if remaining:
        parts.append("Remaining: " + ", ".join(remaining) + ".")
    if _remaining_at_most(snapshot, context_critical_remaining_fraction):
        parts.append(
            "Your context window is nearly exhausted. You must call submit_pathway now "
            "with your best verified route, even if it is an analogue, so it can be scored. "
            "Avoid further exploration; use the evidence already gathered."
        )
    elif (
        _remaining_at_most(snapshot, context_warning_remaining_fraction)
        and not state.context_warning_sent
    ):
        parts.append(
            "Context is running low. Call submit_pathway now with your best verified route, "
            "even if it is an analogue, so it can be scored. Use any remaining tools only "
            "to verify a clearly better candidate."
        )
    return " ".join(parts)


def _rounded_tokens(count: int) -> str:
    # An estimate is no more precise than thousands; round down so headroom is never overstated.
    return f"{count // 1000}k" if count >= 1000 else str(count)


def with_harness_feedback(
    messages: Sequence[dict[str, Any]],
    content: str,
    *,
    field: str = "budget",
) -> list[dict[str, Any]] | None:
    """Attach feedback to pending input without creating another user turn.

    A new user message can make a provider discard prior reasoning. Keep tool
    results as tool results, including native function outputs/thinking blocks
    in transport adapters. Never rewrite an assistant message to add feedback.
    """
    if not messages or messages[-1].get("role") not in {"user", "tool"}:
        if any(m.get("role") == "assistant" for m in messages):
            return None
        return [*messages, {"role": "user", "content": content}]
    last = messages[-1]
    original = str(last.get("content") or "")
    if last["role"] == "tool":
        # BioNeMo-SynClaw tool results are JSON objects. Keep them parseable and distinguish
        # harness guidance from the actual chemistry result, under field names
        # that do not mention the harness, since the model reads them.
        result = json.loads(original)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object for a BioNeMo-SynClaw tool result")
        previous = result.get(field)
        result[field] = f"{previous}\n\n{content}" if previous else content
        updated = json.dumps(result, sort_keys=True, separators=(",", ":"))
    else:
        updated = original + "\n\n" + content
    return [*messages[:-1], {**last, "content": updated}]


def initialize_budget_feedback(config: RolloutPolicyConfig) -> BudgetFeedbackState:
    """Start feedback state for a fresh rollout whose system prompt stated ``config``'s limits.

    ``build_initial_messages`` states them, so progress feedback restates a limit only
    if it changes. Resume existing runs from their saved messages and state instead.
    """

    return BudgetFeedbackState(
        announced_context_limit=config.stated_max_model_len,
        announced_output_limit=config.max_tokens,
    )


def prepare_budget_feedback(
    provider: BudgetProvider,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    state: BudgetFeedbackState,
    *,
    turn_index: int,
    turns_remaining: int | None = None,
    tool_calls_remaining: int | None = None,
    context_warning_remaining_fraction: float = 0.5,
    context_critical_remaining_fraction: float = 0.25,
) -> tuple[str | None, ContextBudgetSnapshot | None, str]:
    """Assess the pending input once and render advisory feedback without mutating state.

    Remaining turns include the upcoming response; omit disabled tool-call limits with None.
    """

    if state.last_turn == turn_index:
        return None, None, "already_reported"
    snapshot = provider(messages, tools)
    if snapshot.input_tokens is not None:
        snapshot = replace(snapshot, input_tokens=snapshot.input_tokens + FEEDBACK_ALLOWANCE_TOKENS)
    content = _feedback_text(
        snapshot,
        state,
        turns_remaining=turns_remaining,
        tool_calls_remaining=tool_calls_remaining,
        context_warning_remaining_fraction=context_warning_remaining_fraction,
        context_critical_remaining_fraction=context_critical_remaining_fraction,
    )
    if not content:
        return None, snapshot, "unavailable_or_unchanged"
    if with_harness_feedback(messages, content) is None:
        return None, snapshot, "no_pending_input"
    return content, snapshot, "emitted"


def budget_summary_fields(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Expose the last request's assessed budget without mixing it with billing.

    ``request_budget`` events record every request, whether or not progress
    feedback was shown, so they are preferred over ``token_budget`` events.
    """

    for kind in ("request_budget", "token_budget"):
        for event in reversed(events):
            if event.get("event_type") == kind:
                payload = event["payload"]
                return {
                    "last_context_input_tokens": payload["input_tokens"],
                    "last_context_remaining_tokens": payload["remaining_tokens"],
                    "context_budget_source": payload["source"],
                    "context_budget_estimated": payload["estimated"],
                    "last_request_output_tokens": payload["request_output_tokens"],
                }
    return {}
