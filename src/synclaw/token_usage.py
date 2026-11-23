# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-usage accounting for hosted-model responses.

BioNeMo-SynClaw counts tokens; it never estimates prices. A provider that reports its own
``cost`` inside a response's usage (OpenRouter does) has that value carried
through and summed, but nothing here computes one.
"""

from __future__ import annotations

from typing import Any

TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_tokens",
)
TOKEN_SUMMARY_FIELDS = (*TOKEN_USAGE_FIELDS, "cost")


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_token_usage(usage: Any) -> dict[str, Any]:
    """Map Chat, Responses and Messages usage objects onto one field set."""

    if not isinstance(usage, dict):
        return {}
    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = usage.get("prompt_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    output_details = usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = usage.get("completion_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}
    cache_read_input_tokens = int_or_none(
        usage.get("cache_read_input_tokens", input_details.get("cached_tokens"))
    )
    values = {
        "input_tokens": int_or_none(usage.get("input_tokens", usage.get("prompt_tokens"))),
        "output_tokens": int_or_none(usage.get("output_tokens", usage.get("completion_tokens"))),
        "total_tokens": int_or_none(usage.get("total_tokens")),
        "cache_creation_input_tokens": int_or_none(
            usage.get("cache_creation_input_tokens", input_details.get("cache_creation_tokens"))
        ),
        "cache_read_input_tokens": cache_read_input_tokens,
        "reasoning_tokens": int_or_none(
            output_details.get("reasoning_tokens", output_details.get("thinking_tokens"))
        ),
        # Provider-reported only; never estimated.
        "cost": usage.get("cost"),
    }
    return {key: value for key, value in values.items() if value is not None}


def usage_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalized token usage of one assistant message payload in a trace."""

    return normalize_token_usage(payload.get("response_usage"))


def sum_token_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-response usage.

    ``cost`` is reported only when every response with usage carried a numeric one; a sum
    over some of them would understate the spend.
    """

    usages = [usage for usage in usages if usage]
    summary: dict[str, Any] = {}
    for field in TOKEN_USAGE_FIELDS:
        values = [v for usage in usages if (v := int_or_none(usage.get(field))) is not None]
        if values:
            summary[field] = sum(values)
    costs = [usage.get("cost") for usage in usages]
    if usages and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in costs):
        summary["cost"] = float(sum(costs))
    return summary


def aggregate_token_usage_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Total token usage over the assistant messages of a trace."""

    usages: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") != "message" or event.get("role") != "assistant":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            usages.append(usage_from_payload(payload))
    return sum_token_usage(usages)
