# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-usage accounting for hosted-model responses.

BioNeMo-SynClaw counts tokens; it never estimates prices. A provider that reports its own
``cost`` inside a response's usage (OpenRouter does) has that value carried
through and summed, but nothing here computes one.
"""

from __future__ import annotations

import json
from typing import Any

TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
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
        "cached_input_tokens": cache_read_input_tokens,
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
    """Token usage recorded on one assistant message payload in a trace."""

    response_usage = payload.get("response_usage")
    if isinstance(response_usage, dict):
        return normalize_token_usage(response_usage)
    usage = payload.get("token_usage")
    return usage if isinstance(usage, dict) else {}


def sum_token_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-response usage.

    ``cost`` is reported only when every response with usage carried one; a sum
    over some of them would understate the spend.
    """

    totals = {field: 0 for field in TOKEN_USAGE_FIELDS}
    seen_token_fields: set[str] = set()
    saw_usage = False
    saw_cost = False
    missing_cost = False
    numeric_cost_total = 0.0
    non_numeric_costs: list[Any] = []
    for usage in usages:
        if not usage:
            continue
        saw_usage = True
        for field in TOKEN_USAGE_FIELDS:
            value = int_or_none(usage.get(field))
            if value is not None:
                seen_token_fields.add(field)
                totals[field] += value
        if "cost" in usage:
            saw_cost = True
            cost = usage.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                numeric_cost_total += float(cost)
            else:
                non_numeric_costs.append(cost)
        else:
            missing_cost = True
    if not saw_usage:
        return {}
    summary: dict[str, Any] = {
        field: totals[field] for field in TOKEN_USAGE_FIELDS if field in seen_token_fields
    }
    if saw_cost and not missing_cost:
        if non_numeric_costs:
            summary["cost"] = json.dumps(non_numeric_costs, sort_keys=True)
        else:
            summary["cost"] = numeric_cost_total
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


def trace_token_usage(
    events: list[dict[str, Any]],
    *,
    run_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Token usage for a run: the run summary's totals when it has any, else summed from the trace."""

    if run_summary:
        summary_usage = {
            field: run_summary.get(field)
            for field in TOKEN_SUMMARY_FIELDS
            if run_summary.get(field) is not None
        }
        if summary_usage:
            return summary_usage
    return aggregate_token_usage_from_events(events)
