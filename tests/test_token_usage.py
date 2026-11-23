# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalizing provider usage objects and totalling them over a trace."""

from __future__ import annotations

import pytest

from synclaw.token_usage import (
    aggregate_token_usage_from_events,
    normalize_token_usage,
    sum_token_usage,
    usage_from_payload,
)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (  # Chat Completions
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 6},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "cache_read_input_tokens": 6,
                "reasoning_tokens": 2,
            },
        ),
        (  # Responses
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "input_tokens_details": {"cached_tokens": 6},
                "output_tokens_details": {"reasoning_tokens": 2},
            },
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_read_input_tokens": 6,
                "reasoning_tokens": 2,
            },
        ),
        (  # Messages, with a provider-reported cost
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 6,
                "cost": 0.01,
            },
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 6,
                "cost": 0.01,
            },
        ),
        (None, {}),
    ],
)
def test_each_api_shape_normalizes_onto_one_field_set(usage, expected) -> None:
    assert normalize_token_usage(usage) == expected


def test_cost_is_summed_only_when_every_response_reports_a_number() -> None:
    reported = [{"input_tokens": 1, "cost": 0.5}, {"input_tokens": 2, "cost": 0.25}]

    assert sum_token_usage(reported) == {"input_tokens": 3, "cost": 0.75}
    assert sum_token_usage([{"input_tokens": 1, "cost": 0.5}, {"input_tokens": 2}]) == {
        "input_tokens": 3
    }
    assert "cost" not in sum_token_usage([{"input_tokens": 1, "cost": "0.5"}])
    assert "cost" not in sum_token_usage([{"input_tokens": 1, "cost": True}])


def test_responses_without_usage_are_skipped() -> None:
    assert sum_token_usage([{}, {"output_tokens": 3}, {}]) == {"output_tokens": 3}
    assert sum_token_usage([{}, {}]) == {}


def test_trace_totals_come_from_the_assistant_messages_raw_usage() -> None:
    events = [
        {"event_type": "message", "role": "user", "payload": {"content": "plan"}},
        {
            "event_type": "message",
            "role": "assistant",
            "payload": {"response_usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        },
        {
            "event_type": "message",
            "role": "assistant",
            "payload": {"response_usage": {"input_tokens": 7, "output_tokens": 1}},
        },
        {"event_type": "tool_result", "payload": {"response_usage": {"input_tokens": 99}}},
    ]

    assert aggregate_token_usage_from_events(events) == {"input_tokens": 12, "output_tokens": 3}
    assert usage_from_payload({"content": "no usage recorded"}) == {}
