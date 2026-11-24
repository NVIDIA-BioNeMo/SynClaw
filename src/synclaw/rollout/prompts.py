# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical prompts for every BioNeMo-SynClaw rollout transport."""

from __future__ import annotations

import hashlib
from typing import Any

from synclaw.rollout.budget import limit_sentences
from synclaw.rollout.policy import RolloutPolicyConfig

DEFAULT_PROMPT_TEMPLATE = "Target molecule: {target_smiles}"

DEFAULT_SYSTEM_PROMPT = (
    "You are performing synthesis planning. Return a proposed synthesis pathway for the exact "
    "target when possible. If exact synthesis is not found, return the highest-similarity "
    "analogue you can identify. Submit pathways with submit_pathway, which you may call more "
    "than once; only your best accepted pathway counts. Each must either contain at least one "
    "reaction step that the validator can reproduce, or a single valid building_block for "
    "direct purchase."
)

# Stored on every task row, so a taskset built from older prompt text is refused.
PROMPT_FINGERPRINT = hashlib.sha256(
    "\0".join((DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_PROMPT)).encode("utf-8")
).hexdigest()[:12]


def require_current_prompt_text(task: dict[str, Any]) -> dict[str, Any]:
    """Refuse a loaded taskset whose prompt text predates this build.

    Callers that supply their own prompt text own it and are not checked.
    """

    recorded = task.get("prompt_fingerprint")
    if recorded == PROMPT_FINGERPRINT:
        return task
    raise RuntimeError(
        f"Task {task.get('task_id', '<unknown>')!r} carries prompt text "
        f"{recorded or 'with no prompt fingerprint'}, but this build "
        f"ships {PROMPT_FINGERPRINT}. Regenerate the taskset with `synclaw make-taskset`."
    )


def _plural(count: int) -> str:
    return "" if count == 1 else "s"


def build_initial_messages(
    *,
    task: dict[str, Any],
    config: RolloutPolicyConfig | None = None,
) -> list[dict[str, Any]]:
    """Build the system and user messages; tool-specific guidance lives in the tool descriptions.

    The budget paragraph states the limits of ``config``, the task's effective settings.
    An unlimited or unstated limit is left out, and with none the paragraph is omitted.
    """

    config = config or RolloutPolicyConfig()
    limits = []
    if config.max_tool_calls > 0:
        limits.append(f"{config.max_tool_calls} total tool call{_plural(config.max_tool_calls)}")
    if config.max_turns is not None:
        limits.append(f"{config.max_turns} assistant turn{_plural(config.max_turns)}")
    budget = [
        *([f"Use no more than {' or '.join(limits)}."] if limits else []),
        *limit_sentences(config.stated_max_model_len, config.max_tokens),
    ]
    harness_content = str(task["system_prompt"]) + ("\n\n" + " ".join(budget) if budget else "")
    user_content = str(task["question"])
    return [
        {"role": "system", "content": harness_content},
        {"role": "user", "content": user_content},
    ]


__all__ = [
    "build_initial_messages",
    "require_current_prompt_text",
]
