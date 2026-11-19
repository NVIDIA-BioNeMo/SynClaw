# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical prompts and task preparation for every BioNeMo-SynClaw rollout transport."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synclaw.core import db
from synclaw.core.libraries import LibrarySelection
from synclaw.rollout.budget import limit_sentences
from synclaw.rollout.policy import RolloutPolicyConfig
from synclaw.tasks.datasets import PROMPT_FINGERPRINT


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
    limits = [
        *([f"{config.max_tool_calls} total tool call{'' if config.max_tool_calls == 1 else 's'}"]
          if config.max_tool_calls > 0 else []),
        *([f"{config.max_turns} assistant turns"] if config.max_turns is not None else []),
    ]
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


def validate_model_run_database(db_path: str | Path, *, library: LibrarySelection = None) -> None:
    with db.session(db_path) as conn:
        db.require_current_reaction_schema(conn, db_path)
        building_blocks = conn.execute("SELECT COUNT(*) FROM building_blocks").fetchone()[0]
        reaction_templates = conn.execute("SELECT COUNT(*) FROM reaction_templates").fetchone()[0]
    if building_blocks <= 0 or reaction_templates <= 0:
        raise RuntimeError(
            f"Model-run database is not ready: {db_path} has "
            f"{building_blocks} building block(s) and {reaction_templates} reaction template(s). "
            "Run setup/ingestion first or pass --db pointing to a populated database."
        )
    db.require_prepared_building_block_search(db_path)
    with db.session(db_path) as conn:
        db.require_reaction_library(conn, library)
        # Fail once here rather than at the first query-less template search.
        db.reaction_template_library_overview(conn, library=library)


__all__ = [
    "build_initial_messages",
    "require_current_prompt_text",
    "validate_model_run_database",
]
