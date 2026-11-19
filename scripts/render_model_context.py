#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the exact initial model prompt and tool descriptions as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from synclaw.core.libraries import LibrarySelection
from synclaw.rollout.policy import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TURNS,
    DEFAULT_REACTION_LIBRARY,
    RolloutPolicyConfig,
    task_budget_config,
)
from synclaw.rollout.prompts import build_initial_messages
from synclaw.runners import model_runner
from synclaw.tools.contracts import (
    filter_chat_tool_schemas,
    filter_responses_tool_schemas,
    normalize_excluded_tools,
    task_excluded_tools,
)

DEFAULT_TASKSET = Path("artifacts/tasksets/chembl_llm_tasks.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/plots")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset", type=Path, default=DEFAULT_TASKSET)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument(
        "--db",
        type=Path,
        help="SQLite database. Defaults to var/synclaw.sqlite, as in synclaw run-model.",
    )
    parser.add_argument(
        "--reaction-library",
        nargs="+",
        default=DEFAULT_REACTION_LIBRARY,
    )
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--max-model-len", type=int, help="Context limit to state; omitted when unknown"
    )
    parser.add_argument(
        "--api-mode",
        choices=("chat", "responses"),
        default="chat",
        help="Tool-schema transport shape to render.",
    )
    parser.add_argument(
        "--exclude-tool",
        action="append",
        default=[],
        help="Tool name to hide from the model, on top of the task's own; may be repeated or comma-separated.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Markdown output path; defaults to artifacts/plots/<task_id>_model_context.md.",
    )
    args = parser.parse_args()

    try:
        markdown, default_output = render_model_context(
            taskset=args.taskset,
            task_index=args.task_index,
            db_path=args.db,
            reaction_library=args.reaction_library,
            max_tool_calls=args.max_tool_calls,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            api_mode=args.api_mode,
            exclude_tools=args.exclude_tool,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output or default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(output)
    return 0


def render_model_context(
    taskset: Path,
    task_index: int,
    db_path: Path | None,
    reaction_library: LibrarySelection,
    max_tool_calls: int,
    api_mode: str,
    exclude_tools: list[str],
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_model_len: int | None = None,
) -> tuple[str, Path]:
    task = model_runner.load_task(str(taskset), task_index)
    resolved_db = model_runner.resolve_task_db_path(db_path)
    excluded = task_excluded_tools(task, normalize_excluded_tools(exclude_tools))
    # The task's own budgets replace the runner's, as in a run.
    budgets = task_budget_config(
        task,
        RolloutPolicyConfig(
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
        ),
    )
    max_turns, max_tool_calls = budgets.max_turns, budgets.max_tool_calls
    messages = build_initial_messages(task=task, config=budgets)
    tool_schemas = selected_tool_schemas(api_mode=api_mode, excluded_tools=excluded)
    task_id = str(task.get("task_id") or f"task-{task_index:04d}")
    default_output = DEFAULT_OUTPUT_DIR / f"{slugify(task_id)}_model_context.md"
    markdown = render_markdown(
        task=task,
        taskset=taskset,
        task_index=task_index,
        db_path=resolved_db,
        reaction_library=reaction_library,
        max_tool_calls=max_tool_calls,
        max_turns=max_turns,
        api_mode=api_mode,
        excluded_tools=excluded,
        messages=messages,
        tool_schemas=tool_schemas,
    )
    return markdown, default_output


def selected_tool_schemas(api_mode: str, excluded_tools: set[str]) -> list[dict[str, Any]]:
    if api_mode == "responses":
        return filter_responses_tool_schemas(excluded_tools)
    return filter_chat_tool_schemas(excluded_tools)


def render_markdown(
    task: dict[str, Any],
    taskset: Path,
    task_index: int,
    db_path: Path,
    reaction_library: LibrarySelection,
    max_tool_calls: int,
    api_mode: str,
    excluded_tools: set[str],
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    max_turns: int = DEFAULT_MAX_TURNS,
) -> str:
    lines: list[str] = [
        f"# Model Context: {task.get('task_id', f'task-{task_index:04d}')}",
        "",
        "This document renders the initial prompt and function-tool descriptions that the harness "
        "would send to the model before the first assistant turn.",
        "",
        "## Task",
        "",
        f"- Task ID: `{task.get('task_id', '')}`",
        f"- Dataset: `{task.get('dataset', '')}`",
        f"- Taskset: `{taskset}`",
        f"- Task index: `{task_index}`",
        f"- Database: `{db_path}`",
        f"- Target SMILES: `{task.get('target_smiles', '')}`",
        f"- API mode rendered: `{api_mode}`",
        f"- Reaction library: `{reaction_library or 'none'}`",
        f"- Max tool calls instruction: `{max_tool_calls}`",
        f"- Max assistant turns instruction: `{max_turns}`",
        f"- Excluded tools: `{', '.join(sorted(excluded_tools)) if excluded_tools else 'none'}`",
        "",
        "## Messages",
        "",
    ]
    for message in messages:
        lines.append(f"### {message['role'].title()} Message")
        lines.append("")
        lines.append("```text")
        lines.append(message["content"])
        lines.append("```")
        lines.append("")
    lines.extend(render_tool_summary(tool_schemas))
    lines.append("")
    lines.append("## Raw Tool Schemas")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(tool_schemas, indent=2, sort_keys=True))
    lines.append("```")
    return "\n".join(lines).rstrip() + "\n"


def render_tool_summary(tool_schemas: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = [
        "## Tool Descriptions",
        "",
        "| Tool | Required arguments | Description |",
        "| --- | --- | --- |",
    ]
    for tool in tool_schemas:
        spec = normalize_tool_schema(tool)
        required = spec["parameters"].get("required") or []
        required_text = ", ".join(f"`{item}`" for item in required) if required else "none"
        lines.append(
            f"| `{spec['name']}` | {required_text} | {markdown_cell(spec.get('description', ''))} |"
        )
    lines.append("")
    for tool in tool_schemas:
        spec = normalize_tool_schema(tool)
        lines.extend(render_tool_detail(spec))
    return lines


def render_tool_detail(tool: dict[str, Any]) -> list[str]:
    parameters = tool.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    lines = [
        f"### `{tool['name']}`",
        "",
        str(tool.get("description") or "").strip(),
        "",
    ]
    if not properties:
        lines.append("No parameters.")
        lines.append("")
        return lines
    lines.extend(
        [
            "| Argument | Required | Type | Description / constraints |",
            "| --- | --- | --- | --- |",
        ]
    )
    nested_sections: list[str] = []
    for name, schema in properties.items():
        details = schema_details(name, schema)
        description = str(schema.get("description") or "")
        constraints = constraint_text(schema)
        detail_text = argument_detail_text(description, constraints)
        lines.append(
            "| "
            f"`{name}` | {'yes' if name in required else 'no'} | "
            f"`{schema_type(schema)}` | {markdown_cell(detail_text or 'none')} |"
        )
        if details:
            nested_sections.extend(details)
    lines.extend(nested_sections)
    lines.append("")
    return lines


def normalize_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if isinstance(function, dict):
        return function
    return tool


def schema_type(schema: dict[str, Any]) -> str:
    if "anyOf" in schema:
        return " | ".join(schema_type(item) for item in schema["anyOf"])
    schema_kind = schema.get("type", "object")
    if schema_kind == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            return f"array[{schema_type(items)}]"
        return "array"
    return str(schema_kind)


def schema_details(argument_name: str, schema: dict[str, Any]) -> list[str]:
    if schema.get("type") != "array":
        return []
    items = schema.get("items")
    if not isinstance(items, dict) or items.get("type") != "object":
        return []
    properties = items.get("properties") or {}
    required = set(items.get("required") or [])
    if not properties:
        return []
    lines = [
        "",
        f"Nested item fields for `{argument_name}`:",
        "",
        "| Field | Required | Type | Description |",
        "| --- | --- | --- | --- |",
    ]
    for name, item_schema in properties.items():
        description = str(item_schema.get("description") or "")
        lines.append(
            f"| `{name}` | {'yes' if name in required else 'no'} | "
            f"`{schema_type(item_schema)}` | {markdown_cell(description or 'none')} |"
        )
    lines.append("")
    return lines


def constraint_text(schema: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("default", "minimum", "maximum", "minItems", "maxItems"):
        if key in schema:
            parts.append(f"{key}: {json.dumps(schema[key])}")
    if schema.get("additionalProperties") is False:
        parts.append("additionalProperties: false")
    return "; ".join(parts)


def argument_detail_text(description: str, constraints: str) -> str:
    description = description.strip()
    constraints = constraints.strip()
    if description and constraints:
        return f"{description.rstrip('.')}. {constraints}"
    return description or constraints


def markdown_cell(text: str) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")


def slugify(value: str) -> str:
    return (
        "".join(character if character.isalnum() else "_" for character in value).strip("_").lower()
    )


if __name__ == "__main__":
    raise SystemExit(main())
