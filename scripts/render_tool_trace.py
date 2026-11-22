#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render an LLM tool-use JSONL trace as human-readable Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from synclaw.core import chem
from synclaw.token_usage import int_or_none, trace_token_usage, usage_from_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="Trace JSONL file to render")
    parser.add_argument("--output", "-o", help="Markdown output path; defaults to stdout")
    parser.add_argument("--title", help="Optional report title")
    parser.add_argument("--model", help="Optional model name to show in the summary")
    parser.add_argument(
        "--max-results", type=int, default=5, help="Maximum search hits to list per tool result"
    )
    args = parser.parse_args()

    events = load_jsonl(args.trace)
    markdown = render_trace(
        events,
        trace_path=Path(args.trace),
        title=args.title,
        model=args.model,
        max_results=max(1, args.max_results),
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            events.append(event)
    return events


def render_trace(
    events: list[dict[str, Any]],
    trace_path: Path,
    title: str | None = None,
    model: str | None = None,
    max_results: int = 5,
) -> str:
    target = infer_target(events)
    submission = final_submission_result(events)
    run_summary = final_run_summary(events)
    submitted_args = final_submission_arguments(events)
    calls = collect_tool_exchanges(events)
    turn_summary = summarize_turns(events)

    final_smiles = actual_final_from_result(submission)
    claimed_final_smiles = first_present(submission, "submitted_final_smiles")
    accepted = first_present(submission, "accepted")
    similarity = final_molecule_similarity(
        final_smiles=final_smiles,
        target_smiles=target,
        submission=submission,
        run_summary=run_summary,
    )
    reaction_validation = first_present(submission, "reaction_validation")
    building_block_grounded = first_present(submission, "building_block_grounded")

    lines: list[str] = []
    lines.append(f"# {title or 'Tool-Use Trace Report'}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Trace: `{trace_path}`")
    if model:
        lines.append(f"- Model: `{model}`")
    if target:
        lines.append(f"- Target: `{target}`")
    if final_smiles:
        lines.append(f"- RDKit final product: `{final_smiles}`")
    if claimed_final_smiles and claimed_final_smiles != final_smiles:
        lines.append(f"- Claimed final-step product: `{claimed_final_smiles}`")
    if accepted is not None:
        lines.append(f"- Submission accepted: `{str(accepted).lower()}`")
    if reaction_validation is not None:
        lines.append(f"- Reaction validation: `{reaction_validation}`")
    if building_block_grounded is not None:
        lines.append(f"- Building-block grounded: `{str(bool(building_block_grounded)).lower()}`")
    if similarity is not None:
        lines.append(f"- Active final-molecule similarity to target: `{similarity}`")
    lines.extend(render_similarity_fields(submission, run_summary=run_summary))
    runtime_seconds = first_present(run_summary, "runtime_seconds")
    if runtime_seconds is not None:
        lines.append(f"- Runtime: `{format_runtime(runtime_seconds)}`")
    token_usage = trace_token_usage(events, run_summary=run_summary)
    if token_usage:
        lines.append(f"- Token usage: {format_token_usage(token_usage)}")
    lines.append(f"- Assistant turns: `{turn_summary['assistant_turns']}`")
    lines.append(f"- Tool-call turns: `{turn_summary['tool_call_turns']}`")
    lines.append(f"- Non-tool turns: `{turn_summary['non_tool_turns']}`")
    lines.append(f"- Tool calls: `{len(calls)}`")

    lines.append("")
    lines.append("## Model Decision Trace")
    lines.append("")
    lines.extend(render_model_decision_trace(events))

    interventions = render_harness_interventions(events)
    if interventions:
        lines.append("")
        lines.append("## Harness Interventions")
        lines.append("")
        lines.extend(interventions)

    if submitted_args:
        lines.append("")
        lines.append("## Submitted Pathway")
        lines.append("")
        pathway_text = submitted_args.get("pathway_text")
        if pathway_text:
            lines.append(str(pathway_text).strip())
            lines.append("")
        if submitted_args.get("building_block"):
            lines.append(
                f"- Direct-purchase building block: `{submitted_args.get('building_block')}`"
            )
            lines.append("")
        for index, step in enumerate(submitted_args.get("steps") or [], start=1):
            if isinstance(step, dict):
                lines.extend(render_submitted_step(index, step))

    if submission:
        lines.append("")
        lines.append("## Validator Outcome")
        lines.append("")
        issues = submission.get("issues") or []
        actual_final = actual_final_from_result(submission)
        lines.append(f"- Accepted: `{str(bool(submission.get('accepted'))).lower()}`")
        lines.append(f"- Reaction validation: `{submission.get('reaction_validation', 'unknown')}`")
        if submission.get("pathway_type"):
            lines.append(f"- Pathway type: `{submission.get('pathway_type')}`")
        direct_block = submission.get("direct_building_block")
        if isinstance(direct_block, dict):
            direct_ref = direct_block.get("submit_reference") or submission.get(
                "direct_building_block_ref"
            )
            direct_smiles = direct_block.get("canonical_smiles") or direct_block.get("smiles")
            lines.append(f"- Direct-purchase building block: `{direct_ref}` -> `{direct_smiles}`")
        if submission.get("building_block_grounded") is not None:
            lines.append(
                f"- Building-block grounded: `{str(bool(submission.get('building_block_grounded'))).lower()}`"
            )
        if submission.get("referenced_building_block_ids"):
            lines.append(
                "- Referenced building blocks: "
                + ", ".join(
                    f"`{item}`" for item in submission.get("referenced_building_block_ids") or []
                )
            )
        if actual_final and submission.get("similarity_to_target") is not None:
            lines.append(f"- Active similarity to target: `{submission['similarity_to_target']}`")
        lines.extend(render_similarity_fields(submission))
        if actual_final:
            label = (
                "Final product"
                if submission.get("pathway_type") == "direct_purchase"
                else "RDKit final product"
            )
            lines.append(f"- {label}: `{actual_final}`")
        if (
            submission.get("submitted_final_smiles")
            and submission.get("submitted_final_smiles") != actual_final
        ):
            lines.append(f"- Claimed final-step product: `{submission['submitted_final_smiles']}`")
        if issues:
            lines.append(f"- Issues: `{json.dumps(issues, sort_keys=True)}`")
        else:
            lines.append("- Issues: none")
        if submission.get("steps"):
            lines.append("")
        for index, step in enumerate(submission.get("steps") or [], start=1):
            if isinstance(step, dict):
                lines.extend(render_validated_step(index, step))

    budget_lines = render_token_budgets(events)
    if budget_lines:
        append_blank_line(lines)
        lines.extend(["## Token Budget", "", *budget_lines])

    append_blank_line(lines)
    lines.append("## Tool Interaction")
    lines.append("")
    for index, exchange in enumerate(calls, start=1):
        lines.extend(render_exchange(index, exchange, max_results=max_results))

    return "\n".join(lines).rstrip() + "\n"


def append_blank_line(lines: list[str]) -> None:
    if lines and lines[-1] == "":
        return
    lines.append("")


def render_model_decision_trace(events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    assistant_messages = [
        event
        for event in events
        if event.get("event_type") == "message"
        and event.get("role") == "assistant"
        and isinstance(event.get("payload"), dict)
    ]
    if not assistant_messages:
        return ["No assistant message events were captured.", ""]

    found_reasoning_text = False
    for index, event in enumerate(assistant_messages, start=1):
        payload = event["payload"]
        lines.append(f"### Assistant Turn {index}")
        lines.append("")
        content = str(payload.get("content") or "").strip()
        tool_calls = payload.get("tool_calls") or []
        if content:
            lines.append("**Assistant-visible text:**")
            lines.append("")
            lines.append(blockquote(content))
            lines.append("")
        else:
            lines.append("**Assistant-visible text:** none captured.")
            lines.append("")

        reasoning = extract_reasoning_text(payload)
        if reasoning:
            found_reasoning_text = True
            lines.append("**Captured reasoning text:**")
            lines.append("")
            lines.append(blockquote(reasoning))
            lines.append("")
        else:
            lines.append("**Captured reasoning text:** none captured in this trace event.")
            lines.append("")

        output_types = payload.get("responses_output_types") or []
        if output_types:
            lines.append(
                f"**Responses output item types:** `{', '.join(str(item) for item in output_types)}`"
            )
            lines.append("")
        anthropic_content_types = payload.get("anthropic_content_types") or []
        if anthropic_content_types:
            lines.append(
                "**Anthropic content block types:** "
                f"`{', '.join(str(item) for item in anthropic_content_types)}`"
            )
            lines.append("")
        usage = usage_from_payload(payload)
        if usage:
            lines.append(f"**Token usage:** {format_token_usage(usage)}")
            lines.append("")
        if tool_calls:
            lines.append("**Tool calls chosen by the model:**")
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = function.get("name")
                arguments = parse_json_object(function.get("arguments"))
                lines.append(f"- `{name}`")
                for detail in summarize_tool_call_arguments(str(name), arguments):
                    lines.append(f"  - {detail}")
            lines.append("")
        else:
            lines.append("**Tool calls chosen by the model:** none.")
            lines.append("")

    if not found_reasoning_text:
        has_submission = final_submission_result(events) is not None
        submission_text = (
            "and final submitted pathway text"
            if has_submission
            else "but no final submitted pathway text"
        )
        lines.append(
            "Note: this report can only render reasoning text present in the JSONL trace. "
            f"For this run, the trace contains tool choices {submission_text}, "
            "and no separate model reasoning transcript."
        )
        lines.append("")
    return lines


def render_token_budgets(events: list[dict[str, Any]]) -> list[str]:
    snapshots = [
        event["payload"]
        for event in events
        if event.get("event_type") == "token_budget" and isinstance(event.get("payload"), dict)
    ]
    if not snapshots:
        return []
    lines = [
        "Context snapshots describe the request before each assistant turn; they are separate "
        "from cumulative billed token usage. The output cap resets for every response and "
        "includes reasoning where the provider counts it as output. Unknown values are shown as —.",
        "",
        "| Turn | Context limit | Input tokens | Remaining tokens | Output cap | Measurement | Feedback status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for snapshot in snapshots:
        turn = snapshot.get("turn_index")
        values = [str(turn + 1) if isinstance(turn, int) else "—"]
        for field in (
            "context_window_tokens",
            "input_tokens",
            "remaining_tokens",
            "max_output_tokens",
        ):
            value = snapshot.get(field)
            if field == "max_output_tokens":
                value = snapshot.get("request_output_tokens", value)
            values.append(f"{value:,}" if isinstance(value, int) else "—")
        estimated = snapshot.get("estimated")
        kind = "estimated" if estimated is True else "counted" if estimated is False else "unknown"
        source = str(snapshot.get("source") or "unavailable").replace("|", "\\|")
        status = str(snapshot.get("status") or "unknown").replace("|", "\\|")
        values.extend([f"{kind}: `{source}`", f"`{status}`"])
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_harness_interventions(events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    prefixes = (
        "Stop searching",
        "Call submit_pathway",
        "Your previous response",
        "Continue.",
        "The validator",
        "The search_",
        "The simulate_",
        "The submit_pathway",
        "You have ",
        "This is the final",
        "The tool-call budget",
        "Your context-window budget",
        "Each response is capped",
        "Remaining:",
        "Your previous response hit its output-token limit",
    )
    for event in events:
        if event.get("event_type") in {"token_budget", "harness_feedback"}:
            content = event.get("payload", {}).get("content")
            if content:
                lines.append(f"- {content}")
            continue
        if event.get("event_type") != "message" or event.get("role") != "user":
            continue
        payload = event.get("payload")
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, str) or not content.startswith(prefixes):
            continue
        lines.append(f"- {content}")
    return lines


def summarize_tool_call_arguments(name: str, arguments: dict[str, Any]) -> list[str]:
    if name in {"similarity", "mces"}:
        return [f"arguments: `{format_similarity_arguments(arguments)}`"]
    if name == "search_reaction_templates":
        details = [f"query: `{arguments.get('query')}`"]
        if arguments.get("reactant_smiles"):
            details.append(
                f"reactants: {format_smiles_list(arguments.get('reactant_smiles') or [])}"
            )
        if arguments.get("product_smiles"):
            details.append(f"products: {format_smiles_list(arguments.get('product_smiles') or [])}")
        if arguments.get("run_forward"):
            details.append("run_forward: `true`")
        return details
    if name == "search_building_blocks":
        details = [f"query: `{arguments.get('query')}`"]
        return details
    if name == "search_reaction_templates_by_molecule":
        details = []
        if arguments.get("reactant_smiles"):
            details.append(
                f"reactants: {format_smiles_list(arguments.get('reactant_smiles') or [])}"
            )
        if arguments.get("product_smiles"):
            details.append(f"products: {format_smiles_list(arguments.get('product_smiles') or [])}")
        if arguments.get("library"):
            details.append(f"library: `{arguments.get('library')}`")
        if arguments.get("run_forward"):
            details.append("run_forward: `true`")
        return details or [f"arguments: `{compact_json(arguments)}`"]
    if name == "simulate_reaction":
        reactants = arguments.get("reactants")
        details = [
            f"template: `{arguments.get('reaction_id')}`",
            f"reactants: {format_smiles_list(reactants or [])}",
        ]
        if reactants is not None and not isinstance(reactants, list):
            details.append(f"reactants argument type: `{type(reactants).__name__}`")
        if arguments.get("product_smiles"):
            details.append(f"claimed product: `{arguments.get('product_smiles')}`")
        return details
    if name == "simulate_reverse_reaction":
        details = [
            f"template: `{arguments.get('reaction_id')}`",
            f"products: {format_smiles_list(normalize_smiles_items(arguments.get('product_smiles')))}",
        ]
        if arguments.get("max_outcomes") is not None:
            details.append(f"max outcomes: `{arguments.get('max_outcomes')}`")
        if "include_near" in arguments:
            details.append(
                f"include near matches: `{str(bool(arguments.get('include_near'))).lower()}`"
            )
        return details
    if name == "submit_pathway":
        details = []
        steps = arguments.get("steps") if isinstance(arguments.get("steps"), list) else []
        details.append(f"submitted steps: `{len(steps)}`")
        if arguments.get("building_block"):
            details.append(f"direct-purchase building block: `{arguments.get('building_block')}`")
        for index, step in enumerate(steps, start=1):
            if isinstance(step, dict):
                details.append(
                    f"step {index}: `{step.get('reaction_id')}` with "
                    f"{format_smiles_list(step.get('reactants') or [])} -> `{step.get('product_smiles')}`"
                )
        pathway_text = arguments.get("pathway_text")
        if pathway_text:
            details.append(f"pathway text: {inline_text(pathway_text)}")
        return details
    return [f"arguments: `{compact_json(arguments)}`"]


def render_submitted_step(index: int, step: dict[str, Any]) -> list[str]:
    reactants = step.get("reactants") or []
    return [
        f"### Step {index}",
        "",
        f"- Template: `{step.get('reaction_id')}`",
        f"- Reactants: {format_smiles_list(reactants)}",
        f"- Claimed product: `{step.get('product_smiles')}`",
        "",
    ]


def render_validated_step(index: int, step: dict[str, Any]) -> list[str]:
    lines = [
        f"### Validated Step {index}",
        "",
        f"- Valid: `{str(bool(step.get('valid'))).lower()}`",
        f"- Template: `{step.get('reaction_id')}`",
    ]
    generated = step.get("generated_products") or []
    if generated:
        lines.append(f"- Generated products: {format_smiles_list(generated)}")
    resolved = step.get("resolved_reactants") or []
    if resolved:
        lines.append("- Resolved reactants:")
        for item in resolved:
            if not isinstance(item, dict):
                continue
            source = item.get("source", "unknown")
            ref = item.get("ref", "")
            smiles = item.get("smiles", "")
            bb = item.get("building_block") or {}
            bb_suffix = (
                f", BB id `{bb.get('id')}`"
                if isinstance(bb, dict) and bb.get("id") is not None
                else ""
            )
            lines.append(f"  - `{ref}` -> `{smiles}` ({source}{bb_suffix})")
    issues = step.get("issues") or []
    if issues:
        lines.append(f"- Issues: `{json.dumps(issues, sort_keys=True)}`")
    lines.append("")
    return lines


def render_exchange(index: int, exchange: dict[str, Any], max_results: int) -> list[str]:
    name = exchange["name"]
    arguments = exchange.get("arguments") or {}
    result = exchange.get("result")
    lines = [f"### {index}. `{name}`", ""]
    lines.extend(render_call(name, arguments))
    coercions = exchange.get("argument_coercions") or []
    if coercions:
        lines.append("")
        lines.append("**Argument coercions:**")
        for item in coercions:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- "
                + f"`{item.get('field')}`: `{item.get('coercion')}` "
                + f"from `{item.get('raw_value')}` to {format_smiles_list(item.get('coerced_value'))}"
            )
    if result is None:
        lines.append("")
        lines.append("**Result:** no matching tool result found")
        lines.append("")
        return lines
    lines.append("")
    compaction = exchange.get("model_result_compaction")
    if isinstance(compaction, dict):
        full_size = compaction.get("full_json_bytes")
        model_size = compaction.get("model_json_bytes")
        if full_size is not None and model_size is not None:
            lines.append(
                "**Model-visible result:** compacted "
                + f"from `{full_size}` to `{model_size}` JSON byte(s)"
            )
            lines.append("")
    lines.extend(render_result(name, result, max_results=max_results))
    lines.append("")
    return lines


def render_call(name: str, arguments: dict[str, Any]) -> list[str]:
    if name in {"similarity", "mces"}:
        return ["**Call:**", f"- Arguments: `{format_similarity_arguments(arguments)}`"]

    if name == "search_reaction_templates":
        lines = ["**Call:**"]
        lines.append(f"- Query: `{arguments.get('query')}`")
        if arguments.get("reactant_smiles"):
            lines.append(
                f"- Reactants: {format_smiles_list(arguments.get('reactant_smiles') or [])}"
            )
        if arguments.get("product_smiles"):
            lines.append(f"- Products: {format_smiles_list(arguments.get('product_smiles') or [])}")
        if arguments.get("limit") is not None:
            lines.append(f"- Limit: `{arguments.get('limit')}`")
        if "run_forward" in arguments:
            lines.append(f"- Run forward: `{str(bool(arguments.get('run_forward'))).lower()}`")
        return lines

    if name == "search_building_blocks":
        lines = ["**Call:**"]
        lines.append(f"- Query: `{arguments.get('query')}`")
        if arguments.get("limit") is not None:
            lines.append(f"- Limit: `{arguments.get('limit')}`")
        if "include_near" in arguments:
            lines.append(
                f"- Include near matches: `{str(bool(arguments.get('include_near'))).lower()}`"
            )
        return lines

    if name == "search_reaction_templates_by_molecule":
        lines = ["**Call:**"]
        if arguments.get("reactant_smiles"):
            lines.append(
                f"- Reactants: {format_smiles_list(arguments.get('reactant_smiles') or [])}"
            )
        if arguments.get("product_smiles"):
            lines.append(f"- Products: {format_smiles_list(arguments.get('product_smiles') or [])}")
        if arguments.get("library"):
            lines.append(f"- Library: `{arguments.get('library')}`")
        if "run_forward" in arguments:
            lines.append(f"- Run forward: `{str(bool(arguments.get('run_forward'))).lower()}`")
        return lines

    if name == "simulate_reaction":
        lines = ["**Call:**"]
        lines.append(f"- Template: `{arguments.get('reaction_id')}`")
        lines.append(f"- Reactants: {format_smiles_list(arguments.get('reactants') or [])}")
        if arguments.get("product_smiles"):
            lines.append(f"- Claimed product: `{arguments.get('product_smiles')}`")
        return lines

    if name == "simulate_reverse_reaction":
        lines = ["**Call:**"]
        lines.append(f"- Template: `{arguments.get('reaction_id')}`")
        lines.append(
            f"- Products: {format_smiles_list(normalize_smiles_items(arguments.get('product_smiles')))}"
        )
        if arguments.get("max_outcomes") is not None:
            lines.append(f"- Max outcomes: `{arguments.get('max_outcomes')}`")
        if "include_near" in arguments:
            lines.append(
                f"- Include near matches: `{str(bool(arguments.get('include_near'))).lower()}`"
            )
        return lines

    if name == "submit_pathway":
        steps = arguments.get("steps") if isinstance(arguments.get("steps"), list) else []
        lines = ["**Call:**"]
        lines.append(f"- Submitted steps: `{len(steps)}`")
        if arguments.get("building_block"):
            lines.append(f"- Direct-purchase building block: `{arguments.get('building_block')}`")
        for index, step in enumerate(steps, start=1):
            if isinstance(step, dict):
                lines.append(
                    f"- Step {index}: `{step.get('reaction_id')}` with "
                    f"{format_smiles_list(step.get('reactants') or [])} -> `{step.get('product_smiles')}`"
                )
        return lines

    return [f"**Call:** `{compact_json(arguments)}`"]


def render_result(name: str, result: dict[str, Any], max_results: int) -> list[str]:
    if name == "similarity":
        lines = [
            f"**Result:** valid `{str(bool(result.get('valid'))).lower()}`, "
            f"similarity `{result.get('similarity')}`"
        ]
        left = result.get("left") if isinstance(result.get("left"), dict) else {}
        right = result.get("right") if isinstance(result.get("right"), dict) else {}
        if left.get("canonical_smiles"):
            lines.append(f"- Canonical smiles_a: `{left.get('canonical_smiles')}`")
        if right.get("canonical_smiles"):
            lines.append(f"- Canonical smiles_b: `{right.get('canonical_smiles')}`")
        if result.get("error"):
            lines.append(f"- Error: `{result.get('error')}`")
        return lines

    if name == "mces":
        lines = [
            f"**Result:** valid `{str(bool(result.get('valid'))).lower()}`, "
            f"normalized overlap `{result.get('similarity')}`"
        ]
        if result.get("mces_smiles"):
            lines.append(f"- MCES SMILES: `{result.get('mces_smiles')}`")
        lines.append(
            "- Bonds: "
            f"matched `{result.get('matched_bonds')}`, union `{result.get('union_bonds')}`, "
            f"left `{result.get('left_bonds')}`, right `{result.get('right_bonds')}`"
        )
        lines.append(f"- Timed out: `{str(bool(result.get('timed_out'))).lower()}`")
        if result.get("error"):
            lines.append(f"- Error: `{result.get('error')}`")
        return lines

    if name == "search_reaction_templates":
        # Combined text-and-molecule searches name the structural list separately.
        structural = result.get("structural_results", result.get("results")) or []
        lines = [
            f"**Result:** {result.get('structural_count', result.get('count', 0))} template hits"
        ]
        if result.get("search_mode"):
            lines.append(f"- Search mode: `{result.get('search_mode')}`")
        reverse_ranking = result.get("reverse_ranking") or {}
        if reverse_ranking.get("applied"):
            lines.append(
                "- Reverse ranking: "
                + f"`{reverse_ranking.get('enriched_result_count')}` template(s) enriched"
            )
        for item in structural[:max_results]:
            structure = item.get("structure_match") or {}
            usefulness = item.get("reverse_usefulness") or {}
            score_bits = []
            if structure.get("score") is not None:
                score_bits.append(f"structure `{structure.get('score')}`")
            if usefulness.get("score") is not None:
                score_bits.append(f"reverse `{usefulness.get('score')}`")
            score_text = f" [{', '.join(score_bits)}]" if score_bits else ""
            lines.append(
                "- "
                + f"`{item.get('reaction_id')}`"
                + f" ({', '.join(item.get('tags') or [])})"
                + score_text
                + f": {item.get('description', '')}"
            )
        if "text_results" in result:
            lines.append(
                f"- Text results: `{len(result['text_results'])}`: "
                + ", ".join(
                    f"`{item.get('reaction_id')}`" for item in result["text_results"][:max_results]
                )
            )
        return lines

    if name == "search_building_blocks":
        lines = [f"**Result:** {result.get('count', 0)} building-block hits"]
        for item in (result.get("results") or [])[:max_results]:
            similarity = item.get("similarity")
            sim_text = f", similarity `{similarity}`" if similarity is not None else ""
            lines.append(
                "- "
                + f"`{item.get('smiles')}`"
                + f" (id `{item.get('id')}`, {item.get('match_type', 'match')}{sim_text})"
            )
        return lines

    if name == "simulate_reaction":
        lines = [
            f"**Result:** valid `{str(bool(result.get('valid'))).lower()}`, "
            f"{result.get('generated_product_count', 0)} product(s)"
        ]
        if result.get("product_smiles"):
            lines.append(
                f"- Claimed product matched: `{str(bool(result.get('product_matched'))).lower()}`"
            )
        products = result.get("generated_products") or []
        if products:
            lines.append(f"- Products: {format_smiles_list(products)}")
        product_sets = result.get("generated_product_sets") or []
        if product_sets and any(
            len(product_set) > 1 for product_set in product_sets if isinstance(product_set, list)
        ):
            shown_sets = [
                format_smiles_list(product_set)
                for product_set in product_sets[:max_results]
                if isinstance(product_set, list)
            ]
            lines.append(f"- Product sets: {'; '.join(shown_sets)}")
        issues = result.get("issues") or []
        if issues:
            lines.append(f"- Issues: `{json.dumps(issues, sort_keys=True)}`")
        if result.get("error_message"):
            lines.append(f"- Validator message: {result['error_message']}")
        return lines

    if name == "simulate_reverse_reaction":
        return render_reverse_reaction_result(result, max_results=max_results)

    if name == "search_reaction_templates_by_molecule":
        lines = [f"**Result:** {result.get('count', 0)} structural template hit(s)"]
        for item in (result.get("results") or [])[:max_results]:
            structure = item.get("structure_match") or {}
            lines.append(
                "- "
                + f"`{item.get('reaction_id')}`"
                + f" score `{structure.get('score')}`"
                + f": {item.get('description', '')}"
            )
        issues = result.get("issues") or []
        if issues:
            lines.append(f"- Issues: `{json.dumps(issues, sort_keys=True)}`")
        return lines

    if name == "submit_pathway":
        lines = [
            f"**Result:** accepted `{str(bool(result.get('accepted'))).lower()}`, "
            f"reaction validation `{result.get('reaction_validation', 'unknown')}`"
        ]
        if result.get("pathway_type"):
            lines.append(f"- Pathway type: `{result.get('pathway_type')}`")
        direct_block = result.get("direct_building_block")
        if isinstance(direct_block, dict):
            direct_ref = direct_block.get("submit_reference") or result.get(
                "direct_building_block_ref"
            )
            direct_smiles = direct_block.get("canonical_smiles") or direct_block.get("smiles")
            lines.append(f"- Direct-purchase building block: `{direct_ref}` -> `{direct_smiles}`")
        actual_final = actual_final_from_result(result)
        if actual_final and result.get("similarity_to_target") is not None:
            lines.append(f"- Active similarity to target: `{result['similarity_to_target']}`")
        lines.extend(render_similarity_fields(result))
        if actual_final:
            label = (
                "Final product"
                if result.get("pathway_type") == "direct_purchase"
                else "RDKit final product"
            )
            lines.append(f"- {label}: `{actual_final}`")
        if (
            result.get("submitted_final_smiles")
            and result.get("submitted_final_smiles") != actual_final
        ):
            lines.append(f"- Claimed final-step product: `{result['submitted_final_smiles']}`")
        issues = result.get("issues") or []
        if issues:
            lines.append(f"- Issues: `{json.dumps(issues, sort_keys=True)}`")
        return lines

    return [f"**Result:** `{compact_json(result)}`"]


def render_reverse_reaction_result(result: dict[str, Any], max_results: int) -> list[str]:
    candidate_sets = result.get("candidate_reactant_sets") or []
    lines = [
        f"**Result:** valid `{str(bool(result.get('valid'))).lower()}`, "
        f"{result.get('returned_candidate_count', len(candidate_sets))} candidate precursor set(s)"
    ]
    template = result.get("template_summary")
    if isinstance(template, dict):
        family = template.get("reaction_family")
        description = template.get("description")
        if family or description:
            lines.append(
                "- Template summary: "
                + " / ".join(str(item) for item in (family, description) if item)
            )
    canonical_products = result.get("product_canonical_smiles")
    if canonical_products:
        lines.append(
            f"- Canonical product input: {format_smiles_list(normalize_smiles_items(canonical_products))}"
        )
    if result.get("product_order_permuted"):
        lines.append(f"- Product order permuted: `{result.get('used_product_order')}`")
    planning = result.get("reverse_planning_summary")
    if isinstance(planning, dict):
        lines.append(
            "- Grounding summary: "
            + f"`{planning.get('fully_grounded_candidate_sets', 0)}` fully grounded candidate set(s); "
            + str(planning.get("recommended_next_step", ""))
        )
    match_summary = result.get("building_block_match_summary")
    if isinstance(match_summary, dict):
        lines.append(
            "- Building-block matches: "
            + f"`{match_summary.get('exact_match_count', 0)}` exact, "
            + f"`{match_summary.get('returned_near_match_count', 0)}` near"
        )

    warnings = result.get("warnings") or []
    if warnings:
        lines.append(f"- Warnings: `{json.dumps(warnings, sort_keys=True)}`")
    issues = result.get("issues") or []
    if issues:
        lines.append(f"- Issues: `{json.dumps(issues, sort_keys=True)}`")

    if candidate_sets:
        lines.append("- Candidate precursor sets:")
        for index, candidate in enumerate(candidate_sets[:max_results], start=1):
            if not isinstance(candidate, dict):
                continue
            reactants = candidate.get("reactants") or []
            grounded = candidate.get("all_reactants_have_exact_building_block_match")
            lines.append(
                f"  - Set {index}: {format_smiles_list(reactants)} "
                + f"(fully grounded `{str(bool(grounded)).lower()}`)"
            )
            details = candidate.get("reactant_details") or []
            for detail in details:
                if isinstance(detail, dict):
                    lines.append(f"    - {format_reverse_reactant_detail(detail)}")
        if len(candidate_sets) > max_results:
            lines.append(
                f"  - ... {len(candidate_sets) - max_results} more candidate set(s) omitted"
            )
    return lines


def format_reverse_reactant_detail(detail: dict[str, Any]) -> str:
    smiles = detail.get("smiles")
    exact = detail.get("exact_building_block_matches") or []
    near = detail.get("top_near_building_block_matches") or []
    parts = [f"`{smiles}`"]
    if exact:
        parts.append("exact " + format_building_block_refs(exact))
    else:
        parts.append("no exact BB match")
    if near:
        parts.append("near " + format_building_block_refs(near))
    return "; ".join(parts)


def format_building_block_refs(matches: list[dict[str, Any]]) -> str:
    formatted = []
    for match in matches[:3]:
        if not isinstance(match, dict):
            continue
        reference = match.get("submit_reference") or f"BB:{match.get('id')}"
        similarity = match.get("similarity")
        suffix = f" sim `{similarity}`" if similarity is not None else ""
        formatted.append(f"`{reference}`{suffix}")
    return ", ".join(formatted) if formatted else "none"


def collect_tool_exchanges(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exchanges: list[dict[str, Any]] = []
    open_indexes: dict[str, list[int]] = {}
    for event in events:
        event_type = event.get("event_type")
        name = event.get("tool_name")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event_type == "tool_call":
            call = payload.get("raw_tool_call") if isinstance(payload, dict) else None
            call_id = call.get("id") if isinstance(call, dict) else None
            exchange = {
                "name": name,
                "arguments": payload.get("dispatch_arguments") or payload.get("arguments", {}),
                "raw_arguments": payload.get("arguments", {}),
                "argument_coercions": payload.get("argument_coercions") or [],
                "call_id": call_id,
                "result": None,
            }
            open_indexes.setdefault(str(name), []).append(len(exchanges))
            exchanges.append(exchange)
        elif event_type == "tool_result" and name:
            indexes = open_indexes.get(str(name)) or []
            index = indexes.pop(0) if indexes else None
            if index is not None:
                exchanges[index]["result"] = payload.get("result")
                if isinstance(payload.get("model_result"), dict):
                    exchanges[index]["model_result"] = payload.get("model_result")
                if isinstance(payload.get("model_result_compaction"), dict):
                    exchanges[index]["model_result_compaction"] = payload.get(
                        "model_result_compaction"
                    )
    return exchanges


def summarize_turns(events: list[dict[str, Any]]) -> dict[str, int]:
    assistant_turns = 0
    tool_call_turns = 0
    for event in events:
        if event.get("event_type") != "message" or event.get("role") != "assistant":
            continue
        assistant_turns += 1
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if payload.get("tool_calls"):
            tool_call_turns += 1
    return {
        "assistant_turns": assistant_turns,
        "tool_call_turns": tool_call_turns,
        "non_tool_turns": assistant_turns - tool_call_turns,
    }


def final_submission_arguments(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event_type") != "selected_submission":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("submission_arguments"), dict):
            return payload["submission_arguments"]
        if isinstance(payload.get("arguments"), dict):
            return payload["arguments"]
        if isinstance(payload.get("result"), dict):
            return _submission_arguments_from_result(payload["result"])
    for event in reversed(events):
        if event.get("event_type") == "tool_call" and event.get("tool_name") == "submit_pathway":
            payload = event.get("payload")
            if isinstance(payload, dict):
                if isinstance(payload.get("dispatch_arguments"), dict):
                    return payload["dispatch_arguments"]
                if isinstance(payload.get("arguments"), dict):
                    return payload["arguments"]
    return None


def _submission_arguments_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    steps = []
    for step_result in result.get("steps") or []:
        if not isinstance(step_result, dict):
            continue
        submitted_step = step_result.get("submitted_step")
        if isinstance(submitted_step, dict):
            steps.append(submitted_step)
    pathway_text = result.get("pathway_text")
    direct_ref = result.get("direct_building_block_ref")
    if not direct_ref:
        direct_block = result.get("direct_building_block")
        if isinstance(direct_block, dict):
            direct_ref = direct_block.get("submit_reference")
    if not steps and not pathway_text and not direct_ref:
        return None
    arguments: dict[str, Any] = {"steps": steps}
    if direct_ref:
        arguments["building_block"] = direct_ref
    if pathway_text:
        arguments["pathway_text"] = pathway_text
    return arguments


def final_submission_result(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event_type") != "selected_submission":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            return payload["result"]
    for event in reversed(events):
        if event.get("event_type") == "tool_result" and event.get("tool_name") == "submit_pathway":
            payload = event.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
                return payload["result"]
    return None


def infer_target(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        target = event.get("target_smiles")
        if target:
            return str(target)
        payload = event.get("payload")
        if isinstance(payload, dict):
            target = payload.get("target_smiles")
            if target:
                return str(target)
    return None


def final_run_summary(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event_type") not in {"run_summary", "summary"}:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            if isinstance(payload.get("evaluation_summary"), dict):
                return payload["evaluation_summary"]
            return payload
    return None


def final_molecule_similarity(
    *,
    final_smiles: str | None,
    target_smiles: str | None,
    submission: dict[str, Any] | None,
    run_summary: dict[str, Any] | None,
) -> float | None:
    for source in (run_summary, submission):
        value = first_present(source, "similarity_to_target")
        if value is not None:
            return value
    if final_smiles and target_smiles:
        return chem.similarity(final_smiles, target_smiles).get("similarity")
    return None


def render_similarity_fields(
    submission: dict[str, Any] | None,
    *,
    run_summary: dict[str, Any] | None = None,
) -> list[str]:
    constitutional = first_present(submission, "constitutional_similarity_to_target")
    if constitutional is None:
        constitutional = first_present(run_summary, "constitutional_similarity_to_target")
    if constitutional is None:
        return []
    return [f"- Constitutional similarity to target: `{constitutional}`"]


def format_token_usage(usage: dict[str, Any]) -> str:
    labels = [
        ("input_tokens", "input"),
        ("output_tokens", "output"),
        ("total_tokens", "total"),
        ("cached_input_tokens", "cached input"),
        ("cache_creation_input_tokens", "cache write"),
        ("cache_read_input_tokens", "cache read"),
        ("reasoning_tokens", "reasoning"),
        ("cost", "cost"),
    ]
    parts = [
        f"{label} `{format_usage_value(field, usage[field])}`"
        for field, label in labels
        if usage.get(field) is not None
    ]
    return ", ".join(parts)


def format_usage_value(field: str, value: Any) -> str:
    if field == "cost" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.12g}"
    return str(value)


def sum_ints(*values: Any) -> int | None:
    total = 0
    saw_value = False
    for value in values:
        integer = int_or_none(value)
        if integer is None:
            continue
        total += integer
        saw_value = True
    return total if saw_value else None


def first_present(source: dict[str, Any] | None, key: str) -> Any:
    if not source:
        return None
    value = source.get(key)
    return value if value is not None else None


def actual_final_from_result(result: dict[str, Any] | None) -> str | None:
    if not result:
        return None
    if "actual_final_smiles" in result:
        return result.get("actual_final_smiles")
    if result.get("accepted") is True or result.get("reaction_validation") == "valid":
        return result.get("final_smiles")
    return None


def format_runtime(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remainder:.1f}s"


def extract_reasoning_text(payload: dict[str, Any]) -> str | None:
    candidates = [
        payload.get("reasoning_summary"),
        payload.get("reasoning_text"),
        payload.get("reasoning"),
        payload.get("reasoning_content"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, dict):
            summary = candidate.get("summary")
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
            if isinstance(summary, list) and summary:
                return "\n".join(str(item) for item in summary)
    return None


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def format_smiles_list(items: Any) -> str:
    if not isinstance(items, list):
        return f"`{items}`"
    return " + ".join(f"`{item}`" for item in items)


def normalize_smiles_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def blockquote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def inline_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return f"`{text}`"


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def format_similarity_arguments(arguments: dict[str, Any]) -> str:
    ordered = {
        "smiles_a": arguments.get("smiles_a"),
        "smiles_b": arguments.get("smiles_b"),
    }
    return json.dumps(ordered, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
