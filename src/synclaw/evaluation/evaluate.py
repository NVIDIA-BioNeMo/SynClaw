# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace ingestion and scoring for LLM synthesis pathway runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from synclaw.core import chem, db, reactions
from synclaw.core.libraries import LibrarySelection
from synclaw.core.toxicity import MODE_OFF, ToxicityConfig, toxicity_summary_fields
from synclaw.rollout.loop import call_local_tool
from synclaw.rollout.policy import (
    DEFAULT_STOP_CONDITION,
    StopCondition,
    is_better_accepted_submission,
    meets_submission_floor,
    parse_stop_condition,
    submission_identical_to_target,
    submission_similarity_to_target,
)
from synclaw.token_usage import float_or_none, trace_token_usage
from synclaw.tools.synthesis import (
    SynthesisTools,
    hidden_target_summary_fields,
    score_pathway_against_target,
)

SEARCH_TOOLS = {"search_building_blocks", "search_reaction_templates"}


def evaluate_trace_file(
    trace_path: str | Path,
    db_path: str | Path = db.DEFAULT_DB_PATH,
    model: str = "unknown",
    prompt_version: str | None = None,
    target_smiles: str | None = None,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one ``run-model`` trace file as a row for the results CSV.

    The submission is the result in the trace's ``selected_submission`` event. A trace
    without one, such as a partial run, has its ``submit_pathway`` calls replayed against
    ``db_path`` and selected as the run selects; if the run's choice cannot be replayed,
    the last recorded ``submit_pathway`` result is used. Only an accepted submission's
    ``actual_final_smiles`` is scored. ``target_smiles`` overrides the target recorded in
    the trace. ``tool_config`` supplies the run's reaction library, stop condition and
    toxicity screen.
    """

    path = Path(trace_path)
    events = list(db.iter_jsonl(path))
    if not events:
        raise ValueError(f"{path} contains no JSONL events")

    inferred_target = target_smiles or _infer_target(events)
    run_summary = extract_run_summary(events)
    hidden_target = hidden_target_summary_fields(events)
    final = extract_selected_submission(events)
    # The run's selection never saw a call it did not execute, so neither path below may.
    executed_events = _without_unexecuted_calls(events)
    if final is None:
        final = replay_selected_submission(
            executed_events,
            db_path=db_path,
            target_smiles=inferred_target,
            # Partial traces have no run_summary, so re-scoring them needs the selection passed in.
            library=(tool_config or {}).get("reaction_library")
            or (run_summary or {}).get("reaction_library"),
            hide_target=hidden_target["target_hidden_from_stock"],
        )
    if final is None:
        recorded = _recorded_submit_results(executed_events)
        final = recorded[-1] if recorded else None
    final = final or {}
    accepted = final.get("accepted")
    final_smiles = final.get("actual_final_smiles") if accepted is True else None
    reaction_validation = final.get("reaction_validation")
    reaction_valid = reaction_validation == "valid" if reaction_validation is not None else None
    building_block_grounded = final.get("building_block_grounded")
    referenced_building_block_ids = final.get("referenced_building_block_ids")

    scoring = score_pathway_against_target(
        [final_smiles] if final_smiles else [],
        target_smiles=inferred_target,
        target_validation=chem.validate_smiles(inferred_target) if inferred_target else None,
        preferred_smiles=None,
        accepted=accepted is True,
    )
    similarity_to_target = scoring["similarity_to_target"]
    identical_to_target = scoring["identical_to_target"]
    runtime_seconds = float_or_none(run_summary.get("runtime_seconds") if run_summary else None)
    token_usage = trace_token_usage(events, run_summary=run_summary)
    similarity_floor = _similarity_floor(tool_config)
    meets_similarity_floor = accepted is True and meets_submission_floor(
        similarity_to_target, similarity_floor, identical_to_target=identical_to_target,
    )
    run_success = accepted is True

    summary = {
        "trace_file": str(path),
        "model": model,
        "prompt_version": prompt_version,
        "target_smiles": inferred_target,
        "final_smiles": final_smiles,
        "submitted_final_smiles": final.get("submitted_final_smiles"),
        "accepted": accepted,
        "reaction_validation": reaction_validation,
        "reaction_valid": reaction_valid,
        "building_block_grounded": building_block_grounded,
        "referenced_building_block_ids": (
            json.dumps(referenced_building_block_ids, sort_keys=True)
            if referenced_building_block_ids is not None
            else None
        ),
        "similarity_floor": similarity_floor,
        "meets_similarity_floor": meets_similarity_floor,
        "run_success": run_success,
        "has_final_submission": bool(final),
        "has_pathway_text": bool(final.get("pathway_text")),
        "similarity_to_target": similarity_to_target,
        "identical_to_target": identical_to_target,
        "constitutional_similarity_to_target": scoring["constitutional_similarity_to_target"],
        "stereo_aware_similarity_to_target": scoring["stereo_aware_similarity_to_target"],
        "runtime_seconds": runtime_seconds,
        **token_usage,
        **summarize_events(events),
        **hidden_target,
        **toxicity_summary_fields(
            events,
            (run_summary or {}).get("toxicity_screen") or (tool_config or {}).get("toxicity_screen"),
        ),
    }
    return summary


def replay_selected_submission(
    events: list[dict[str, Any]],
    db_path: str | Path,
    target_smiles: str | None,
    *,
    library: LibrarySelection = None,
    hide_target: bool = False,
) -> dict[str, Any] | None:
    """Replay every submit_pathway call and select one as the runners do.

    Each call is dispatched through ``call_local_tool`` with its recorded
    ``dispatch_arguments``, so malformed arguments fail as they did in the run. A call
    whose arguments did not parse was never dispatched and keeps its recorded result.
    The best accepted submission is kept, so a later, worse one cannot replace it. With
    none accepted, the last submission stands. ``hide_target`` replays with the target out
    of stock, as a run that hid it validated its submissions.

    Returns the selected ``submit_pathway`` result, or None when the submission the run
    would select cannot be replayed: none was accepted and the last call cannot be. An
    earlier replayed call is not substituted, since the run did not select it; the caller
    reads the last call's recorded result instead.
    """

    submit_calls = [
        event for event in events
        if event["event_type"] == "tool_call" and event["tool_name"] == "submit_pathway"
    ]
    if not submit_calls:
        return None
    recorded_results = _recorded_submit_results(events)
    tools = SynthesisTools(
        db_path,
        reaction_libraries=library,
        # Replay re-scores a recorded trace without the screen; recorded refusals are kept below.
        toxicity=ToxicityConfig(mode=MODE_OFF),
        hidden_target_smiles=target_smiles if hide_target else None,
        hidden_building_block_ids=frozenset(
            json.loads(hidden_target_summary_fields(events)["hidden_building_block_ids"] or "[]")
        ),
    )
    best: dict[str, Any] | None = None
    best_identical: bool | None = None
    best_similarity: float | None = None
    last: dict[str, Any] | None = None
    for ordinal, event in enumerate(submit_calls):
        recorded = recorded_results[ordinal] if ordinal < len(recorded_results) else {}
        last = _replay_submission(tools, event["payload"], recorded, target_smiles)
        if last is not None and recorded.get("refused") is True:
            # Replay runs without the screen, so keep the refusal the run itself recorded.
            last = {**last, "accepted": False, "refused": True}
        if last is None or not last.get("accepted"):
            continue
        similarity = submission_similarity_to_target(last, target_smiles)
        identical = submission_identical_to_target(last, target_smiles) is True
        if best is None or is_better_accepted_submission(identical, similarity, best_identical, best_similarity):
            best, best_identical, best_similarity = last, identical, similarity
    return best or last


def _replay_submission(
    tools: SynthesisTools,
    payload: dict[str, Any],
    recorded: dict[str, Any],
    target_smiles: str | None,
) -> dict[str, Any] | None:
    if payload["argument_error"] is not None:
        # The run never dispatched this call; its result came from the argument error.
        return recorded or None
    arguments = payload["dispatch_arguments"]
    steps = arguments.get("steps")
    if isinstance(steps, list):
        resolved_steps = _resolve_reaction_ids(tools, steps, _recorded_step_smarts(recorded))
        if resolved_steps is None:
            return None
        arguments = {**arguments, "steps": resolved_steps}
    return call_local_tool(
        tools,
        "submit_pathway",
        arguments,
        reaction_library=tools.reaction_libraries,
        target_smiles=target_smiles,
    )


def _resolve_reaction_ids(
    tools: SynthesisTools,
    steps: list[Any],
    recorded_smarts: dict[int, str],
) -> list[Any] | None:
    """Resolve each step's template ID, or return None when one cannot be resolved.

    Template IDs change between catalog versions, but each step's result records the
    SMARTS it ran, so an unknown ID is resolved to the current template with that SMARTS.
    A step without a usable ``reaction_id`` is left for ``submit_pathway`` to reject.
    """

    resolved = []
    library = tools.reaction_libraries
    with db.session(tools.db_path) as conn:
        for index, step in enumerate(steps):
            reaction_id = step.get("reaction_id") if isinstance(step, dict) else None
            # submit_pathway grounds a reaction SMARTS to a library template itself.
            if not reaction_id or not isinstance(reaction_id, str) or reactions.is_reaction_smarts(reaction_id):
                resolved.append(step)
                continue
            template = db.get_reaction_template(conn, reaction_id, library=library)
            if template is None and index in recorded_smarts:
                template = db.get_reaction_template_by_smarts(conn, recorded_smarts[index], library=library)
                if template is not None:
                    step = {**step, "reaction_id": template["reaction_id"]}
            if template is None:
                return None
            resolved.append(step)
    return resolved


def extract_selected_submission(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The result of the run's ``selected_submission`` event, or None if it recorded none."""

    selected = [event for event in events if event["event_type"] == "selected_submission"]
    return selected[-1]["payload"]["result"] if selected else None


def _recorded_submit_results(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The recorded result of each submit_pathway call, in call order, as results are appended."""

    return [
        event["payload"]["result"] for event in events
        if event["event_type"] == "tool_result" and event["tool_name"] == "submit_pathway"
    ]


def _without_unexecuted_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the call and result events of tool calls the run recorded but never executed.

    Past the tool-call budget the loop records each call and a stub result without running
    it. A tool's results are recorded in call order, so the n-th result of a tool pairs with
    its n-th call.
    """

    def tool_ordinals(event_type: str) -> dict[int, tuple[str, int]]:
        """Key each event of this type by its index to its tool name and per-tool ordinal."""

        counts: dict[str, int] = {}
        keys = {}
        for index, event in enumerate(events):
            if event["event_type"] == event_type:
                name = event["tool_name"]
                counts[name] = counts.get(name, -1) + 1
                keys[index] = (name, counts[name])
        return keys

    results = tool_ordinals("tool_result")
    unexecuted = {
        key for index, key in results.items()
        if events[index]["payload"]["outcome"] == "over_budget"
    }
    if not unexecuted:
        return events
    dropped = {
        index for keys in (results, tool_ordinals("tool_call"))
        for index, key in keys.items() if key in unexecuted
    }
    return [event for index, event in enumerate(events) if index not in dropped]


def _recorded_step_smarts(result: dict[str, Any]) -> dict[int, str]:
    """Map step positions to the template SMARTS a recorded submit_pathway result ran."""

    steps = result.get("steps")
    recorded: dict[int, str] = {}
    for position, step in enumerate(steps if isinstance(steps, list) else []):
        template = step.get("template") if isinstance(step, dict) else None
        if isinstance(template, dict) and template.get("smarts"):
            index = step.get("step_index")
            recorded[int(index) - 1 if isinstance(index, int) and index > 0 else position] = str(template["smarts"])
    return recorded


def extract_run_summary(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event["event_type"] == "run_summary":
            return event["payload"]
    return None


def summarize_events(events: list[dict[str, Any]]) -> dict[str, int]:
    assistant_turns = 0
    tool_call_turns = 0
    tool_calls = 0
    validation_failures = 0
    searches = 0
    submissions = 0
    invalid_submissions = 0
    for event in events:
        event_type = event["event_type"]
        if event_type == "message" and event["role"] == "assistant":
            assistant_turns += 1
            if event["payload"].get("tool_calls"):
                tool_call_turns += 1
        elif event_type == "tool_call":
            tool_calls += 1
            if event["tool_name"] in SEARCH_TOOLS:
                searches += 1
            if event["tool_name"] == "submit_pathway":
                submissions += 1
        elif event_type == "tool_result":
            result = event["payload"]["result"]
            if result.get("valid") is False or result.get("accepted") is False:
                validation_failures += 1
            if event["tool_name"] == "submit_pathway" and result.get("accepted") is False:
                invalid_submissions += 1
    return {
        "events": len(events),
        "assistant_turns": assistant_turns,
        "tool_call_turns": tool_call_turns,
        "non_tool_turns": assistant_turns - tool_call_turns,
        "tool_calls": tool_calls,
        "search_calls": searches,
        "submission_calls": submissions,
        "invalid_submissions": invalid_submissions,
        "validation_failures": validation_failures,
    }


def write_summary_csv(summaries: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trace_file",
        "model",
        "prompt_version",
        "target_smiles",
        "final_smiles",
        "submitted_final_smiles",
        "accepted",
        "reaction_validation",
        "reaction_valid",
        "building_block_grounded",
        "referenced_building_block_ids",
        "similarity_floor",
        "meets_similarity_floor",
        "run_success",
        "has_final_submission",
        "has_pathway_text",
        "similarity_to_target",
        "identical_to_target",
        "constitutional_similarity_to_target",
        "stereo_aware_similarity_to_target",
        "runtime_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "reasoning_tokens",
        "cost",
        "events",
        "assistant_turns",
        "tool_call_turns",
        "non_tool_turns",
        "tool_calls",
        "search_calls",
        "submission_calls",
        "invalid_submissions",
        "validation_failures",
        "target_hidden_from_stock",
        "hidden_building_block_ids",
        "toxicity_screen",
        "toxicity_screened_count",
        "toxicity_refused_count",
        "toxicity_refused_smiles",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})


def _similarity_floor(tool_config: dict[str, Any] | None) -> StopCondition:
    condition = (tool_config or {}).get("stop_condition")
    if condition is None:
        return DEFAULT_STOP_CONDITION
    try:
        return parse_stop_condition(condition)
    except ValueError:
        # A malformed recorded condition must not abort the rest of the evaluation batch.
        return DEFAULT_STOP_CONDITION


def _infer_target(events: list[dict[str, Any]]) -> str | None:
    """The target the run recorded on its first event that names one."""

    for event in events:
        if event.get("target_smiles"):
            return str(event["target_smiles"])
    return None
