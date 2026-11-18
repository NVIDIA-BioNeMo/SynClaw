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
from synclaw.rollout.policy import (
    DEFAULT_STOP_CONDITION,
    StopCondition,
    is_better_accepted_submission,
    meets_submission_floor,
    parse_stop_condition,
    submission_identical_to_target,
    submission_similarity_to_target,
)
from synclaw.token_usage import int_or_none, trace_token_usage
from synclaw.tools.synthesis import (
    hidden_target_summary_fields,
    score_pathway_against_target,
    select_final_product,
)

FINAL_EVENT_TYPES = {"final", "final_pathway", "pathway", "selected_submission", "submission", "submit_pathway"}
TOOL_CALL_TYPES = {"tool_call", "tool_use", "function_call"}
TOOL_RESULT_TYPES = {"tool_result", "function_result"}


def evaluate_trace_file(
    trace_path: str | Path,
    db_path: str | Path = db.DEFAULT_DB_PATH,
    model: str = "unknown",
    prompt_version: str | None = None,
    target_smiles: str | None = None,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(trace_path)
    events = list(db.iter_jsonl(path))
    if not events:
        raise ValueError(f"{path} contains no JSONL events")

    inferred_target = target_smiles or _infer_target(events)
    run_summary = extract_run_summary(events)
    hidden_target = hidden_target_summary_fields(events)
    final = extract_selected_submission(events, target_smiles=inferred_target)
    if final is None:
        final = replay_selected_submission(
            events,
            db_path=db_path,
            target_smiles=inferred_target,
            # Partial traces have no run_summary, so re-scoring them needs the selection passed in.
            library=(tool_config or {}).get("reaction_library")
            or (run_summary or {}).get("reaction_library"),
            hide_target=hidden_target["target_hidden_from_stock"],
        )
    if final is None:
        final = extract_final_submission(events, target_smiles=inferred_target)
    pathway_text = final.get("pathway_text") if final else None
    final_smiles = final.get("final_smiles") if final else None
    submitted_final_smiles = final.get("submitted_final_smiles") if final else None
    accepted = final.get("accepted") if final else None
    reaction_validation = final.get("reaction_validation") if final else None
    reaction_valid = reaction_validation == "valid" if reaction_validation is not None else None
    building_block_grounded = final.get("building_block_grounded") if final else None
    referenced_building_block_ids = final.get("referenced_building_block_ids") if final else None
    # Traces without an `accepted` field carry no verdict, so only drop the
    # product when the validator explicitly rejected the submission.
    if accepted is not None and not accepted:
        final_smiles = None
    elif pathway_text and not final_smiles and not final.get("structured_submission"):
        candidates = chem.extract_smiles_candidates(str(pathway_text))
        final_smiles = candidates[-1] if candidates else None

    final_validation = chem.validate_smiles(final_smiles) if final_smiles else None
    if final_validation and not final_validation["valid"]:
        final_smiles = None

    scoring = score_pathway_against_target(
        [final_smiles] if final_smiles else [],
        target_smiles=inferred_target,
        target_validation=chem.validate_smiles(inferred_target) if inferred_target else None,
        preferred_smiles=None,
        accepted=accepted is not False,
    )
    similarity_to_target = scoring["similarity_to_target"]
    identical_to_target = scoring["identical_to_target"]
    runtime_seconds = _float_or_none(run_summary.get("runtime_seconds") if run_summary else None)
    token_usage = trace_token_usage(events, run_summary=run_summary)
    similarity_floor = _similarity_floor(tool_config)
    meets_similarity_floor = accepted is True and meets_submission_floor(
        similarity_to_target, similarity_floor, identical_to_target=identical_to_target,
    )
    run_success = bool(accepted)

    summary = {
        "trace_file": str(path),
        "model": model,
        "prompt_version": prompt_version,
        "target_smiles": inferred_target,
        "final_smiles": final_smiles,
        "submitted_final_smiles": submitted_final_smiles,
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
        "has_pathway_text": bool(pathway_text),
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

    The best accepted submission is kept, so a later, worse one cannot replace it. With none
    accepted, the last submission stands. ``hide_target`` replays with the target out of
    stock, as a run that hid it validated its submissions.
    """

    submit_calls = [
        event for event in events
        if _event_type(event) in TOOL_CALL_TYPES and _tool_name(event) == "submit_pathway"
    ]
    best: dict[str, Any] | None = None
    best_identical: bool | None = None
    best_similarity: float | None = None
    last: dict[str, Any] | None = None
    for ordinal, event in enumerate(submit_calls):
        payload = _payload(event)
        if not isinstance(payload, dict):
            continue
        last = _replay_submission(
            events,
            payload,
            ordinal,
            db_path,
            target_smiles,
            library=library,
            hide_target=hide_target,
        )
        if last is not None and _recorded_refused(events, ordinal):
            # Replay runs without the screen, so keep the refusal the run itself recorded.
            last = {**last, "accepted": False, "refused": True}
        if last is None or not last.get("accepted"):
            continue
        similarity = submission_similarity_to_target(last, target_smiles)
        identical = submission_identical_to_target(last, target_smiles) is True
        if best is None or is_better_accepted_submission(identical, similarity, best_identical, best_similarity):
            best, best_identical, best_similarity = last, identical, similarity
    selected = best or last
    if selected is None:
        return None
    return {
        "pathway_text": selected.get("pathway_text"),
        "final_smiles": selected.get("actual_final_smiles"),
        "submitted_final_smiles": selected.get("submitted_final_smiles"),
        "accepted": selected.get("accepted"),
        "reaction_validation": selected.get("reaction_validation"),
        "pathway_type": selected.get("pathway_type"),
        "direct_building_block": selected.get("direct_building_block"),
        "direct_building_block_ref": selected.get("direct_building_block_ref"),
        "building_block_grounded": selected.get("building_block_grounded"),
        "referenced_building_block_ids": selected.get("referenced_building_block_ids"),
        "constitutional_similarity_to_target": selected.get("constitutional_similarity_to_target"),
        "stereo_aware_similarity_to_target": selected.get("stereo_aware_similarity_to_target"),
        "steps": selected.get("steps"),
        "structured_submission": True,
    }


def _replay_submission(
    events: list[dict[str, Any]],
    payload: dict[str, Any],
    ordinal: int,
    db_path: str | Path,
    target_smiles: str | None,
    *,
    library: LibrarySelection,
    hide_target: bool,
) -> dict[str, Any] | None:
    arguments = payload.get("dispatch_arguments") if isinstance(payload.get("dispatch_arguments"), dict) else None
    if arguments is None and isinstance(payload.get("arguments"), dict):
        arguments = dict(payload["arguments"])
        arguments.pop("target_smiles", None)
        arguments.pop("final_smiles", None)
    if not isinstance(arguments, dict):
        return None
    steps = arguments.get("steps")
    if steps in (None, "") and arguments.get("building_block"):
        steps = []
    if not isinstance(steps, list):
        return None
    # Template IDs change between catalog versions, but each step's result records the
    # SMARTS it ran, so an unknown ID is resolved to the current template with that SMARTS.
    recorded_smarts = _recorded_step_smarts(events, call_ordinal=ordinal)
    resolved_steps = []
    with db.session(db_path) as conn:
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                return None
            reaction_id = str(step.get("reaction_id") or step.get("template_id") or "")
            if reactions.is_reaction_smarts(reaction_id):
                # submit_pathway grounds a reaction SMARTS to a library template itself.
                resolved_steps.append(step)
                continue
            template = db.get_reaction_template(conn, reaction_id, library=library) if reaction_id else None
            if template is None and index in recorded_smarts:
                template = db.get_reaction_template_by_smarts(conn, recorded_smarts[index], library=library)
                if template is not None:
                    step = {**step, "reaction_id": template["template_id"]}
            if template is None:
                return None
            resolved_steps.append(step)
    from synclaw.tools.synthesis import SynthesisTools

    # Replay re-scores a recorded trace without the screen; recorded refusals are kept by the caller.
    return SynthesisTools(
        db_path,
        reaction_libraries=library,
        toxicity=ToxicityConfig(mode=MODE_OFF),
        hidden_target_smiles=target_smiles if hide_target else None,
        hidden_building_block_ids=frozenset(
            json.loads(hidden_target_summary_fields(events)["hidden_building_block_ids"] or "[]")
        ),
    ).submit_pathway(
        steps=resolved_steps,
        building_block=arguments.get("building_block"),
        pathway_text=arguments.get("pathway_text"),
        target_smiles=target_smiles,
    )


def extract_selected_submission(
    events: list[dict[str, Any]],
    target_smiles: str | None = None,
) -> dict[str, Any] | None:
    selected_events = [event for event in events if _event_type(event) == "selected_submission"]
    if not selected_events:
        return None
    return extract_final_submission(selected_events, target_smiles=target_smiles)


def _recorded_refused(events: list[dict[str, Any]], ordinal: int) -> bool:
    """Whether the run's own screen refused the n-th submit_pathway call."""

    results = [
        event for event in events
        if _event_type(event) in TOOL_RESULT_TYPES and _tool_name(event) == "submit_pathway"
    ]
    if ordinal >= len(results):
        return False
    payload = _payload(results[ordinal])
    result = payload.get("result") if isinstance(payload, dict) else None
    return isinstance(result, dict) and result.get("refused") is True


def _recorded_step_smarts(events: list[dict[str, Any]], *, call_ordinal: int) -> dict[int, str]:
    """Map step positions to template SMARTS from the result of the given submit_pathway call.

    Results are appended in call order, so the n-th submit_pathway result belongs to the n-th call.
    """

    results = [
        event for event in events
        if _event_type(event) in TOOL_RESULT_TYPES and _tool_name(event) == "submit_pathway"
    ]
    if call_ordinal >= len(results):
        return {}
    payload = _payload(results[call_ordinal])
    result = payload.get("result") if isinstance(payload, dict) else None
    steps = result.get("steps") if isinstance(result, dict) else None
    recorded: dict[int, str] = {}
    for position, step in enumerate(steps if isinstance(steps, list) else []):
        template = step.get("template") if isinstance(step, dict) else None
        if isinstance(template, dict) and template.get("smarts"):
            index = step.get("step_index")
            recorded[int(index) - 1 if isinstance(index, int) and index > 0 else position] = str(template["smarts"])
    return recorded


def extract_run_summary(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if _event_type(event) != "run_summary":
            continue
        payload = _payload(event)
        if isinstance(payload, dict):
            return payload
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
        event_type = _event_type(event)
        tool_name = _tool_name(event)
        payload = _payload(event)
        if event_type == "message" and event.get("role") == "assistant":
            assistant_turns += 1
            if isinstance(payload, dict) and payload.get("tool_calls"):
                tool_call_turns += 1
        is_call = event_type in TOOL_CALL_TYPES
        is_result = event_type in TOOL_RESULT_TYPES or (not is_call and "result" in event)
        if is_call:
            tool_calls += 1
            if tool_name in {
                "search_building_blocks",
                "search_reaction_templates",
                "search_reaction_templates_by_molecule",
            }:
                searches += 1
            if tool_name == "submit_pathway":
                submissions += 1
        if isinstance(payload, dict):
            result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
            if result.get("valid") is False or result.get("accepted") is False:
                validation_failures += 1
            if (
                is_result
                and tool_name == "submit_pathway"
                and result.get("accepted") is False
            ):
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


def extract_final_submission(
    events: list[dict[str, Any]],
    target_smiles: str | None = None,
) -> dict[str, Any] | None:
    for event in reversed(events):
        event_type = _event_type(event)
        payload = _payload(event)
        tool_name = _tool_name(event)
        candidates = []
        if isinstance(payload, dict):
            candidates.append(payload)
            if isinstance(payload.get("result"), dict):
                candidates.append(payload["result"])
            if isinstance(payload.get("arguments"), dict):
                candidates.append(payload["arguments"])
        if event_type in FINAL_EVENT_TYPES or tool_name == "submit_pathway":
            for candidate in candidates:
                pathway_text = _first_present(candidate, ("pathway_text", "pathway", "route", "content", "text"))
                structured_submission = tool_name == "submit_pathway" or "accepted" in candidate or "reaction_validation" in candidate
                submit_call_arguments = tool_name == "submit_pathway" and event_type in TOOL_CALL_TYPES
                has_actual_final_field = "actual_final_smiles" in candidate
                if has_actual_final_field:
                    final_smiles = candidate.get("actual_final_smiles")
                elif structured_submission:
                    final_smiles = _final_smiles_from_structured_steps(
                        candidate, target_smiles=target_smiles
                    )
                elif submit_call_arguments:
                    final_smiles = None
                else:
                    final_smiles = _first_present(candidate, ("final_smiles", "product_smiles", "product"))
                submitted_final_smiles = _first_present(candidate, ("submitted_final_smiles",))
                accepted = candidate.get("accepted")
                reaction_validation = candidate.get("reaction_validation")
                pathway_type = candidate.get("pathway_type")
                direct_building_block = candidate.get("direct_building_block")
                direct_building_block_ref = candidate.get("direct_building_block_ref")
                building_block_grounded = candidate.get("building_block_grounded")
                referenced_building_block_ids = candidate.get("referenced_building_block_ids")
                steps = candidate.get("steps")
                if (
                    not final_smiles
                    and isinstance(steps, list)
                    and steps
                    and not has_actual_final_field
                    and not submit_call_arguments
                    and not structured_submission
                ):
                    last_step = steps[-1]
                    if isinstance(last_step, dict):
                        final_smiles = _first_present(last_step, ("product_smiles", "product"))
                if not pathway_text and isinstance(steps, list):
                    pathway_text = json.dumps({"steps": steps}, sort_keys=True)
                if pathway_text:
                    return {
                        "pathway_text": pathway_text,
                        "final_smiles": final_smiles,
                        "submitted_final_smiles": submitted_final_smiles,
                        "accepted": accepted,
                        "reaction_validation": reaction_validation,
                        "pathway_type": pathway_type,
                        "direct_building_block": direct_building_block,
                        "direct_building_block_ref": direct_building_block_ref,
                        "building_block_grounded": building_block_grounded,
                        "referenced_building_block_ids": referenced_building_block_ids,
                        "constitutional_similarity_to_target": candidate.get(
                            "constitutional_similarity_to_target"
                        ),
                        "stereo_aware_similarity_to_target": candidate.get(
                            "stereo_aware_similarity_to_target"
                        ),
                        "steps": steps,
                        "structured_submission": structured_submission,
                    }
    return None


def _final_smiles_from_structured_steps(
    candidate: dict[str, Any],
    target_smiles: str | None,
) -> str | None:
    steps = candidate.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    final_step = steps[-1]
    if not isinstance(final_step, dict):
        return None
    if final_step.get("valid") is True:
        actual_product = final_step.get("actual_product_smiles")
        if actual_product:
            validation = chem.validate_smiles(str(actual_product))
            if validation["valid"]:
                return str(validation["canonical_smiles"])
        product = final_step.get("product")
        if isinstance(product, dict) and product.get("valid") and product.get("canonical_smiles"):
            return str(product["canonical_smiles"])
    generated = _valid_smiles_list(final_step.get("generated_products"))
    return select_final_product(generated, target_smiles=target_smiles, preferred_smiles=None)


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


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_ints(*values: Any) -> int | None:
    total = 0
    saw_value = False
    for value in values:
        integer = int_or_none(value)
        if integer is None:
            continue
        total += integer
        saw_value = True
    return total if saw_value else None


def _similarity_floor(tool_config: dict[str, Any] | None) -> StopCondition:
    condition = (tool_config or {}).get("stop_condition")
    if condition is None:
        return DEFAULT_STOP_CONDITION
    try:
        return parse_stop_condition(condition)
    except ValueError:
        # A malformed recorded condition must not abort the rest of the evaluation batch.
        return DEFAULT_STOP_CONDITION


def _valid_smiles_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    valid: list[str] = []
    for value in values:
        validation = chem.validate_smiles(str(value))
        if validation["valid"]:
            valid.append(str(validation["canonical_smiles"]))
    return sorted(set(valid))


def _infer_target(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        target = _event_target(event)
        if target:
            return target
    return None


def _event_target(event: dict[str, Any]) -> str | None:
    payload = _payload(event)
    for source in (event, payload if isinstance(payload, dict) else {}):
        value = _first_present(source, ("target_smiles", "target", "input_smiles"))
        if value:
            return str(value)
    return None


def _event_type(event: dict[str, Any]) -> str | None:
    value = event.get("event_type") or event.get("type")
    return str(value) if value is not None else None


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            return {"text": payload}
    if isinstance(payload, dict):
        return payload
    return event


def _tool_name(event: dict[str, Any]) -> str | None:
    payload = _payload(event)
    for source in (event, payload):
        value = _first_present(source, ("tool_name", "name", "function_name"))
        if value:
            return str(value)
    return None


def _first_present(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = source.get(key)
        if value not in (None, ""):
            return value
    return None
