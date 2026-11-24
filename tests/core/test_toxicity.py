# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watchlist matching, submitted-product validation, and hosted-run reporting."""

import argparse
import csv
import json
from pathlib import Path

import pytest

from synclaw.core import db, toxicity
from synclaw.core.reactions import ingest_reaction_templates
from synclaw.runners import model_runner
from synclaw.tasks.datasets import PROMPT_FINGERPRINT
from synclaw.tools.synthesis import SynthesisTools

FLAGGED = "C=O"


@pytest.fixture(autouse=True)
def formaldehyde_watchlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Watch formaldehyde, which the demo reaction below produces from methanol."""

    watchlist = tmp_path / "watchlist.txt"
    watchlist.write_text("SMILES\nC=O\n", encoding="utf-8")
    monkeypatch.setenv(toxicity.ENV_WATCHLIST, str(watchlist))
    monkeypatch.setenv(toxicity.ENV_MODE, "on")
    monkeypatch.delenv(toxicity.ENV_THRESHOLD, raising=False)


@pytest.mark.parametrize(
    ("smiles", "threshold", "refused"),
    [(FLAGGED, 0.8, True), ("CCN", 0.8, False), (FLAGGED, 1.0, False)],
)
def test_watchlist_similarity_must_exceed_threshold(
    monkeypatch, smiles, threshold, refused
) -> None:
    monkeypatch.setenv(toxicity.ENV_THRESHOLD, str(threshold))
    screen = toxicity.ToxicityScreen()

    assert bool(screen.check([("target", smiles)])["refused"]) is refused


def test_screen_normalises_whitespace_and_ignores_non_strings() -> None:
    screen = toxicity.ToxicityScreen()

    assert screen.check([("target", f"  {FLAGGED}\n")])["refused"]
    assert screen.check([("target", 123), ("target", ["C=O"])]) is None


def test_screen_audit_only_counts_molecules_that_were_compared() -> None:
    screen = toxicity.ToxicityScreen()

    assert screen.screen_target("invalid") is None
    block = screen.check([("product", "invalid"), ("product", "CCN"), ("product", FLAGGED)])
    assert block["screened"] == ["CCN", FLAGGED]
    assert [item["smiles"] for item in block["refused"]] == [FLAGGED]


def test_new_screen_loads_the_current_watchlist(tmp_path: Path) -> None:
    path = tmp_path / "custom.txt"
    config = toxicity.ToxicityConfig(mode="on", watchlist=path)
    path.write_text("CCO\n")
    first = toxicity.ToxicityScreen(config)
    path.write_text("CCN\n")
    second = toxicity.ToxicityScreen(config)
    molecules = [("target", "CCO"), ("product", "CCN")]

    assert [item["smiles"] for item in first.check(molecules)["refused"]] == ["CCO"]
    assert [item["smiles"] for item in second.check(molecules)["refused"]] == ["CCN"]
    path.write_text("SMILES\ninvalid\n")
    with pytest.raises(toxicity.WatchlistUnavailable, match="no valid SMILES"):
        toxicity.ToxicityScreen(config)
    path.unlink()
    with pytest.raises(toxicity.WatchlistUnavailable, match="download-public-data"):
        toxicity.ToxicityScreen(config)


def test_screens_built_from_an_unchanged_watchlist_parse_it_once(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "unchanged.txt"
    path.write_text("CCOCC\nCCNCC\n")
    config = toxicity.ToxicityConfig(mode="on", watchlist=path)
    parsed = []
    real = toxicity._fingerprint
    monkeypatch.setattr(
        toxicity, "_fingerprint", lambda smiles: parsed.append(smiles) or real(smiles)
    )

    toxicity.ToxicityScreen(config)
    toxicity.ToxicityScreen(config)

    assert parsed == ["CCOCC", "CCNCC"]


def test_screen_is_opt_in_and_requires_a_watchlist_only_when_enabled(monkeypatch) -> None:
    monkeypatch.delenv(toxicity.ENV_MODE)
    monkeypatch.setenv(toxicity.ENV_WATCHLIST, "/nonexistent/watchlist.txt")

    assert toxicity.ToxicityConfig.resolve().mode == "off"
    assert toxicity.ToxicityScreen().check([("target", FLAGGED)]) is None
    monkeypatch.setenv(toxicity.ENV_MODE, "on")
    with pytest.raises(toxicity.WatchlistUnavailable, match="download-public-data"):
        toxicity.ToxicityScreen()
    # An explicit mode overrides the environment and never loads the watchlist when off.
    assert (
        toxicity.ToxicityScreen(toxicity.ToxicityConfig.resolve(mode="off")).check(
            [("target", FLAGGED)]
        )
        is None
    )


def _tools(tmp_path: Path) -> SynthesisTools:
    (tmp_path / "bbs.txt").write_text("SMILES\nCO\nCCN\nCCCC\n", encoding="utf-8")
    (tmp_path / "templates.txt").write_text(
        "[C:1][O:2]>>[C:1]=[O:2]\n[C:1]=[O:2]>>[C:1][O:2]\n[C:1][C:2]>>[C:1]=[C:2]\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "test.sqlite"
    db.ingest_building_blocks(tmp_path / "bbs.txt", db_path)
    ingest_reaction_templates(tmp_path / "templates.txt", db_path=db_path, library="demo")
    return SynthesisTools(db_path)


@pytest.mark.parametrize("intermediate", [False, True])
def test_route_through_a_flagged_product_is_rejected_with_a_reason(
    tmp_path: Path, intermediate: bool
) -> None:
    steps = [{"reaction_id": "R001", "reactants": ["BB:1"], "product_smiles": FLAGGED}]
    if intermediate:
        steps.append({"reaction_id": "R002", "reactants": ["INT:1"], "product_smiles": "CO"})

    result = _tools(tmp_path).submit_pathway(steps=steps, target_smiles="CC")

    assert result["reaction_validation"] == "valid"
    assert result["accepted"] is False and result["refused"] is True
    assert "dangerous-molecule screen" in result["error_message"]
    assert {item["role"] for item in result["toxicity_screen"]["refused"]} == {
        "intermediate" if intermediate else "product",
    }


def test_unselected_generated_route_products_are_screened(tmp_path: Path, monkeypatch) -> None:
    watchlist = tmp_path / "alternative_watchlist.txt"
    watchlist.write_text("CC=CC\n")
    monkeypatch.setenv(toxicity.ENV_WATCHLIST, str(watchlist))
    tools = _tools(tmp_path)

    result = tools.submit_pathway(
        steps=[
            {
                "reaction_id": "R003",
                "reactants": ["BB:3"],
                "product_smiles": "C=CCC",
            }
        ]
    )

    assert result["reaction_validation"] == "valid"
    assert result["accepted"] is False and result["refused"] is True
    assert "CC=CC" in result["actual_final_products"]
    assert {item["smiles"] for item in result["toxicity_screen"]["refused"]} == {"CC=CC"}


@pytest.mark.parametrize(("building_block", "accepted"), [("BB:1", True), ("BB:2", False)])
def test_direct_purchase_checks_the_purchased_product(
    tmp_path: Path, monkeypatch, building_block, accepted
) -> None:
    watchlist = tmp_path / "purchase_watchlist.txt"
    watchlist.write_text("CCN\n")
    monkeypatch.setenv(toxicity.ENV_WATCHLIST, str(watchlist))

    result = _tools(tmp_path).submit_pathway(building_block=building_block)

    assert result["accepted"] is accepted
    assert result["toxicity_screen"]["screened"] == result["actual_final_products"]
    assert bool(result["toxicity_screen"]["refused"]) is not accepted


@pytest.mark.parametrize(
    ("mode", "target", "expected_requests"),
    [("on", FLAGGED, 0), ("on", "CCN", 1), ("off", FLAGGED, 1)],
)
def test_hosted_target_screening_is_opt_in_and_preserves_trace_results(
    tmp_path: Path,
    monkeypatch,
    mode: str,
    target: str,
    expected_requests: int,
) -> None:
    taskset = tmp_path / "tasks.jsonl"
    taskset.write_text(
        json.dumps(
            {
                "task_id": "fixture",
                "target_smiles": target,
                "system_prompt": "Plan syntheses.",
                "question": "Make the target.",
                "prompt_fingerprint": PROMPT_FINGERPRINT,
            }
        )
        + "\n"
    )
    db.init_db(tmp_path / "test.sqlite")
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    args = parser.parse_args(
        [
            "--model",
            "fixture",
            "--api-mode",
            "chat",
            "--taskset",
            str(taskset),
            "--db",
            str(tmp_path / "test.sqlite"),
            "--trace",
            str(tmp_path / "trace.jsonl"),
            "--summary-output",
            str(tmp_path / "summary.csv"),
            "--max-turns",
            "1",
            "--toxicity-screen",
            mode,
        ]
    )
    monkeypatch.setattr(model_runner, "resolve_api_key", lambda *args, **kwargs: "fixture-key")
    monkeypatch.setattr(db, "validate_model_run_database", lambda *args, **kwargs: None)
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}
            ]
        }

    monkeypatch.setattr(model_runner, "chat_completion", complete)
    summary = model_runner.run_model_from_args(args)

    assert len(requests) == expected_requests
    assert summary["toxicity_screen"] == mode
    assert summary["toxicity_screened_count"] == (1 if mode == "on" else 0)
    assert summary["toxicity_refused_count"] == (1 if expected_requests == 0 else 0)
    events = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
    if mode == "on":
        assert events[0]["event_type"] == (
            "target_refused" if expected_requests == 0 else "target_screened"
        )
    with Path(args.summary_output).open() as handle:
        row = next(csv.DictReader(handle))
    assert row["target_smiles"] == target
    assert row["toxicity_screen"] == mode
    assert int(row["toxicity_refused_count"]) == summary["toxicity_refused_count"]


def test_replay_keeps_a_refusal_the_run_recorded(tmp_path: Path) -> None:
    from synclaw.evaluation.evaluate import replay_selected_submission
    from synclaw.rollout.policy import prepare_tool_call

    tools = _tools(tmp_path)
    steps = [{"reaction_id": "R001", "reactants": ["BB:1"], "product_smiles": FLAGGED}]
    live = tools.submit_pathway(steps=steps, target_smiles="CC")
    assert live["refused"] is True
    call = prepare_tool_call(
        raw_name="submit_pathway",
        raw_arguments={"steps": steps},
        raw_tool_call={},
        event_index=0,
    )
    events = [
        {
            "event_type": "tool_call",
            "tool_name": "submit_pathway",
            "target_smiles": "CC",
            "payload": call.event_payload,
        },
        {
            "event_type": "tool_result",
            "tool_name": "submit_pathway",
            "target_smiles": "CC",
            "payload": {"result": live, "outcome": "ok"},
        },
    ]

    replayed = replay_selected_submission(events, tools.db_path, "CC")

    assert replayed is not None and replayed["accepted"] is False


def test_enabled_screen_keeps_the_initial_messages_in_the_trace(
    tmp_path: Path, monkeypatch
) -> None:
    taskset = tmp_path / "tasks.jsonl"
    taskset.write_text(
        json.dumps(
            {
                "task_id": "fixture",
                "target_smiles": "CCN",
                "system_prompt": "Plan syntheses.",
                "question": "Make the target.",
                "prompt_fingerprint": PROMPT_FINGERPRINT,
            }
        )
        + "\n"
    )
    db.init_db(tmp_path / "test.sqlite")
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    args = parser.parse_args(
        [
            "--model",
            "fixture",
            "--api-mode",
            "chat",
            "--taskset",
            str(taskset),
            "--db",
            str(tmp_path / "test.sqlite"),
            "--trace",
            str(tmp_path / "trace.jsonl"),
            "--max-turns",
            "1",
        ]
    )
    monkeypatch.setattr(model_runner, "resolve_api_key", lambda *args, **kwargs: "fixture-key")
    monkeypatch.setattr(db, "validate_model_run_database", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model_runner,
        "chat_completion",
        lambda **kwargs: {
            "choices": [
                {"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}
            ]
        },
    )
    model_runner.run_model_from_args(args)

    events = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
    assert events[0]["event_type"] == "target_screened"
    assert [e["role"] for e in events if e["event_type"] == "message"][:2] == ["system", "user"]


def test_refused_submission_is_compacted_without_the_screen_audit(tmp_path: Path) -> None:
    from synclaw.tools.compaction import compact_tool_result

    steps = [{"reaction_id": "R001", "reactants": ["BB:1"], "product_smiles": FLAGGED}]
    result = _tools(tmp_path).submit_pathway(steps=steps, target_smiles="CC")

    compact = compact_tool_result("submit_pathway", result)

    assert compact["refused"] is True and "dangerous-molecule screen" in compact["error_message"]
    assert "steps" not in compact and toxicity.BLOCK_KEY not in compact


def test_refused_direct_purchase_carries_no_similarity(tmp_path: Path, monkeypatch) -> None:
    watchlist = tmp_path / "purchase_watchlist.txt"
    watchlist.write_text("CCN\n")
    monkeypatch.setenv(toxicity.ENV_WATCHLIST, str(watchlist))

    result = _tools(tmp_path).submit_pathway(building_block="BB:2", target_smiles="CC")

    assert result["accepted"] is False and result["refused"] is True
    assert (
        result["similarity_to_target"] is None
        and result["constitutional_similarity_to_target"] is None
    )


def test_checkpoint_from_a_different_screening_mode_is_not_resumed() -> None:
    from synclaw.rollout import policy

    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    args = parser.parse_args(["--model", "fixture", "--toxicity-screen", "on"])
    checkpoint = {
        "model": "fixture",
        "provider": args.provider,
        "api_mode": args.api_mode,
        "task_id": "t",
        "target_smiles": "CC",
    }
    task = {"task_id": "t", "target_smiles": "CC"}
    config = policy.RolloutPolicyConfig()

    assert not model_runner.checkpoint_matches_run(checkpoint, args, task, config)
    assert not model_runner.checkpoint_matches_run(
        {**checkpoint, "toxicity_screen": "off"}, args, task, config
    )
    screened = {
        **checkpoint,
        "toxicity_screen": "on",
        "toxicity_watchlist": str(toxicity.ToxicityConfig.resolve().watchlist.resolve()),
        "toxicity_threshold": toxicity.DEFAULT_THRESHOLD,
    }
    assert model_runner.checkpoint_matches_run(screened, args, task, config)
    assert not model_runner.checkpoint_matches_run(
        {**screened, "toxicity_threshold": 0.5}, args, task, config
    )
    assert not model_runner.checkpoint_matches_run(
        {**screened, "toxicity_watchlist": "/elsewhere.txt"}, args, task, config
    )


@pytest.mark.parametrize(
    ("name", "value"), [(toxicity.ENV_THRESHOLD, "0,8"), (toxicity.ENV_THRESHOLD, "2")]
)
def test_bad_screen_settings_exit_with_a_clean_error(
    tmp_path: Path, monkeypatch, capsys, name: str, value: str
) -> None:
    from synclaw import cli

    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            ["tool", "submit-pathway", "--steps-json", "[]", "--db", str(tmp_path / "test.sqlite")]
        )

    assert exit_info.value.code == 1 and "error:" in capsys.readouterr().err


def test_screen_mode_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv(toxicity.ENV_MODE, "ON")
    assert toxicity.ToxicityConfig.resolve().mode == "on"


def test_tools_that_never_screen_do_not_need_the_watchlist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(toxicity.ENV_WATCHLIST, str(tmp_path / "missing.txt"))
    monkeypatch.setenv(toxicity.ENV_THRESHOLD, "not-a-number")
    tools = _tools(tmp_path)

    assert tools.get_building_block(1)["found"] is True
    with pytest.raises(RuntimeError):
        tools.screen
