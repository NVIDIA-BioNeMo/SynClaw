# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hiding the target from stock, so an exact match needs a real synthesis route."""

import argparse
import csv
import json
from pathlib import Path

import pytest

from synclaw.core import db, fingerprint_index
from synclaw.core.reactions import ingest_reaction_templates
from synclaw.evaluation.evaluate import replay_selected_submission
from synclaw.rollout.policy import prepare_tool_call
from synclaw.runners import model_runner
from synclaw.tasks.datasets import PROMPT_FINGERPRINT
from synclaw.tools.synthesis import (
    SynthesisTools,
    hidden_target_event,
    hidden_target_summary_fields,
)

TARGET = "CO"  # BB:1
ANALOGUE = "CCO"  # BB:2
# Oxidise the target, then reduce it back: a route that only buys the target.
ROUND_TRIP = [
    {"reaction_id": "R001", "reactants": ["BB:1"], "product_smiles": "C=O"},
    {"reaction_id": "R002", "reactants": ["INT:1"], "product_smiles": TARGET},
]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    # 2-hydroxypyridine and 2-pyridone are one molecule under Standard InChI.
    (tmp_path / "bbs.txt").write_text(
        f"SMILES\n{TARGET}\n{ANALOGUE}\nOc1ccccn1\nO=c1cccc[nH]1\n", encoding="utf-8"
    )
    (tmp_path / "templates.txt").write_text(
        "[C:1][O:2]>>[C:1]=[O:2]\n[C:1]=[O:2]>>[C:1][O:2]\n", encoding="utf-8"
    )
    path = tmp_path / "test.sqlite"
    db.ingest_building_blocks(tmp_path / "bbs.txt", path)
    ingest_reaction_templates(tmp_path / "templates.txt", db_path=path, library="demo")
    db.prepare_building_block_search(path)
    return path


def _tools(db_path: Path, hidden: str | None = TARGET) -> SynthesisTools:
    return SynthesisTools(db_path, hidden_target_smiles=hidden)


def _submit_call(arguments: dict) -> dict:
    """A submit_pathway call event as the rollout loop records it."""

    call = prepare_tool_call(
        raw_name="submit_pathway",
        raw_arguments=arguments,
        raw_tool_call={},
        event_index=0,
    )
    return {
        "event_type": "tool_call",
        "tool_name": "submit_pathway",
        "target_smiles": TARGET,
        "payload": call.event_payload,
    }


def test_hidden_ids_cover_every_entry_scored_as_the_target(db_path: Path) -> None:
    assert _tools(db_path).hidden_building_block_ids == {1}
    assert _tools(db_path, "Oc1ccccn1").hidden_building_block_ids == {3, 4}
    assert _tools(db_path, "CCCC").hidden_building_block_ids == frozenset()
    assert _tools(db_path, None).hidden_building_block_ids == frozenset()


def test_search_and_lookup_treat_the_target_as_out_of_stock(db_path: Path) -> None:
    tools = _tools(db_path)

    exact = tools.search_building_blocks(TARGET, limit=10, min_similarity=0.0)
    assert exact["exact_match_count"] == 0
    assert 1 not in {row["id"] for row in exact["results"]}
    near = tools.search_building_blocks(ANALOGUE, limit=10, min_similarity=0.0)
    assert 1 not in {row["id"] for row in near["results"]}
    assert 2 in {row["id"] for row in near["results"]}
    browse = tools.search_building_blocks("*", limit=10)
    assert [row["id"] for row in browse["results"]] == [2, 3, 4]
    assert tools.get_building_block(1)["found"] is False
    assert tools.get_building_block(2)["found"] is True


@pytest.mark.parametrize("building_block", ["BB:1", TARGET])
def test_the_target_cannot_be_bought_directly(db_path: Path, building_block: str) -> None:
    result = _tools(db_path).submit_pathway(building_block=building_block, target_smiles=TARGET)

    assert result["accepted"] is False
    assert result["similarity_to_target"] is None


def test_an_analogue_can_still_be_bought(db_path: Path) -> None:
    result = _tools(db_path).submit_pathway(building_block="BB:2", target_smiles=TARGET)

    assert result["accepted"] is True
    assert result["similarity_to_target"] < 1.0


def test_the_target_cannot_be_a_starting_reactant(db_path: Path) -> None:
    unhidden = _tools(db_path, None).submit_pathway(steps=ROUND_TRIP, target_smiles=TARGET)
    hidden = _tools(db_path).submit_pathway(steps=ROUND_TRIP, target_smiles=TARGET)

    assert unhidden["accepted"] is True and unhidden["final_smiles"] == TARGET
    assert hidden["accepted"] is False
    assert "unknown building block reference: BB:1" in hidden["issues"][0]
    simulated = _tools(db_path).simulate_reaction("R001", ["BB:1"])
    assert simulated["valid"] is False


def test_reverse_grounding_does_not_offer_the_target(db_path: Path) -> None:
    def grounded(tools: SynthesisTools) -> bool:
        result = tools.simulate_reverse_reaction("R001", "C=O")
        return result["reverse_planning_summary"]["has_fully_grounded_candidate"]

    assert grounded(_tools(db_path, None)) is True
    assert grounded(_tools(db_path)) is False


def test_replay_keeps_the_target_hidden_when_the_run_did(db_path: Path) -> None:
    call = _submit_call({"building_block": "BB:1"})
    hidden_event = hidden_target_event(TARGET, frozenset({1}))

    assert replay_selected_submission([call], db_path, TARGET)["accepted"] is True
    replayed = replay_selected_submission([hidden_event, call], db_path, TARGET, hide_target=True)
    assert replayed["accepted"] is False


def test_summary_fields_report_the_setting_and_hidden_ids() -> None:
    assert hidden_target_summary_fields([]) == {
        "target_hidden_from_stock": False,
        "hidden_building_block_ids": None,
    }
    assert hidden_target_summary_fields([hidden_target_event(TARGET, frozenset({4, 3}))]) == {
        "target_hidden_from_stock": True,
        "hidden_building_block_ids": "[3, 4]",
    }


@pytest.mark.parametrize(
    ("flags", "hidden_ids", "hide_target"),
    [
        (["--hide-target-building-block"], frozenset({1}), True),
        (["--hide-building-block-id", "2"], frozenset({2}), False),
        (
            ["--hide-target-building-block", "--hide-building-block-id", "2"],
            frozenset({1, 2}),
            True,
        ),
        (["--hide-building-block-similarity", "1"], frozenset({1}), True),
    ],
)
def test_hosted_run_records_the_hidden_target(
    tmp_path: Path,
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    hidden_ids: frozenset[int],
    hide_target: bool,
) -> None:
    taskset = tmp_path / "tasks.jsonl"
    taskset.write_text(
        json.dumps(
            {
                "task_id": "fixture",
                "target_smiles": TARGET,
                "system_prompt": "Plan syntheses.",
                "question": "Make the target.",
                "prompt_fingerprint": PROMPT_FINGERPRINT,
            }
        )
        + "\n"
    )
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
            str(db_path),
            "--trace",
            str(tmp_path / "trace.jsonl"),
            "--summary-output",
            str(tmp_path / "summary.csv"),
            "--max-turns",
            "1",
            *flags,
        ]
    )
    monkeypatch.setattr(model_runner, "resolve_api_key", lambda *args, **kwargs: "fixture-key")
    monkeypatch.setattr(
        model_runner.db, "validate_model_run_database", lambda *args, **kwargs: None
    )
    requests = []
    scans = []
    original = fingerprint_index.FingerprintIndex.top_k

    def tracked(self, query, k, **kwargs):
        if k is None:
            scans.append(True)
        return original(self, query, k, **kwargs)

    monkeypatch.setattr(fingerprint_index.FingerprintIndex, "top_k", tracked)

    def completion(**kwargs):
        # The runner must resolve similarity exclusions before the first provider call.
        if args.hide_building_block_similarity is not None:
            assert scans == [True]
        requests.append(True)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}
            ]
        }

    monkeypatch.setattr(
        model_runner,
        "chat_completion",
        completion,
    )

    summary = model_runner.run_model_from_args(args)

    events = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
    assert requests
    assert events[0] == hidden_target_event(
        TARGET,
        hidden_ids,
        hide_target=hide_target,
        similarity_threshold=args.hide_building_block_similarity,
    )
    assert summary["target_hidden_from_stock"] is hide_target
    assert summary["hidden_building_block_ids"] == json.dumps(sorted(hidden_ids))
    with Path(args.summary_output).open() as handle:
        row = next(csv.DictReader(handle))
    assert row["target_hidden_from_stock"] == str(hide_target)
    assert row["hidden_building_block_ids"] == json.dumps(sorted(hidden_ids))


def test_resume_requires_the_same_hiding_setting() -> None:
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    args = parser.parse_args(["--model", "fixture", "--hide-target-building-block"])
    task = {"task_id": "fixture", "target_smiles": TARGET}
    checkpoint = {"model": "fixture", "provider": args.provider, "api_mode": args.api_mode, **task}
    config = model_runner.policy.RolloutPolicyConfig()

    assert not model_runner.checkpoint_matches_run(checkpoint, args, task, config)
    checkpoint["hide_target_building_block"] = True
    assert model_runner.checkpoint_matches_run(checkpoint, args, task, config)


@pytest.mark.parametrize("target", [None, TARGET, "CCCC"])
def test_fixed_ids_apply_to_every_target(db_path: Path, target: str | None) -> None:
    tools = SynthesisTools(
        db_path, hidden_target_smiles=target, hidden_building_block_ids=frozenset({2})
    )
    assert tools.hidden_building_block_ids == ({1, 2} if target == TARGET else {2})
    assert not tools.get_building_block(2)["found"]
    for query in (ANALOGUE, TARGET, "*"):
        results = tools.search_building_blocks(query, limit=10, min_similarity=0.0)
        assert 2 not in {row["id"] for row in results["results"]}
    for reference in ("BB:2", ANALOGUE):
        assert not tools.submit_pathway(building_block=reference)["accepted"]


def test_fixed_ids_block_reactants_and_reverse_grounding(db_path: Path) -> None:
    tools = SynthesisTools(db_path, hidden_building_block_ids=frozenset({1}))
    assert not tools.submit_pathway(steps=ROUND_TRIP, target_smiles=TARGET)["accepted"]
    raw_steps = [{**ROUND_TRIP[0], "reactants": [TARGET]}, ROUND_TRIP[1]]
    assert not tools.submit_pathway(steps=raw_steps, target_smiles=TARGET)["accepted"]
    assert not tools.simulate_reaction("R001", ["BB:1"])["valid"]
    reverse = tools.simulate_reverse_reaction("R001", "C=O")
    assert not reverse["reverse_planning_summary"]["has_fully_grounded_candidate"]


def test_fixed_ids_replay_without_target_hiding(db_path: Path) -> None:
    event = hidden_target_event(TARGET, frozenset({2}), hide_target=False)
    call = _submit_call({"building_block": "BB:2"})
    assert replay_selected_submission([call], db_path, TARGET)["accepted"]
    assert not replay_selected_submission([event, call], db_path, TARGET)["accepted"]
    assert hidden_target_summary_fields([event]) == {
        "target_hidden_from_stock": False,
        "hidden_building_block_ids": "[2]",
    }


@pytest.mark.parametrize("bb_id", [0, -1, 999])
def test_fixed_ids_must_exist(db_path: Path, bb_id: int) -> None:
    tools = SynthesisTools(db_path, hidden_building_block_ids=frozenset({bb_id}))
    with pytest.raises(RuntimeError, match="IDs must exist"):
        tools.hidden_building_block_ids


def test_resume_requires_the_same_fixed_set() -> None:
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    args = parser.parse_args(
        [
            "--model",
            "fixture",
            "--hide-building-block-id",
            "2",
            "--hide-building-block-id",
            "1",
            "--hide-building-block-id",
            "2",
        ]
    )
    task = {"task_id": "fixture", "target_smiles": TARGET}
    checkpoint = {"model": "fixture", "provider": args.provider, "api_mode": args.api_mode, **task}
    config = model_runner.policy.RolloutPolicyConfig()
    assert not model_runner.checkpoint_matches_run(checkpoint, args, task, config)
    checkpoint["hide_building_block_ids"] = [1, 2]
    assert model_runner.checkpoint_matches_run(checkpoint, args, task, config)
    checkpoint["hide_building_block_ids"] = [1]
    assert not model_runner.checkpoint_matches_run(checkpoint, args, task, config)


def test_similarity_filter_allows_synthesizing_a_hidden_product(tmp_path: Path) -> None:
    (tmp_path / "bbs.txt").write_text("SMILES\nCO\nC=O\n")
    (tmp_path / "templates.txt").write_text("[C:1]=[O:2]>>[C:1][O:2]\n")
    path = tmp_path / "test.sqlite"
    db.ingest_building_blocks(tmp_path / "bbs.txt", path)
    ingest_reaction_templates(tmp_path / "templates.txt", db_path=path, library="demo")
    db.prepare_building_block_search(path)
    tools = SynthesisTools(path, hidden_target_smiles="CO", hidden_similarity=1)
    assert tools.hidden_building_block_ids == {1}
    result = tools.submit_pathway(
        steps=[{"reaction_id": "R001", "reactants": ["BB:2"], "product_smiles": "CO"}],
        target_smiles="CO",
    )
    assert result["accepted"] and result["similarity_to_target"] == 1


def test_resume_requires_the_same_similarity_cutoff() -> None:
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    args = parser.parse_args(["--model", "fixture", "--hide-building-block-similarity", "0.95"])
    task = {"task_id": "fixture", "target_smiles": TARGET}
    checkpoint = {
        "model": "fixture",
        "provider": args.provider,
        "api_mode": args.api_mode,
        **task,
        "hide_target_building_block": True,
        "hide_building_block_similarity": 0.9,
    }
    config = model_runner.policy.RolloutPolicyConfig()
    assert not model_runner.checkpoint_matches_run(checkpoint, args, task, config)
    checkpoint["hide_building_block_similarity"] = 0.95
    assert model_runner.checkpoint_matches_run(checkpoint, args, task, config)


def test_hidden_ids_cover_open_stereo_isomers_and_unmerged_tautomers(tmp_path: Path) -> None:
    # Standard InChI keeps cyclohexanone and its enol apart; exact-match scoring does not.
    (tmp_path / "bbs.txt").write_text(
        "SMILES\nF[C@H](Cl)Br\nF[C@@H](Cl)Br\nFC(Cl)Br\nFC(Cl)I\nO=C1CCCCC1\nOC1=CCCCC1\n"
        "C[C@H](F)C(Cl)CC\nOC1=CC[C@H](F)C(Cl)C1\n",
        encoding="utf-8",
    )
    path = tmp_path / "test.sqlite"
    db.ingest_building_blocks(tmp_path / "bbs.txt", path)

    assert _tools(path, "FC(Cl)Br").hidden_building_block_ids == {1, 2, 3}
    assert _tools(path, "F[C@H](Cl)Br").hidden_building_block_ids == {1}
    assert _tools(path, "O=C1CCCCC1").hidden_building_block_ids == {5, 6}
    # Stock that fixes only part of the target's open stereo, also as a tautomer.
    assert _tools(path, "CC(F)C(Cl)CC").hidden_building_block_ids == {7}
    assert _tools(path, "O=C1CCC(F)C(Cl)C1").hidden_building_block_ids == {8}
    # A wildcard target has no InChI, so only an entry with its exact SMILES is hidden.
    assert _tools(path, "*CC").hidden_building_block_ids == frozenset()
