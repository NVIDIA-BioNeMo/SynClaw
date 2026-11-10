# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

from synclaw.core import db, reactions
from synclaw.paths import REPO_ROOT
from synclaw.tasks.datasets import build_llm_task


def load_script():
    script = REPO_ROOT / "scripts" / "render_model_context.py"
    spec = importlib.util.spec_from_file_location("render_model_context", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_renders_the_run_prompt_with_task_and_run_exclusions(tmp_path: Path) -> None:
    templates = tmp_path / "templates.jsonl"
    templates.write_text(json.dumps({
        "template_id": "R000", "smarts": "[C:1][O:2]>>[C:1]=[O:2]", "family": "oxidation",
        "family_metadata": {"description": "Alcohol to carbonyl.", "search_terms": ["alcohol oxidation"]},
    }) + "\n", encoding="utf-8")
    bbs = tmp_path / "bbs.txt"
    bbs.write_text("SMILES\nCO\n", encoding="utf-8")
    db_path = tmp_path / "test.sqlite"
    reactions.ingest_reaction_templates(templates, db_path=db_path, library="demo")
    db.ingest_building_blocks(bbs, db_path)
    db.prepare_building_block_search(db_path)
    taskset = tmp_path / "tasks.jsonl"
    task = build_llm_task("CCO", task_id="fixture-0001", dataset="fixture", source_file="fixture",
                          excluded_tools=["search_reaction_templates"], budgets={"max_turns": 40})
    taskset.write_text(json.dumps(task) + "\n", encoding="utf-8")

    markdown, _ = load_script().render_model_context(
        taskset=taskset, task_index=0, db_path=db_path, reaction_library="demo",
        max_tool_calls=10, api_mode="chat", exclude_tools=["mces"],
    )

    assert "Excluded tools: `mces, search_reaction_templates`" in markdown
    assert "Use no more than 10 total tool calls or 40 assistant turns." in markdown
    assert '"name": "search_reaction_templates"' not in markdown
