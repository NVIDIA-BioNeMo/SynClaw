# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import os
import signal
import sys
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

from synclaw.cli import build_parser
from synclaw.core import db
from synclaw.core.reactions import ingest_reaction_templates
from synclaw.rollout.loop import call_local_tool
from synclaw.runners import mcp_server
from synclaw.runners.mcp_server import (
    AsyncToolExecutor,
    ToolWorkerConfig,
    ToolWorkerPoolBroken,
    create_server,
)
from synclaw.tools.contracts import TOOL_ARGUMENT_DEFAULTS, TOOL_SCHEMAS


def _block_process_worker(marker_path: str) -> None:
    Path(marker_path).write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(30)


def test_async_tool_executor_caps_concurrency_and_releases_waiters() -> None:
    active = 0
    peak_active = 0
    started = 0
    state_lock = threading.Lock()
    sixteen_started = threading.Event()
    release = threading.Event()

    def blocking_tool(*, value: int) -> dict:
        nonlocal active, peak_active, started
        with state_lock:
            active += 1
            started += 1
            peak_active = max(peak_active, active)
            if started == 16:
                sixteen_started.set()
        assert release.wait(timeout=5)
        with state_lock:
            active -= 1
        return {"value": value}

    async def run() -> list[dict]:
        executor = AsyncToolExecutor(16, use_process_pool=False)
        tasks = [
            asyncio.create_task(
                executor.execute("blocking_tool", blocking_tool, value=index)
            )
            for index in range(20)
        ]
        assert await asyncio.to_thread(sixteen_started.wait, 5)
        await asyncio.sleep(0)
        with state_lock:
            assert active == 16
            assert started == 16
        release.set()
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())

    assert peak_active == 16
    assert sorted(result["value"] for result in results) == list(range(20))


def test_mcp_tool_schemas_remain_public_and_wrappers_are_async(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(["mcp", "serve"])
    assert args.reaction_library == ("synformer", "paroutes", "ord", "nvidia")
    assert inspect.signature(create_server).parameters[
        "reaction_library"
    ].default == ("synformer", "paroutes", "ord", "nvidia")

    db.init_db(tmp_path / "synclaw.sqlite")
    db.prepare_building_block_search(tmp_path / "synclaw.sqlite")
    server = create_server(
        tmp_path / "synclaw.sqlite", reaction_library="", use_process_pool=False,
    )

    tools = server._tool_manager._tools
    assert set(tools) == {
        "search_building_blocks",
        "similarity",
        "mces",
        "search_reaction_templates",
        "simulate_reaction",
        "simulate_reverse_reaction",
        "submit_pathway",
    }
    assert all(inspect.iscoroutinefunction(tool.fn) for tool in tools.values())
    advertised = {tool.name: tool.input_schema for tool in asyncio.run(server.list_tools())}
    assert advertised == {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOL_SCHEMAS}
    for tool in tools.values():
        properties = tool.parameters["properties"]
        assert "max_concurrent_executions" not in properties
        parameters = inspect.signature(tool.fn).parameters
        for key, value in TOOL_ARGUMENT_DEFAULTS[tool.name].items():
            if key != "steps":
                assert parameters[key].default == value, (tool.name, key)


def test_server_refuses_a_library_without_family_metadata_at_startup(tmp_path: Path) -> None:
    database = tmp_path / "synclaw.sqlite"
    templates = tmp_path / "templates.txt"
    templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
    ingest_reaction_templates(templates, db_path=database, library="custom")
    db.prepare_building_block_search(database)

    with pytest.raises(ValueError, match="missing family_metadata"):
        create_server(database, reaction_library="custom", use_process_pool=False)


def test_process_worker_rejects_schema_invalid_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GuardedTools:
        called = False

        def simulate_reaction(self, **_kwargs: object) -> dict:
            self.called = True
            return {"valid": True}

    tools = GuardedTools()
    monkeypatch.setattr(mcp_server, "_WORKER_TOOLS", tools)
    arguments = {"reaction_id": "R1", "reactants": "CC"}

    result = mcp_server._invoke_tool_process(
        "simulate_reaction",
        arguments,
    )

    assert result == call_local_tool(tools, "simulate_reaction", arguments)
    assert result["error"] == "invalid_tool_call"
    assert result["retryable"] is True
    assert "must be a JSON array" in result["issues"][0]
    assert tools.called is False


def test_identical_inflight_calls_reattach_without_duplicate_execution() -> None:
    calls = 0
    started = threading.Event()
    release = threading.Event()

    def blocking_tool(*, value: int) -> dict:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=5)
        return {"value": value}

    async def run() -> tuple[dict, dict]:
        executor = AsyncToolExecutor(4, use_process_pool=False)
        first = asyncio.create_task(
            executor.execute("blocking_tool", blocking_tool, value=7)
        )
        assert await asyncio.to_thread(started.wait, 5)
        second = asyncio.create_task(
            executor.execute("blocking_tool", blocking_tool, value=7)
        )
        await asyncio.sleep(0)
        release.set()
        return await first, await second

    first, second = asyncio.run(run())

    assert calls == 1
    assert first["value"] == second["value"] == 7


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process workers are Linux-only")
@pytest.mark.parametrize("process_start_method", ["fork", "spawn"])
def test_process_workers_prestart_and_rebuild_a_broken_pool(
    tmp_path: Path,
    process_start_method: str,
) -> None:
    database_path = tmp_path / "synclaw.sqlite"
    db.init_db(database_path)
    db.prepare_building_block_search(database_path)
    executor = AsyncToolExecutor(
        2,
        tool_worker_config=ToolWorkerConfig(db_path=str(database_path)),
        process_start_method=process_start_method,
    )

    async def run() -> None:
        assert len(executor.worker_pids) == 2
        result = await executor.execute(
            "similarity",
            lambda **_: {},
            smiles_a="CCO",
            smiles_b="CCO",
        )
        assert result["similarity"] == 1.0

        marker_path = tmp_path / "active-worker-pid"
        blocking_future = executor._worker_pool.submit(
            _block_process_worker, str(marker_path)
        )
        for _ in range(100):
            if marker_path.exists():
                break
            await asyncio.sleep(0.05)
        assert marker_path.exists()
        os.kill(int(marker_path.read_text(encoding="utf-8")), signal.SIGKILL)
        with pytest.raises(BrokenProcessPool):
            await asyncio.wrap_future(blocking_future)
        fatal: ToolWorkerPoolBroken | None = None
        for _ in range(50):
            await asyncio.sleep(0.05)
            try:
                await executor.execute(
                    "similarity",
                    lambda **_: {},
                    smiles_a="CCO",
                    smiles_b="CCN",
                )
            except ToolWorkerPoolBroken as exc:
                fatal = exc
                break
        assert fatal is not None
        assert "retry the call" in str(fatal)

        # The pool was rebuilt: retrying the failed call and making a new one both succeed.
        for smiles_b in ("CCN", "CCC"):
            result = await executor.execute(
                "similarity",
                lambda **_: {},
                smiles_a="CCO",
                smiles_b=smiles_b,
            )
            assert 0.0 <= result["similarity"] < 1.0

    try:
        asyncio.run(run())
    finally:
        executor.shutdown()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork workers are Linux-only")
def test_process_worker_tool_exception_does_not_break_pool(tmp_path: Path) -> None:
    database_path = tmp_path / "synclaw.sqlite"
    db.init_db(database_path)
    db.prepare_building_block_search(database_path)
    executor = AsyncToolExecutor(
        2,
        tool_worker_config=ToolWorkerConfig(db_path=str(database_path)),
    )

    async def run() -> None:
        with pytest.raises(ValueError, match="Unknown SynClaw tool"):
            await executor.execute("not-a-public-tool", lambda **_: {})
        result = await executor.execute(
            "similarity",
            lambda **_: {},
            smiles_a="CCO",
            smiles_b="CCO",
        )
        assert result["similarity"] == 1.0

    try:
        asyncio.run(run())
    finally:
        executor.shutdown()
