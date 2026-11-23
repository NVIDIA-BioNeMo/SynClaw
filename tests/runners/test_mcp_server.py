# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import os
import shutil
import signal
import sys
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

from synclaw.core import db
from synclaw.core.reactions import ingest_reaction_templates
from synclaw.runners import mcp_server
from synclaw.runners.mcp_server import (
    AsyncToolExecutor,
    ToolWorkerConfig,
    ToolWorkerPoolBroken,
    create_server,
)
from synclaw.tools.compaction import compact_tool_result
from synclaw.tools.contracts import TOOL_ARGUMENT_DEFAULTS, TOOL_SCHEMAS
from synclaw.tools.synthesis import SynthesisTools


class _FakeTools:
    """Stands in for ``SynthesisTools`` in thread mode; ``similarity`` runs ``behaviour``."""

    def __init__(self, behaviour) -> None:
        self.behaviour = behaviour

    def similarity(self, smiles_a: str, smiles_b: str) -> dict:
        return self.behaviour(smiles_a, smiles_b)


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

    def blocking_tool(value: str, _other: str) -> dict:
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
        return {"value": int(value)}

    async def run() -> list[dict]:
        executor = AsyncToolExecutor(16, use_process_pool=False, tools=_FakeTools(blocking_tool))
        tasks = [
            asyncio.create_task(executor.execute("similarity", smiles_a=str(index), smiles_b="C"))
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


def test_mcp_serves_the_compacted_result_rollouts_show_the_model() -> None:
    result = {
        "query": "OCC",
        "results": [
            {
                "id": 7,
                "smiles": "OCC",
                "canonical_smiles": "CCO",
                "match_type": "exact",
                "similarity": 1.0,
                "submit_reference": "BB:7",
                "submission_note": "Exact purchasable building-block match.",
            }
        ],
    }

    class SearchTools:
        def search_building_blocks(self, *_args: object, **_kwargs: object) -> dict:
            return result

    executor = AsyncToolExecutor(1, use_process_pool=False, tools=SearchTools())

    served = asyncio.run(executor.execute("search_building_blocks", query="OCC"))

    assert served == compact_tool_result("search_building_blocks", result)
    assert "submission_note" not in served["results"][0]


def test_mcp_tool_schemas_remain_public_and_wrappers_are_async(
    tmp_path: Path, bundled_db: Path
) -> None:
    database = Path(shutil.copy(bundled_db, tmp_path / "synclaw.sqlite"))
    db.prepare_building_block_search(database)
    server = create_server(database, use_process_pool=False)

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
    assert advertised == {
        tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOL_SCHEMAS
    }
    contracts = {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOL_SCHEMAS}
    for tool in tools.values():
        contract = contracts[tool.name]
        parameters = inspect.signature(tool.fn).parameters
        assert list(parameters) == list(contract["properties"]), tool.name
        required = set(contract.get("required") or ())
        for key, parameter in parameters.items():
            # Optional arguments default to "not given", so the contract's own default applies.
            expected = inspect.Parameter.empty if key in required else None
            assert parameter.default is expected, (tool.name, key)


def test_an_omitted_argument_reaches_the_tool_with_the_contract_default(
    tmp_path: Path, bundled_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Path(shutil.copy(bundled_db, tmp_path / "synclaw.sqlite"))
    db.prepare_building_block_search(database)
    seen: list[tuple] = []

    def search(_self, query, limit, *, include_near, min_similarity):
        seen.append((query, limit, include_near, min_similarity))
        return {"query": query, "results": []}

    monkeypatch.setattr(SynthesisTools, "search_building_blocks", search)
    server = create_server(database, use_process_pool=False)

    asyncio.run(server.call_tool("search_building_blocks", {"query": "CCO"}))

    defaults = TOOL_ARGUMENT_DEFAULTS["search_building_blocks"]
    assert seen == [
        ("CCO", defaults["limit"], defaults["include_near"], defaults["min_similarity"])
    ]


def test_thread_and_process_workers_dispatch_through_the_same_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[str, str]] = []

    def similarity(smiles_a: str, smiles_b: str) -> dict:
        called.append((smiles_a, smiles_b))
        return {}

    tools = _FakeTools(similarity)
    monkeypatch.setattr(mcp_server, "_WORKER_TOOLS", tools)
    executor = AsyncToolExecutor(1, use_process_pool=False, tools=tools)
    arguments = {"smiles_a": "", "smiles_b": "CCO"}

    in_thread = asyncio.run(executor.execute("similarity", **arguments))
    in_process = mcp_server._invoke_tool_process("similarity", arguments)

    assert in_process["error"] == "invalid_tool_call"
    assert in_thread == compact_tool_result("similarity", in_process)
    assert called == []


def test_server_refuses_a_library_without_family_metadata_at_startup(tmp_path: Path) -> None:
    database = tmp_path / "synclaw.sqlite"
    templates = tmp_path / "templates.txt"
    templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
    ingest_reaction_templates(templates, db_path=database, library="custom")
    db.prepare_building_block_search(database)

    with pytest.raises(ValueError, match="missing family_metadata"):
        create_server(database, use_process_pool=False)


def test_identical_inflight_calls_reattach_without_duplicate_execution() -> None:
    calls = 0
    started = threading.Event()
    release = threading.Event()

    def blocking_tool(value: str, _other: str) -> dict:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=5)
        return {"value": int(value)}

    async def run() -> tuple[dict, dict]:
        executor = AsyncToolExecutor(4, use_process_pool=False, tools=_FakeTools(blocking_tool))
        first = asyncio.create_task(executor.execute("similarity", smiles_a="7", smiles_b="C"))
        assert await asyncio.to_thread(started.wait, 5)
        second = asyncio.create_task(executor.execute("similarity", smiles_a="7", smiles_b="C"))
        await asyncio.sleep(0)
        release.set()
        return await first, await second

    first, second = asyncio.run(run())

    assert calls == 1
    assert first["value"] == second["value"] == 7


def test_a_finished_success_is_reused_but_a_failure_is_retried() -> None:
    outcomes = [RuntimeError("database is locked"), {"value": 1}]
    calls = 0

    def flaky(_a: str, _b: str) -> dict:
        nonlocal calls
        calls += 1
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def run() -> tuple[dict, dict]:
        executor = AsyncToolExecutor(1, use_process_pool=False, tools=_FakeTools(flaky))
        with pytest.raises(RuntimeError, match="database is locked"):
            await executor.execute("similarity", smiles_a="C", smiles_b="C")
        retried = await executor.execute("similarity", smiles_a="C", smiles_b="C")
        reused = await executor.execute("similarity", smiles_a="C", smiles_b="C")
        return retried, reused

    retried, reused = asyncio.run(run())

    assert retried == reused == {"value": 1}
    assert calls == 2  # the failure ran again; the success was served from the first run


def test_a_pool_that_breaks_again_right_after_a_restart_is_not_restarted_forever(
    tmp_path: Path,
) -> None:
    executor = AsyncToolExecutor(
        1,
        use_process_pool=False,
        tools=_FakeTools(lambda *_: {}),
        tool_worker_config=ToolWorkerConfig(db_path=str(tmp_path / "unused.sqlite")),
    )
    try:
        first = executor._replace_broken_pool(executor._worker_pool, BrokenProcessPool("died"))
        second = executor._replace_broken_pool(executor._worker_pool, BrokenProcessPool("again"))

        assert isinstance(first, ToolWorkerPoolBroken)
        assert not isinstance(second, ToolWorkerPoolBroken)
        assert "failed again right after a restart" in str(second)
        with pytest.raises(RuntimeError, match="failed again right after a restart"):
            asyncio.run(executor.execute("similarity", smiles_a="C", smiles_b="C"))
    finally:
        executor.shutdown()


def test_a_broken_pool_during_shutdown_is_not_reported_as_retryable() -> None:
    executor = AsyncToolExecutor(1, use_process_pool=False, tools=_FakeTools(lambda *_: {}))
    pool = executor._worker_pool
    executor.shutdown()

    error = executor._replace_broken_pool(pool, BrokenProcessPool("stopping"))

    assert not isinstance(error, ToolWorkerPoolBroken)
    assert "shutting down" in str(error)


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
            smiles_a="CCO",
            smiles_b="CCO",
        )
        assert result["similarity"] == 1.0

        marker_path = tmp_path / "active-worker-pid"
        blocking_future = executor._worker_pool.submit(_block_process_worker, str(marker_path))
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
                smiles_a="CCO",
                smiles_b=smiles_b,
            )
            assert 0.0 <= result["similarity"] < 1.0

    try:
        asyncio.run(run())
    finally:
        executor.shutdown()
