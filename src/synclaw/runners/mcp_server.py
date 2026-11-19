# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server wrapper for the synthesis tools."""

from __future__ import annotations

import asyncio
import atexit
import json
import multiprocessing
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synclaw.core import db
from synclaw.core.libraries import LibrarySelection, normalize_libraries
from synclaw.core.toxicity import ToxicityConfig
from synclaw.rollout.loop import call_local_tool
from synclaw.rollout.policy import DEFAULT_REACTION_LIBRARY
from synclaw.tools.contracts import TOOL_NAMES, TOOL_SCHEMAS
from synclaw.tools.synthesis import SynthesisTools

INFLIGHT_RESULT_RETENTION_SECONDS = 30.0
MAX_RETAINED_TOOL_CALLS = 2048
PROCESS_POOL_STARTUP_TIMEOUT_SECONDS = 120.0

_TOOL_METHOD_NAMES = frozenset(TOOL_NAMES)
_WORKER_TOOLS: SynthesisTools | None = None


class ToolWorkerPoolBroken(RuntimeError):
    """A chemistry worker died during this call; the pool was rebuilt, so a retry can succeed."""


@dataclass(frozen=True, slots=True)
class ToolWorkerConfig:
    db_path: str
    reaction_libraries: tuple[str, ...] = ()
    toxicity: ToxicityConfig = field(default_factory=ToxicityConfig.resolve)


def _initialize_tool_worker(
    config: ToolWorkerConfig,
    ready_queue: Any | None,
) -> None:
    """Build worker-local chemistry state and report startup before serving calls."""

    global _WORKER_TOOLS
    try:
        _WORKER_TOOLS = SynthesisTools(
            config.db_path,
            reaction_libraries=config.reaction_libraries,
            toxicity=config.toxicity,
            _initialize_database=False,
        )
        _WORKER_TOOLS.screen  # Load the watchlist now so a missing one fails at startup.
    except BaseException as exc:
        if ready_queue is not None:
            ready_queue.put((os.getpid(), type(exc).__name__, str(exc)))
        raise
    if ready_queue is not None:
        ready_queue.put((os.getpid(), "", ""))


def _invoke_tool_process(
    tool_name: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Execute one allowlisted tool against worker-local state."""

    if tool_name not in _TOOL_METHOD_NAMES:
        raise ValueError(f"Unknown BioNeMo-SynClaw tool {tool_name!r}")
    if _WORKER_TOOLS is None:
        raise RuntimeError("BioNeMo-SynClaw tool worker was not initialized")

    return call_local_tool(
        _WORKER_TOOLS,
        tool_name,
        kwargs,
        reaction_library=kwargs.get("library", DEFAULT_REACTION_LIBRARY),
        target_smiles=kwargs.get("target_smiles"),
    )


def _tool_worker_pid() -> int:
    """Return a worker identity after its initializer has completed."""

    if _WORKER_TOOLS is None:
        raise RuntimeError("BioNeMo-SynClaw tool worker was not initialized")
    return os.getpid()


class AsyncToolExecutor:
    """Run synchronous chemistry tools off-loop with bounded, coalesced concurrency."""

    def __init__(
        self,
        max_concurrent_executions: int,
        *,
        use_process_pool: bool = True,
        tool_worker_config: ToolWorkerConfig | None = None,
        process_start_method: str = "fork",
    ) -> None:
        if max_concurrent_executions <= 0:
            raise ValueError("max_concurrent_executions must be greater than zero")
        self._semaphore = asyncio.Semaphore(max_concurrent_executions)
        self._uses_process_pool = use_process_pool and sys.platform.startswith("linux")
        self._worker_pool: ProcessPoolExecutor | ThreadPoolExecutor
        self._worker_pids: frozenset[int] = frozenset()
        self._max_workers = max_concurrent_executions
        self._tool_worker_config = tool_worker_config
        self._shutdown = False
        self._pool_lock = threading.Lock()
        if self._uses_process_pool:
            if tool_worker_config is None:
                raise ValueError("tool_worker_config is required for process workers")
            ready_queue = multiprocessing.get_context(process_start_method).Queue()
            self._worker_pool = self._new_process_pool(process_start_method, ready_queue)
            self._worker_pids = self._prestart_process_pool(
                ready_queue,
                expected_workers=max_concurrent_executions,
            )
        else:
            self._worker_pool = ThreadPoolExecutor(
                max_workers=max_concurrent_executions,
                thread_name_prefix="synclaw-mcp-tool",
            )
        self._inflight_lock = asyncio.Lock()
        self._inflight: OrderedDict[str, asyncio.Task[dict[str, Any]]] = OrderedDict()
        atexit.register(self.shutdown)

    def _new_process_pool(
        self,
        start_method: str,
        ready_queue: Any | None = None,
    ) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context(start_method),
            initializer=_initialize_tool_worker,
            initargs=(self._tool_worker_config, ready_queue),
        )

    def _prestart_process_pool(
        self,
        ready_queue: Any,
        *,
        expected_workers: int,
    ) -> frozenset[int]:
        """Start every process worker before the host creates additional threads."""

        assert isinstance(self._worker_pool, ProcessPoolExecutor)
        probes = [self._worker_pool.submit(_tool_worker_pid) for _ in range(expected_workers)]
        deadline = time.monotonic() + PROCESS_POOL_STARTUP_TIMEOUT_SECONDS
        worker_pids: set[int] = set()
        try:
            while len(worker_pids) < expected_workers:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "Timed out while prestarting BioNeMo-SynClaw chemistry workers "
                        f"({len(worker_pids)}/{expected_workers} ready)"
                    )
                try:
                    pid, error_type, error_message = ready_queue.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError(
                        "Timed out while prestarting BioNeMo-SynClaw chemistry workers "
                        f"({len(worker_pids)}/{expected_workers} ready)"
                    ) from exc
                if error_type:
                    raise RuntimeError(
                        f"BioNeMo-SynClaw chemistry worker {pid} failed startup: "
                        f"{error_type}: {error_message}"
                    )
                worker_pids.add(int(pid))
            for probe in probes:
                probe.result(timeout=max(0.0, deadline - time.monotonic()))
        except BaseException:
            self._worker_pool.shutdown(wait=True, cancel_futures=True)
            raise
        finally:
            ready_queue.close()
            ready_queue.join_thread()
        return frozenset(worker_pids)

    @property
    def worker_pids(self) -> frozenset[int]:
        """Return process worker identities for startup diagnostics and tests."""

        return self._worker_pids

    def shutdown(self) -> None:
        """Release tool workers; safe to call more than once."""

        with self._pool_lock:
            if self._shutdown:
                return
            self._shutdown = True
            pool = self._worker_pool
        pool.shutdown(wait=True, cancel_futures=True)

    def _replace_broken_pool(
        self, broken: ProcessPoolExecutor, exc: BrokenProcessPool
    ) -> ToolWorkerPoolBroken:
        """Swap in a fresh pool once, however many in-flight calls saw the same breakage."""

        with self._pool_lock:
            replace = not self._shutdown and self._worker_pool is broken
            if replace:
                # The server now runs other threads, so fork could copy a held lock into the
                # child; spawn starts each replacement worker from a clean interpreter.
                self._worker_pool = self._new_process_pool("spawn")
        if replace:
            broken.shutdown(wait=False, cancel_futures=True)
            print(f"synclaw_mcp_worker_pool_restarted ({exc})", file=sys.stderr, flush=True)
        return ToolWorkerPoolBroken(
            "A chemistry worker terminated unexpectedly during this call; "
            f"the worker pool was restarted, so retry the call. ({exc})"
        )

    async def execute(
        self,
        tool_name: str,
        function: Callable[..., dict[str, Any]] | None,
        /,
        **kwargs: Any,
    ) -> dict[str, Any]:
        call_key = json.dumps(
            {"tool_name": tool_name, "arguments": kwargs},
            separators=(",", ":"),
            sort_keys=True,
        )
        async with self._inflight_lock:
            task = self._inflight.get(call_key)
            if task is None:
                task = asyncio.create_task(self._execute_new(tool_name, function, kwargs))
                self._inflight[call_key] = task
                task.add_done_callback(
                    lambda completed, key=call_key: self._schedule_expiry(key, completed)
                )
                self._evict_completed_overflow()
            else:
                self._inflight.move_to_end(call_key)
        return await asyncio.shield(task)

    def _schedule_expiry(self, call_key: str, task: asyncio.Task[dict[str, Any]]) -> None:
        if not task.cancelled() and isinstance(task.exception(), ToolWorkerPoolBroken):
            # Drop the failure before any waiter resumes, so a retry reaches the rebuilt pool.
            if self._inflight.get(call_key) is task:
                self._inflight.pop(call_key, None)
            return
        asyncio.get_running_loop().call_later(
            INFLIGHT_RESULT_RETENTION_SECONDS,
            lambda: asyncio.create_task(self._expire_inflight(call_key, task)),
        )

    async def _expire_inflight(
        self,
        call_key: str,
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        async with self._inflight_lock:
            if self._inflight.get(call_key) is task:
                self._inflight.pop(call_key, None)

    def _evict_completed_overflow(self) -> None:
        while len(self._inflight) > MAX_RETAINED_TOOL_CALLS:
            completed_key = next(
                (key for key, task in self._inflight.items() if task.done()),
                None,
            )
            if completed_key is None:
                break
            self._inflight.pop(completed_key, None)

    async def _execute_new(
        self,
        tool_name: str,
        function: Callable[..., dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            pool = self._worker_pool
            try:
                if self._uses_process_pool:
                    return await loop.run_in_executor(pool, _invoke_tool_process, tool_name, kwargs)
                if function is None:
                    raise ValueError("function is required when process workers are disabled")
                return await loop.run_in_executor(pool, lambda: function(**kwargs))
            except BrokenProcessPool as exc:
                assert isinstance(pool, ProcessPoolExecutor)
                raise self._replace_broken_pool(pool, exc) from exc


def create_server(
    db_path: str | Path = db.DEFAULT_DB_PATH,
    *,
    reaction_library: LibrarySelection = DEFAULT_REACTION_LIBRARY,
    max_concurrent_executions: int = 64,
    use_process_pool: bool = True,
    toxicity: ToxicityConfig | None = None,
):
    toxicity = toxicity or ToxicityConfig.resolve()
    with db.session(db_path) as conn:
        db.require_reaction_library(conn, reaction_library)
        # Fail at startup rather than at the model's first query-less template search.
        db.reaction_template_library_overview(conn, library=reaction_library)
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise RuntimeError(
            "The MCP SDK is not installed. Install this package with the 'mcp' "
            "extra, for example: pip install -e '.[mcp]'."
        ) from exc

    db.require_prepared_building_block_search(db_path)
    # Forked tool workers inherit the loaded index and share its memory.
    with db.session(db_path) as conn:
        db.preload_building_block_search(conn)
    tools = SynthesisTools(db_path, reaction_libraries=reaction_library, toxicity=toxicity)
    tools.screen  # Fail at startup, not on the first submission, if the watchlist is missing.
    executor = AsyncToolExecutor(
        max_concurrent_executions,
        use_process_pool=use_process_pool,
        tool_worker_config=ToolWorkerConfig(
            db_path=str(Path(db_path).resolve()),
            reaction_libraries=normalize_libraries(reaction_library),
            toxicity=toxicity,
        ),
    )

    class SynClawMCP(MCPServer):
        async def list_tools(self):
            # MCPServer derives schemas from the signatures; advertise the canonical ones instead.
            return [
                tool.model_copy(update={"input_schema": deepcopy(parameters[tool.name])})
                for tool in await super().list_tools()
            ]

    server = SynClawMCP("bionemo-synclaw-server")
    descriptions = {
        tool["function"]["name"]: tool["function"]["description"] for tool in TOOL_SCHEMAS
    }
    parameters = {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOL_SCHEMAS}

    @server.tool(description=descriptions["search_building_blocks"])
    async def search_building_blocks(
        query: str,
        limit: int = 10,
        include_near: bool = True,
        min_similarity: float = 0.15,
    ) -> dict:
        return await executor.execute(
            "search_building_blocks",
            tools.search_building_blocks,
            query=query,
            limit=limit,
            include_near=include_near,
            min_similarity=min_similarity,
        )

    @server.tool(description=descriptions["similarity"])
    async def similarity(smiles_a: str, smiles_b: str) -> dict:
        return await executor.execute(
            "similarity", tools.similarity, smiles_a=smiles_a, smiles_b=smiles_b
        )

    @server.tool(description=descriptions["mces"])
    async def mces(smiles_a: str, smiles_b: str) -> dict:
        return await executor.execute("mces", tools.mces, smiles_a=smiles_a, smiles_b=smiles_b)

    @server.tool(description=descriptions["search_reaction_templates"])
    async def search_reaction_templates(
        query: str | None = None,
        reactant_smiles: list[str] | None = None,
        product_smiles: str | list[str] | None = None,
        limit: int = 10,
        run_forward: bool = False,
    ) -> dict:
        return await executor.execute(
            "search_reaction_templates",
            tools.search_reaction_templates,
            query=query,
            reactant_smiles=reactant_smiles,
            product_smiles=product_smiles,
            limit=limit,
            library=reaction_library or None,
            run_forward=run_forward,
        )

    @server.tool(description=descriptions["simulate_reaction"])
    async def simulate_reaction(
        reaction_id: str,
        reactants: list[str],
        product_smiles: str | None = None,
        max_products: int = 100,
    ) -> dict:
        return await executor.execute(
            "simulate_reaction",
            tools.simulate_reaction,
            reaction_id=reaction_id,
            reactants=reactants,
            product_smiles=product_smiles,
            max_products=max_products,
        )

    @server.tool(description=descriptions["simulate_reverse_reaction"])
    async def simulate_reverse_reaction(
        reaction_id: str,
        product_smiles: str | list[str],
        max_outcomes: int = 50,
        include_near: bool = True,
        min_similarity: float = 0.15,
        top_near_matches: int = 3,
    ) -> dict:
        return await executor.execute(
            "simulate_reverse_reaction",
            tools.simulate_reverse_reaction,
            reaction_id=reaction_id,
            product_smiles=product_smiles,
            max_outcomes=max_outcomes,
            include_near=include_near,
            min_similarity=min_similarity,
            top_near_matches=top_near_matches,
        )

    @server.tool(description=descriptions["submit_pathway"])
    async def submit_pathway(
        steps: list[dict] | None = None,
        building_block: str | None = None,
        pathway_text: str | None = None,
    ) -> dict:
        return await executor.execute(
            "submit_pathway",
            tools.submit_pathway,
            steps=steps or [],
            building_block=building_block,
            pathway_text=pathway_text,
        )

    return server


def serve(
    db_path: str | Path = db.DEFAULT_DB_PATH,
    *,
    reaction_library: LibrarySelection = DEFAULT_REACTION_LIBRARY,
    max_concurrent_executions: int = 64,
    toxicity: ToxicityConfig | None = None,
) -> None:
    create_server(
        db_path,
        reaction_library=reaction_library,
        max_concurrent_executions=max_concurrent_executions,
        toxicity=toxicity,
    ).run()
