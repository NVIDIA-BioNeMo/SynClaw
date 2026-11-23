# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server wrapper for the synthesis tools."""

from __future__ import annotations

import asyncio
import atexit
import functools
import inspect
import json
import multiprocessing
import operator
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from synclaw.core import db
from synclaw.core.toxicity import ToxicityConfig
from synclaw.rollout.loop import call_local_tool
from synclaw.tools.compaction import compact_tool_result
from synclaw.tools.contracts import TOOL_ARGUMENT_DEFAULTS, TOOL_SCHEMAS
from synclaw.tools.synthesis import SynthesisTools

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer
    from mcp_types import Tool

INFLIGHT_RESULT_RETENTION_SECONDS = 30.0
MAX_RETAINED_TOOL_CALLS = 2048
PROCESS_POOL_STARTUP_TIMEOUT_SECONDS = 120.0

_WORKER_TOOLS: SynthesisTools | None = None
# The Python type MCP validates each JSON Schema type in a tool contract against.
_JSON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
}


class ToolWorkerPoolBroken(RuntimeError):
    """A chemistry worker died during this call; the pool was rebuilt, so a retry can succeed."""


@dataclass(frozen=True, slots=True)
class ToolWorkerConfig:
    """What each process worker needs to build its own ``SynthesisTools``."""

    db_path: str
    toxicity: ToxicityConfig = field(default_factory=ToxicityConfig.resolve)


def _initialize_tool_worker(
    config: ToolWorkerConfig,
    ready_queue: Any | None,
) -> None:
    """Build worker-local chemistry state and report startup before serving calls."""

    global _WORKER_TOOLS
    try:
        _WORKER_TOOLS = SynthesisTools(config.db_path, toxicity=config.toxicity)
        _WORKER_TOOLS.screen  # Load the watchlist now so a missing one fails at startup.
    except BaseException as exc:
        if ready_queue is not None:
            ready_queue.put((os.getpid(), type(exc).__name__, str(exc)))
        raise
    if ready_queue is not None:
        ready_queue.put((os.getpid(), "", ""))


def _invoke_tool_process(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one call against worker-local state, exactly as a rollout does."""

    if _WORKER_TOOLS is None:
        raise RuntimeError("BioNeMo-SynClaw tool worker was not initialized")
    return call_local_tool(_WORKER_TOOLS, tool_name, kwargs)


def _tool_worker_pid() -> int:
    """Return a worker identity after its initializer has completed."""

    if _WORKER_TOOLS is None:
        raise RuntimeError("BioNeMo-SynClaw tool worker was not initialized")
    return os.getpid()


class AsyncToolExecutor:
    """Run synchronous chemistry tools off-loop with bounded, coalesced concurrency.

    Calls go to a pool of process workers on Linux and to a thread pool elsewhere, or when
    ``use_process_pool`` is False. Either way each call is dispatched by ``call_local_tool``,
    so arguments are defaulted and checked exactly as in a rollout.
    """

    def __init__(
        self,
        max_concurrent_executions: int,
        *,
        use_process_pool: bool = True,
        tool_worker_config: ToolWorkerConfig | None = None,
        tools: Any | None = None,
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
        self._tools = tools
        self._shutdown = False
        # Replacement workers report startup failures here, as the first pool's do at prestart.
        self._startup_reports: Any | None = None
        # Set once replacement workers fail at startup; restarting again would only loop.
        self._unusable: str | None = None
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
            if tools is None:
                raise ValueError("tools is required for thread workers")
            self._worker_pool = ThreadPoolExecutor(
                max_workers=max_concurrent_executions,
                thread_name_prefix="synclaw-mcp-tool",
            )
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
                try:
                    pid, error_type, error_message = ready_queue.get(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
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

    def _replace_broken_pool(self, broken: Any, exc: BrokenProcessPool) -> RuntimeError:
        """Swap in a fresh pool once, however many in-flight calls saw the same breakage.

        Returns the error for the caller: a retryable ``ToolWorkerPoolBroken`` after a
        restart, or a final error when the server is shutting down or replacement workers
        reported that they fail at startup. A worker crashed by a call is restarted however
        often that recurs, so one bad input cannot take the server down for every client.
        """

        with self._pool_lock:
            if self._shutdown:
                return RuntimeError(
                    f"The MCP server is shutting down; the call did not run. ({exc})"
                )
            if self._worker_pool is broken and self._unusable is None:
                if startup_error := self._replacement_startup_error():
                    self._unusable = (
                        f"BioNeMo-SynClaw chemistry workers fail at startup ({startup_error}); "
                        "fix the cause and restart the server."
                    )
                else:
                    # The server now runs other threads, so fork could copy a held lock into
                    # the child; spawn starts each replacement worker from a clean interpreter.
                    reports = multiprocessing.get_context("spawn").Queue()
                    self._worker_pool = self._new_process_pool("spawn", reports)
                    self._startup_reports = reports
                    broken.shutdown(wait=False, cancel_futures=True)
                    print(f"synclaw_mcp_worker_pool_restarted ({exc})", file=sys.stderr, flush=True)
            if self._unusable is not None:
                return RuntimeError(self._unusable)
        return ToolWorkerPoolBroken(
            "A chemistry worker terminated unexpectedly during this call; "
            f"the worker pool was restarted, so retry the call. ({exc})"
        )

    def _replacement_startup_error(self) -> str | None:
        """The first startup failure a replacement worker reported, if any."""

        error = None
        while self._startup_reports is not None:
            try:
                pid, error_type, error_message = self._startup_reports.get_nowait()
            except queue.Empty:
                break
            if error_type and error is None:
                error = f"worker {pid}: {error_type}: {error_message}"
        return error

    async def execute(self, tool_name: str, /, **kwargs: Any) -> dict[str, Any]:
        """Run one tool call and return the compacted result rollouts show the model.

        An identical call that is still running, or that succeeded within the last
        ``INFLIGHT_RESULT_RETENTION_SECONDS``, shares that result instead of running again.
        """

        call_key = json.dumps(
            {"tool_name": tool_name, "arguments": kwargs},
            separators=(",", ":"),
            sort_keys=True,
        )
        task = self._inflight.get(call_key)
        if task is None:
            task = asyncio.create_task(self._execute_new(tool_name, kwargs))
            self._inflight[call_key] = task
            task.add_done_callback(lambda done, key=call_key: self._retain_or_drop(key, done))
            self._evict_completed_overflow()
        else:
            self._inflight.move_to_end(call_key)
        # Clients pass results straight to their model, so serve the view rollouts give it.
        return compact_tool_result(tool_name, await asyncio.shield(task))

    def _retain_or_drop(self, call_key: str, task: asyncio.Task[dict[str, Any]]) -> None:
        """Keep a success for identical retries a while; drop a failure so a retry runs again."""

        if task.cancelled() or task.exception() is not None:
            # Dropped before any waiter resumes, so an immediate retry starts a fresh call.
            self._drop_inflight(call_key, task)
            return
        asyncio.get_running_loop().call_later(
            INFLIGHT_RESULT_RETENTION_SECONDS, self._drop_inflight, call_key, task
        )

    def _drop_inflight(self, call_key: str, task: asyncio.Task[dict[str, Any]]) -> None:
        if self._inflight.get(call_key) is task:
            del self._inflight[call_key]

    def _evict_completed_overflow(self) -> None:
        # Only finished calls are evicted; calls still waiting for a worker stay attachable.
        while len(self._inflight) > MAX_RETAINED_TOOL_CALLS:
            completed_key = next(
                (key for key, task in self._inflight.items() if task.done()),
                None,
            )
            if completed_key is None:
                break
            self._inflight.pop(completed_key, None)

    async def _execute_new(self, tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        async with self._semaphore:
            if self._unusable is not None:
                raise RuntimeError(self._unusable)
            loop = asyncio.get_running_loop()
            pool = self._worker_pool
            if not self._uses_process_pool:
                return await loop.run_in_executor(
                    pool, call_local_tool, self._tools, tool_name, kwargs
                )
            try:
                return await loop.run_in_executor(pool, _invoke_tool_process, tool_name, kwargs)
            except BrokenProcessPool as exc:
                raise self._replace_broken_pool(pool, exc) from exc


def _annotation(schema: dict[str, Any]) -> Any:
    """The Python type MCP validates a tool-contract property against."""

    if "anyOf" in schema:
        return functools.reduce(operator.or_, (_annotation(option) for option in schema["anyOf"]))
    if schema["type"] == "array":
        return list[_annotation(schema["items"])]
    return _JSON_TYPES[schema["type"]]


def _register_tool(
    server: MCPServer, executor: AsyncToolExecutor, function: dict[str, Any]
) -> None:
    """Register one contract tool, taking its arguments and defaults from the contract alone."""

    name = function["name"]
    parameters = function["parameters"]
    required = set(parameters.get("required") or ())

    async def call(**arguments: Any) -> dict:
        # An omitted optional argument arrives as None; the contract default takes its place, so a
        # call that spells out a default shares its result with one that leaves it out.
        given = {key: value for key, value in arguments.items() if value is not None}
        return await executor.execute(name, **{**TOOL_ARGUMENT_DEFAULTS.get(name, {}), **given})

    call.__name__ = call.__qualname__ = name
    call.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY, annotation=_annotation(schema))
            if key in required
            else inspect.Parameter(
                key,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=_annotation(schema) | None,
                default=None,
            )
            for key, schema in parameters["properties"].items()
        ],
        return_annotation=dict,
    )
    server.tool(name=name, description=function["description"])(call)


def create_server(
    db_path: str | Path,
    *,
    max_concurrent_executions: int = 64,
    use_process_pool: bool = True,
    toxicity: ToxicityConfig | None = None,
) -> MCPServer:
    """Build the MCP server for the synthesis tools.

    Checks the database and the toxicity watchlist, loads the search index and starts the tool
    workers, so a misconfigured server fails at startup rather than on its first call. Each
    tool's arguments, defaults and advertised schema come from ``tools/contracts.py``.

    Raises:
        RuntimeError: The MCP SDK is missing, or the database or watchlist is not usable.
    """

    toxicity = toxicity or ToxicityConfig.resolve()
    with db.session(db_path) as conn:
        db.require_reaction_templates(conn)
        # Fail at startup rather than at the model's first query-less template search.
        db.reaction_template_library_overview(conn)
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
    tools = SynthesisTools(db_path, toxicity=toxicity)
    tools.screen  # Fail at startup, not on the first submission, if the watchlist is missing.
    executor = AsyncToolExecutor(
        max_concurrent_executions,
        use_process_pool=use_process_pool,
        tool_worker_config=ToolWorkerConfig(
            db_path=str(Path(db_path).resolve()),
            toxicity=toxicity,
        ),
        tools=tools,
    )
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOL_SCHEMAS}

    class SynClawMCP(MCPServer):
        async def list_tools(self) -> list[Tool]:
            # MCPServer derives schemas from the signatures; advertise the canonical ones instead.
            return [
                tool.model_copy(update={"input_schema": deepcopy(schemas[tool.name])})
                for tool in await super().list_tools()
            ]

    server = SynClawMCP("bionemo-synclaw-server")
    for tool in TOOL_SCHEMAS:
        _register_tool(server, executor, tool["function"])
    return server


def serve(
    db_path: str | Path,
    *,
    max_concurrent_executions: int = 64,
    toxicity: ToxicityConfig | None = None,
) -> None:
    """Build the MCP server and run it until it is stopped."""

    create_server(
        db_path,
        max_concurrent_executions=max_concurrent_executions,
        toxicity=toxicity,
    ).run()
