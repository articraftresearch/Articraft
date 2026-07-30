"""Opt-in load test for the exec reactor.

This reproduces the flake that motivated the session-scoped event loop fix:
under many concurrent/repeated exec calls the subprocess machinery would
intermittently drop stdout or raise "Event loop is closed". It is a stress and
regression guard, not part of the default suite.

Run it explicitly:

    uv run pytest -m volume

It is skipped by default (see ``addopts`` in pyproject.toml).
"""

from __future__ import annotations

import asyncio

import pytest

from mini_articraft.agent.tools import ToolContext, get
from mini_articraft.compiler.runner import LocalEnvironment

SEQUENTIAL_CALLS = 250
CONCURRENT_BATCH = 60


def _run(awaitable):
    return asyncio.get_event_loop().run_until_complete(awaitable)


def _context(tmp_path) -> ToolContext:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("volume")
    return ToolContext(env, run_dir, run_dir / "workspace")


async def _exec(ctx: ToolContext, index: int) -> tuple[int, str]:
    result = await get("exec_command").run(ctx, {"command": f"printf vol-{index}", "login": False})
    return index, str(result["stdout"])


@pytest.mark.volume
def test_exec_command_survives_repeated_loop_entries(tmp_path) -> None:
    """Many separate event-loop entries — the path the flake lived on."""
    ctx = _context(tmp_path)
    lost: list[tuple[int, str]] = []
    for i in range(SEQUENTIAL_CALLS):
        _, out = _run(_exec(ctx, i))
        if out != f"vol-{i}":
            lost.append((i, out))
    assert not lost, f"{len(lost)}/{SEQUENTIAL_CALLS} exec calls lost output: {lost[:5]}"


@pytest.mark.volume
def test_exec_command_survives_concurrency(tmp_path) -> None:
    """Many concurrent sessions sharing one loop."""
    ctx = _context(tmp_path)

    async def drive() -> list[tuple[int, str]]:
        results = await asyncio.gather(*(_exec(ctx, i) for i in range(CONCURRENT_BATCH)))
        return [(i, out) for i, out in results if out != f"vol-{i}"]

    lost = _run(drive())
    assert not lost, f"{len(lost)}/{CONCURRENT_BATCH} exec calls lost output: {lost[:5]}"
