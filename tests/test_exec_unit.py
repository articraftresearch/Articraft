"""Fast, deterministic unit tests for exec internals.

These cover the pure pieces of the exec reactor without spawning subprocesses:
the ``OutputBuffer`` head/tail truncation math and the ``ExecSession`` teardown
lifecycle (``aclose`` closes stdin, cancels its reader tasks, and is idempotent).
The subprocess-level behavior lives in ``test_agent_tools.py`` and the opt-in
``test_exec_volume.py`` load tests.
"""

from __future__ import annotations

import asyncio
from typing import cast

from articraft.agent.tools._exec import (
    MAX_OUTPUT_BYTES,
    ExecSession,
    OutputBuffer,
)

HEAD_BUDGET = MAX_OUTPUT_BYTES // 2
TAIL_BUDGET = MAX_OUTPUT_BYTES - HEAD_BUDGET


def run(awaitable):
    return asyncio.get_event_loop().run_until_complete(awaitable)


def test_output_buffer_passes_small_output_through() -> None:
    buffer = OutputBuffer()
    buffer.append(b"hello ")
    buffer.append(b"world")

    assert buffer.drain() == (b"hello world", 0)


def test_output_buffer_drain_clears_state() -> None:
    buffer = OutputBuffer()
    buffer.append(b"x")

    assert buffer.drain() == (b"x", 0)
    assert buffer.drain() == (b"", 0)


def test_output_buffer_truncates_middle_keeping_head_and_tail() -> None:
    buffer = OutputBuffer()
    dropped = 5_000
    buffer.append(b"H" * HEAD_BUDGET + b"M" * dropped + b"T" * TAIL_BUDGET)

    data, omitted = buffer.drain()

    assert omitted == dropped
    assert len(data) == MAX_OUTPUT_BYTES
    assert data == b"H" * HEAD_BUDGET + b"T" * TAIL_BUDGET


def test_output_buffer_rolls_tail_and_counts_dropped_bytes() -> None:
    buffer = OutputBuffer()
    buffer.append(b"H" * HEAD_BUDGET)  # fills the head exactly
    buffer.append(b"A" * TAIL_BUDGET)  # fills the tail exactly
    buffer.append(b"B" * 100)  # pushes the 100 oldest tail bytes out

    data, omitted = buffer.drain()

    assert omitted == 100
    assert len(data) == MAX_OUTPUT_BYTES
    assert data == b"H" * HEAD_BUDGET + b"A" * (TAIL_BUDGET - 100) + b"B" * 100


class _FakeStdin:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeProc:
    """A stand-in for a finished subprocess (returncode set, so not alive)."""

    def __init__(self) -> None:
        self.returncode: int | None = 0
        self.pid = None
        self.stdin = _FakeStdin()


class _ControlledProc(_FakeProc):
    """A subprocess whose exit order the test controls."""

    def __init__(self) -> None:
        super().__init__()
        self.returncode = None
        self._exit = asyncio.Event()

    async def wait(self) -> int:
        await self._exit.wait()
        self.returncode = 0
        return 0

    def finish(self) -> None:
        self._exit.set()


async def _dead_session() -> tuple[ExecSession, _FakeProc]:
    proc = _FakeProc()
    session = ExecSession(
        session_id=1,
        proc=cast(asyncio.subprocess.Process, proc),
        stdout=OutputBuffer(),
        stderr=OutputBuffer(),
        output_event=asyncio.Event(),
    )

    async def _wait() -> None:
        await session.output_event.wait()

    session._readers = [
        asyncio.create_task(_wait()),
        asyncio.create_task(_wait()),
    ]
    await asyncio.sleep(0)  # let the reader tasks start running
    return session, proc


def test_exec_session_aclose_closes_stdin_and_cancels_readers() -> None:
    async def exercise() -> tuple[ExecSession, _FakeProc, list[asyncio.Task[None]]]:
        session, proc = await _dead_session()
        readers = session._readers
        await session.aclose()
        return session, proc, readers

    session, proc, readers = run(exercise())

    assert session._closed is True
    assert proc.stdin.close_calls == 1
    assert all(task.done() for task in readers)
    assert all(task.cancelled() for task in readers)


def test_exec_session_aclose_is_idempotent() -> None:
    async def exercise() -> tuple[ExecSession, _FakeProc]:
        session, proc = await _dead_session()
        await session.aclose()
        await session.aclose()  # second call is a no-op, must not raise
        return session, proc

    session, proc = run(exercise())

    assert session._closed is True
    assert proc.stdin.close_calls == 1


def test_reaper_completion_wakes_poll_after_streams_close() -> None:
    async def exercise() -> dict[str, object]:
        proc = _ControlledProc()
        session = ExecSession(
            session_id=1,
            proc=cast(asyncio.subprocess.Process, proc),
            stdout=OutputBuffer(),
            stderr=OutputBuffer(),
            output_event=asyncio.Event(),
        )

        async def close_stream() -> None:
            session.output_event.set()

        stream_tasks = [
            asyncio.create_task(close_stream()),
            asyncio.create_task(close_stream()),
        ]
        reaper_task = asyncio.create_task(session._reap_descendants_after_exit())
        session._readers = [*stream_tasks, reaper_task]
        await asyncio.gather(*stream_tasks)

        poll_task = asyncio.create_task(
            session.poll(yield_time=10.0, timeout=None, max_output_chars=100)
        )
        await asyncio.sleep(0)
        assert not poll_task.done()
        assert not session.output_event.is_set()

        proc.finish()
        return await asyncio.wait_for(poll_task, timeout=0.5)

    result = run(exercise())

    assert result["returncode"] == 0
    assert result["running"] is False
