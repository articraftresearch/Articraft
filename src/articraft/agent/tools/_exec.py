"""Interactive subprocess sessions for exec_command and write_stdin."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from articraft.agent._child_process import child_environment
from articraft.agent.tools._paths import scoped_path

DEFAULT_YIELD_TIME_MS = 10_000
MAX_YIELD_TIME_MS = 30_000
MAX_OUTPUT_TOKENS = 10_000
MAX_OUTPUT_BYTES = 1024 * 1024
SESSION_CLOSE_TIMEOUT = 5.0

LOGGER = logging.getLogger(__name__)

YIELD_TIME_MS_PROPERTY = {
    "type": "integer",
    "description": (
        f"Wait before yielding output. Defaults to {DEFAULT_YIELD_TIME_MS} ms; "
        f"values above {MAX_YIELD_TIME_MS} ms are capped."
    ),
}
MAX_OUTPUT_TOKENS_PROPERTY = {
    "type": "integer",
    "description": (
        f"Output token budget. Defaults to {MAX_OUTPUT_TOKENS} tokens; "
        "larger requests may be capped by policy."
    ),
}


class OutputBuffer:
    """Head/tail byte buffer with a fixed budget.

    Keeps the first ``head_budget`` bytes and the last ``tail_budget`` bytes,
    counting whatever falls out of the middle as ``omitted``. One writer (the
    stream pump) and one reader (poll) share it on a single event loop, so it
    needs no lock.
    """

    def __init__(self) -> None:
        self._head_budget = MAX_OUTPUT_BYTES // 2
        self._tail_budget = MAX_OUTPUT_BYTES - self._head_budget
        self._head = bytearray()
        self._tail = bytearray()
        self._omitted = 0

    def append(self, chunk: bytes) -> None:
        if len(self._head) < self._head_budget:
            head = chunk[: self._head_budget - len(self._head)]
            self._head.extend(head)
            chunk = chunk[len(head) :]
        if not chunk:
            return
        if len(chunk) >= self._tail_budget:
            self._omitted += len(self._tail) + len(chunk) - self._tail_budget
            self._tail = bytearray(chunk[-self._tail_budget :])
            return
        self._tail.extend(chunk)
        overflow = len(self._tail) - self._tail_budget
        if overflow > 0:
            del self._tail[:overflow]
            self._omitted += overflow

    def drain(self) -> tuple[bytes, int]:
        data = bytes(self._head + self._tail)
        omitted = self._omitted
        self._head.clear()
        self._tail.clear()
        self._omitted = 0
        return data, omitted


@dataclass
class ExecSession:
    """A running subprocess plus its output pumps.

    Owns its own lifecycle: reader tasks are stored (not fire-and-forget) so
    ``aclose`` can cancel them, and stdin is closed on teardown. A finished
    session tears itself down from ``poll`` while the loop is still alive.
    """

    session_id: int
    proc: asyncio.subprocess.Process
    stdout: OutputBuffer
    stderr: OutputBuffer
    output_event: asyncio.Event
    _readers: list[asyncio.Task[None]] = field(default_factory=list)
    _closed: bool = False

    @property
    def alive(self) -> bool:
        return self.proc.returncode is None

    @property
    def _streams_closed(self) -> bool:
        return bool(self._readers) and all(task.done() for task in self._readers)

    async def write(self, chars: str) -> None:
        if not chars:
            return
        if not self.alive or self.proc.stdin is None:
            raise ValueError(f"stdin is closed for exec session: {self.session_id}")
        if chars == "\x03":
            _signal_process_group(self.proc, signal.SIGINT)
            return
        self.proc.stdin.write(chars.encode("utf-8"))
        await self.proc.stdin.drain()

    async def poll(
        self,
        *,
        yield_time: float,
        timeout: float | None,
        max_output_chars: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        deadline = started + yield_time
        stdout: list[bytes] = []
        stderr: list[bytes] = []
        omitted = 0
        while True:
            out, out_omitted = self.stdout.drain()
            err, err_omitted = self.stderr.drain()
            if out or err:
                stdout.append(out)
                stderr.append(err)
                omitted += out_omitted + err_omitted
                continue
            if not self.alive and self._streams_closed:
                break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            self.output_event.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.output_event.wait(), timeout=remaining)

        timed_out = self.alive and timeout is not None and time.perf_counter() - started >= timeout
        if timed_out:
            _signal_process_group(self.proc, signal.SIGKILL)
            await self.proc.wait()
            out, out_omitted = self.stdout.drain()
            err, err_omitted = self.stderr.drain()
            stdout.append(out)
            stderr.append(err)
            omitted += out_omitted + err_omitted

        if not self.alive:
            await self.aclose()

        return {
            "chunk_id": secrets.token_hex(3),
            "stdout": _decode(b"".join(stdout), max_output_chars),
            "stderr": _decode(b"".join(stderr), max_output_chars),
            "returncode": self.proc.returncode,
            "session_id": self.session_id if self.alive else None,
            "running": self.alive,
            "timed_out": timed_out,
            "wall_time_seconds": round(time.perf_counter() - started, 4),
            "omitted_bytes": omitted,
        }

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        wait_error: Exception | None = None
        if self.alive:
            _signal_process_group(self.proc, signal.SIGKILL)
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=SESSION_CLOSE_TIMEOUT)
            except Exception as exc:
                wait_error = exc
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        for task in self._readers:
            task.cancel()
        if self._readers:
            await asyncio.gather(*self._readers, return_exceptions=True)
        if wait_error is not None:
            raise wait_error

    def _spawn_readers(self) -> None:
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None
        self._readers = [
            asyncio.create_task(self._read_stream(self.proc.stdout, self.stdout)),
            asyncio.create_task(self._read_stream(self.proc.stderr, self.stderr)),
            # Commands may not leave detached work running after the tracked
            # shell exits; cancelled on aclose like the stream readers.
            asyncio.create_task(self._reap_descendants_after_exit()),
        ]

    async def _reap_descendants_after_exit(self) -> None:
        await self.proc.wait()
        _signal_process_group(self.proc, signal.SIGKILL)
        # Wake any poll that is waiting. The stream readers signal once at EOF,
        # and a poll that woke on that signal before this task finished would
        # find the exit condition still false with nothing left to wake it, so
        # it would wait out its whole deadline for a command already done.
        self.output_event.set()

    async def _read_stream(self, stream: asyncio.StreamReader, buffer: OutputBuffer) -> None:
        try:
            while chunk := await stream.read(8192):
                buffer.append(chunk)
                self.output_event.set()
        finally:
            self.output_event.set()


class ExecSessions:
    """Per-run owner of exec sessions.

    Held on ``ToolContext`` so sessions live and die with a single run instead
    of leaking through module-global state.
    """

    def __init__(self) -> None:
        self._next_session_id = 1
        self._sessions: dict[int, ExecSession] = {}

    def live_ids(self) -> tuple[int, ...]:
        """Ids of sessions still running, oldest first."""
        return tuple(
            session_id for session_id, session in sorted(self._sessions.items()) if session.alive
        )

    async def start(self, run_dir: Path, workspace: Path, args: dict[str, Any]) -> ExecSession:
        if live := self.live_ids():
            raise ValueError(
                "an exec_command session is already running; finish it with write_stdin "
                f"before starting another command (session_id={live[0]})"
            )
        cwd = _cwd(workspace, args.get("cwd"))
        command = [
            _shell(args.get("shell")),
            "-lc" if _bool(args.get("login"), default=True) else "-c",
            str(args["command"]),
        ]
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=_env(run_dir, workspace, command[0]),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        session = ExecSession(
            session_id=self._allocate(),
            proc=proc,
            stdout=OutputBuffer(),
            stderr=OutputBuffer(),
            output_event=asyncio.Event(),
        )
        session._spawn_readers()
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: int) -> ExecSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"unknown exec session: {session_id}")
        return session

    async def poll(self, session: ExecSession, args: dict[str, Any]) -> dict[str, Any]:
        chunk = await session.poll(
            yield_time=_yield_time(args),
            timeout=_timeout(args),
            max_output_chars=_char_budget(args.get("max_output_tokens")),
        )
        if not chunk["running"]:
            self._sessions.pop(session.session_id, None)
        return chunk

    async def aclose(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        # Best-effort teardown: close sessions concurrently and log per-session
        # errors so one stuck or failing session can't block the others or mask the
        # run result when this is awaited from the harness `finally`.
        results = await asyncio.gather(
            *(session.aclose() for session in sessions),
            return_exceptions=True,
        )
        for session, result in zip(sessions, results, strict=True):
            if isinstance(result, BaseException):
                LOGGER.warning(
                    "failed to close exec session %s during run cleanup",
                    session.session_id,
                    exc_info=(type(result), result, result.__traceback__),
                )

    def _allocate(self) -> int:
        session_id = self._next_session_id
        self._next_session_id += 1
        return session_id


def parse_session_id(value: object) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError("session_id must be a positive integer")
    try:
        session_id = int(value)
    except ValueError as exc:
        raise ValueError("session_id must be a positive integer") from exc
    if session_id < 1:
        raise ValueError("session_id must be a positive integer")
    return session_id


def _cwd(workspace: Path, raw: object) -> Path:
    if raw is None or str(raw).strip() == "":
        return workspace.resolve()
    path = scoped_path(workspace, str(raw), "run workspace")
    if not path.is_dir():
        raise ValueError("cwd must be a directory inside the run workspace")
    return path


def _env(run_dir: Path, workspace: Path, shell: str) -> dict[str, str]:
    env = child_environment()
    env.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    env["SHELL"] = shell
    env["ARTICRAFT_RUN_DIR"] = str(run_dir)
    env["ARTICRAFT_WORKSPACE_DIR"] = str(workspace)
    env["ARTICRAFT_PYTHON"] = sys.executable
    return env


def _shell(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return os.environ.get("SHELL") or "/bin/sh"


def _bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _yield_time(args: dict[str, Any]) -> float:
    if args.get("timeout") is not None:
        return max(0.0, float(args["timeout"]))
    raw = args.get("yield_time_ms")
    try:
        value = int(raw) if raw is not None else DEFAULT_YIELD_TIME_MS
    except (TypeError, ValueError):
        value = DEFAULT_YIELD_TIME_MS
    return max(0.0, min(value, MAX_YIELD_TIME_MS) / 1000)


def _timeout(args: dict[str, Any]) -> float | None:
    raw = args.get("timeout")
    return float(raw) if raw is not None else None


def _decode(data: bytes, budget: int) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(text) <= budget:
        return text
    if budget == 0:
        return f"…{len(text)} chars truncated…"
    left = text[: budget // 2]
    right = text[-(budget - len(left)) :]
    return f"{left}…{len(text) - budget} chars truncated…{right}"


def _char_budget(raw_tokens: object) -> int:
    tokens = MAX_OUTPUT_TOKENS
    if isinstance(raw_tokens, (int, float, str)):
        try:
            tokens = int(raw_tokens)
        except ValueError:
            tokens = MAX_OUTPUT_TOKENS
    return max(0, min(MAX_OUTPUT_BYTES, tokens * 4))


def _signal_process_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Send ``sig`` to the process group, falling back to the direct child.

    ``start_new_session=True`` makes each session its own group leader, so the
    group id equals the child pid. ``killpg`` is unavailable on some platforms
    (``AttributeError``); either signal may race process exit (``ProcessLookupError``).
    """
    pid = proc.pid
    if pid is None:
        return
    try:
        os.killpg(pid, sig)
    except (AttributeError, ProcessLookupError):
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(sig)
