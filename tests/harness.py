"""Modular test environment for articraft.

Verify the generation loop deeply without paying for model calls. Four
lanes, cheapest first:

1. Unit lane -- call pure functions (SDK checks, compile feedback) directly;
   no harness needed.
2. Warm compile lane -- :class:`WarmEnvironment` keeps one compile worker
   subprocess alive for the whole test session: the same isolation, timeout,
   and cleanup contract as ``LocalWorkspace``, but the geometry imports are
   paid once instead of once per compile (~3s -> ~0.1s).
3. Scripted agent lane -- :class:`ScriptedModel` plus :func:`run_scenario`
   drive the full agent loop (tools, reminders, compile freshness, record)
   with a deterministic model that can react to what the agent sends it.
4. Tape lane -- :class:`ReplayHarness` manages any number of named
   recordings: record one from a live (paid) model once, author one from a
   scripted model for free, or install one by hand; then plug any of them
   into :func:`run_scenario` by name for offline regression runs.

The cold lane (``LocalWorkspace``, fresh worker per compile) stays the
reference for the fresh-interpreter and installed-wheel contracts; both
lanes share the worker payload shape and ``local.py``'s result assembly, so
behavior differences between them are bugs, not semantics.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import hashlib
import inspect
import itertools
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, TypeVar, cast

from articraft.agent import Agent, Model, events
from articraft.agent.record import Record, read_conversation
from articraft.agent.tools import Tool, ToolContext
from articraft.agent.tools._core import schema as _tool_schema
from articraft.agent.tools._core import workspace_digest
from articraft.agent.workspace.local import LocalWorkspace, _error_result, _finalize_payload
from articraft.compiler.result import CompilePayload, CompileResult


def fake_compile_payload(**overrides: Any) -> CompilePayload:
    """A structurally complete payload for fakes that only care about a few keys.

    Built through the real producer so a field added to ``CompileResult`` shows
    up here too, instead of every fake drifting from the shape it stands in for.
    """

    payload = CompileResult(status="success").to_payload()
    payload.update(cast(CompilePayload, overrides))
    return payload


T = TypeVar("T")
Item = TypeVar("Item")

Response = dict[str, Any]
ToolCall = dict[str, Any]
Messages = list[dict[str, Any]]


def run(awaitable: Awaitable[T]) -> T:
    """Drive an awaitable on the session event loop owned by conftest."""
    return asyncio.get_event_loop().run_until_complete(awaitable)


# ---------------------------------------------------------------------------
# Queries and responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelQuery:
    """One recorded model query: the exact messages and tools the agent sent."""

    turn: int
    messages: Messages
    tools: list[dict[str, Any]]

    def contains(self, *needles: str) -> bool:
        """True when a single message content includes every ``needle``."""
        if not needles:
            raise ValueError("contains needs at least one needle")
        return any(
            all(needle in content for needle in needles)
            for content in (str(message.get("content")) for message in self.messages)
        )

    def tool_outputs(self) -> list[dict[str, Any]]:
        """Parsed ``function_call_output`` payloads the model has seen so far."""
        return _tool_outputs(self.messages)


def _tool_outputs(messages: Messages) -> list[dict[str, Any]]:
    return [
        _tool_output_payload(message)
        for message in messages
        if message.get("type") == "function_call_output"
    ]


def _tool_output_payload(message: dict[str, Any]) -> dict[str, Any]:
    output = message.get("output") or "{}"
    if isinstance(output, list):
        output = next(
            (
                item.get("text")
                for item in output
                if isinstance(item, dict) and item.get("type") == "input_text"
            ),
            "{}",
        )
    if isinstance(output, dict):
        return output
    parsed = json.loads(str(output))
    return parsed if isinstance(parsed, dict) else {}


def _record_query(
    queries: list[ModelQuery],
    messages: Messages,
    tools: list[dict[str, Any]] | None,
) -> ModelQuery:
    query = ModelQuery(
        turn=len(queries) + 1,
        messages=list(messages),
        tools=list(tools or []),
    )
    queries.append(query)
    return query


Step = Response | Callable[[ModelQuery], "Response | Awaitable[Response]"]

_call_ids = itertools.count(1)


def tool_call(
    name: str, args: dict[str, Any] | None = None, *, call_id: str | None = None
) -> ToolCall:
    """Build one tool call in the wire shape the agent harness consumes.

    Auto-generated ids are unique per process, not per test; pass ``call_id``
    when a test asserts on specific ids.
    """
    return {
        "id": call_id or f"call_{next(_call_ids)}",
        "name": name,
        "arguments": json.dumps(args or {}),
    }


def calls(*tool_calls: ToolCall, **fields: Any) -> Response:
    """A scripted response of tool calls with no visible text."""
    return {"text": "", "tool_calls": list(tool_calls), **fields}


def text(content: str, **fields: Any) -> Response:
    """A scripted response of visible text with no tool calls."""
    return {"text": content, "tool_calls": [], **fields}


def _normalize_response(response: Response) -> Response:
    """Fill the optional response fields without mutating the caller's dict."""
    if not isinstance(response, dict):
        raise TypeError(f"model responses must be dicts, got {type(response).__name__}")
    return {"text": "", "tool_calls": [], "cost": 0.0, "token_usage": {}, **response}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ScriptExhaustedError(RuntimeError):
    """The agent asked for more turns than the script (or tape) provides."""


class TapeError(RuntimeError):
    """A recording is missing, empty, or otherwise unusable."""


class TapeMismatchError(TapeError):
    """A replayed run diverged from the recorded message trajectory."""


@dataclass(frozen=True)
class ModelIdentity:
    """The model attributes the agent harness reads through ``model.config``."""

    openai_model: str = "gpt-test"
    openai_reasoning_effort: str = "low"


class QueuedModel(Generic[Item]):
    """Base for finite models: a queue of canned responses plus a query log.

    Subclasses keep only their semantics -- what a queued item is and how one
    becomes a response. The queue itself (pop with a clear exhaustion error,
    ``remaining``, ``assert_exhausted``) lives here once.
    """

    noun: ClassVar[str] = "item"
    supports_images: bool = True

    def __init__(
        self,
        items: Iterable[Item],
        *,
        model_name: str = "gpt-test",
        reasoning_effort: str = "low",
        context_window_tokens: int = 0,
    ):
        self._queue = list(items)
        self.queries: list[ModelQuery] = []
        self.close_calls = 0
        self.config = ModelIdentity(model_name, reasoning_effort)
        self.context_window_tokens = context_window_tokens

    @property
    def remaining(self) -> int:
        return len(self._queue)

    def _next(self, query: ModelQuery) -> Item:
        if not self._queue:
            raise ScriptExhaustedError(
                f"agent queried turn {query.turn} beyond the {query.turn - 1} {self.noun}(s)"
            )
        return self._queue.pop(0)

    def assert_exhausted(self) -> None:
        """Fail when the agent finished without consuming the whole queue."""
        if self._queue:
            raise AssertionError(
                f"{len(self._queue)} {self.noun}(s) were never consumed; "
                "the agent finished earlier than expected"
            )

    async def close(self) -> None:
        self.close_calls += 1


class ScriptedModel(QueuedModel[Step]):
    """A deterministic ``Model`` that plays a script of responses.

    Each step is either a response dict (see :func:`text` and :func:`calls`)
    or a callable receiving the :class:`ModelQuery` for this turn. Callables
    may assert on the messages the agent sent, react to earlier tool outputs,
    be ``async`` (e.g. block forever to test cancellation), or raise (to test
    model failure handling). Querying past the last step raises
    :class:`ScriptExhaustedError`, which the agent records as a model failure.
    """

    noun: ClassVar[str] = "scripted step"

    async def query(
        self,
        messages: Messages,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Response:
        query = _record_query(self.queries, messages, tools)
        step = self._next(query)
        response = step(query) if callable(step) else step
        if inspect.isawaitable(response):
            response = await response
        return _normalize_response(response)


# ---------------------------------------------------------------------------
# Tape lane: record and replay models
# ---------------------------------------------------------------------------


def _fingerprint(messages: Messages, tools: list[dict[str, Any]] | None = None) -> str:
    """Hash the structural shape of a request, not its payload text.

    Covers message roles/types, tool-call names, call ids, and the offered
    tool set: a run whose conversation or tool surface drifts from the
    recording fails strict replay, while payload text (which embeds
    machine-specific run paths) stays excluded.
    """
    structural: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {
            "role": message.get("role"),
            "type": message.get("type"),
        }
        for key in ("name", "call_id"):
            if message.get(key):
                entry[key] = message[key]
        tool_calls = message.get("tool_calls")
        if tool_calls:
            entry["tool_calls"] = [call.get("name") for call in tool_calls]
        structural.append(entry)
    raw = json.dumps(
        {
            "messages": structural,
            "tools": sorted(str(tool.get("name")) for tool in tools or []),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode()).hexdigest()


def _read_tape(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise TapeError(f"invalid tape row {path}:{lineno}: {exc}") from exc
    return rows


def _append_tape_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, default=str) + "\n")


class RecordingModel:
    """Wrap a live model and append every exchange to a JSONL tape.

    Each line stores a structural fingerprint of the request messages plus the
    full response dict. Record once with real credentials, commit or keep the
    tape, then regression-test offline with :class:`ReplayModel`. Prefer
    :meth:`ReplayHarness.record` over constructing this directly.
    """

    def __init__(self, model: Model, tape: Path | str):
        self.model = model
        self.tape = Path(tape)
        self.queries: list[ModelQuery] = []
        self.config = getattr(model, "config", None)
        self.context_window_tokens = getattr(model, "context_window_tokens", 0)

    async def query(
        self,
        messages: Messages,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Response:
        _record_query(self.queries, messages, tools)
        response = await self.model.query(messages, tools=tools)
        _append_tape_row(
            self.tape,
            {"fingerprint": _fingerprint(messages, tools), "response": response},
        )
        return response

    async def close(self) -> None:
        await self.model.close()

    def assert_exhausted(self) -> None:
        """Delegate to the wrapped model when it is finite (e.g. scripted)."""
        if isinstance(self.model, QueuedModel):
            self.model.assert_exhausted()


class ReplayModel(QueuedModel[dict[str, Any]]):
    """Replay one tape recorded by :class:`RecordingModel`.

    In strict mode every query's structural fingerprint (roles, message
    types, tool-call names, call ids, the offered tool set) must match the
    recording, so a run that diverges from the recorded trajectory fails
    loudly. Payload contents are intentionally excluded: tool outputs embed
    machine-specific run paths.
    Rows without a fingerprint -- hand-authored via :meth:`ReplayHarness.set`
    -- match any request. Prefer :meth:`ReplayHarness.replay` over
    constructing this directly.
    """

    noun: ClassVar[str] = "tape row"

    def __init__(
        self,
        tape: Path | str,
        *,
        strict: bool = True,
        model_name: str = "gpt-replay",
        reasoning_effort: str = "",
    ):
        super().__init__(
            [row for row in _read_tape(Path(tape)) if "response" in row],
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
        self._strict = strict

    async def query(
        self, messages: Messages, *, tools: list[dict[str, Any]] | None = None
    ) -> Response:
        query = _record_query(self.queries, messages, tools)
        entry = self._next(query)
        expected = str(entry.get("fingerprint") or "")
        if self._strict and expected and expected != _fingerprint(messages, tools):
            raise TapeMismatchError(
                f"turn {query.turn} diverged from the recorded trajectory; "
                "re-record the tape if this change is intentional"
            )
        return _normalize_response(entry["response"])


ScenarioModel = ScriptedModel | RecordingModel | ReplayModel


# ---------------------------------------------------------------------------
# Tape lane: the replay harness
# ---------------------------------------------------------------------------

TAPE_ROOT = Path(__file__).resolve().parent / "tapes"
_TAPE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _tape_name(name: str) -> str:
    value = str(name).strip()
    if not _TAPE_NAME.fullmatch(value):
        raise ValueError("tape name must be a simple file name (letters, digits, _ . -)")
    return value


class ReplayHarness:
    """A named library of tape recordings for offline replay testing.

    One recording is one ``<name>.jsonl`` tape under ``root``. Any number
    of recordings coexist; each plugs into :func:`run_scenario` by name::

        library = ReplayHarness(tmp_path / "tapes")

        # record a run into a tape (paid with a live model, free with a
        # ScriptedModel -- the cheap way to author one)
        with library.record("hinged-box", model) as recording:
            run(Agent(recording, env).run("a hinged box"))

        # set: install a hand-authored recording without any run at all
        library.set("plain-box", [calls(tool_call("compile")), text("done")])

        # replay: plug any recording into a scenario by name
        artifacts = run_scenario("a hinged box", model=library.replay("hinged-box"))

        # erase: remove one recording, or clear() to wipe the library
        library.erase("plain-box")

    ``TAPE_ROOT`` (``tests/tapes/``) is the conventional root for
    curated recordings; use a ``tmp_path`` root for ephemeral ones.
    """

    def __init__(self, root: Path | str = TAPE_ROOT):
        self.root = Path(root)

    def path(self, name: str) -> Path:
        """The tape path for a recording name (validated, no traversal)."""
        return self.root / f"{_tape_name(name)}.jsonl"

    def has(self, name: str) -> bool:
        return self.path(name).is_file()

    def names(self) -> list[str]:
        """All recording names in the library, sorted."""
        if not self.root.is_dir():
            return []
        return sorted(path.stem for path in self.root.glob("*.jsonl"))

    def entries(self, name: str) -> list[dict[str, Any]]:
        """Parsed tape rows, e.g. to transform and re-``set`` a recording."""
        path = self.path(name)
        if not path.is_file():
            raise TapeError(f"unknown recording: {name!r}")
        return _read_tape(path)

    @contextmanager
    def record(
        self,
        name: str,
        model: Model,
        *,
        meta: dict[str, Any] | None = None,
    ) -> Iterator[RecordingModel]:
        """Record a fresh tape for ``name`` from the exchanges in the block.

        Any existing tape is replaced immediately. ``meta`` is stored as
        a leading metadata row (e.g. the original prompt); replay skips it.
        Exiting the block without a single recorded exchange raises
        :class:`TapeError` -- an empty tape is always a broken
        record. If the block raises, whatever was recorded stays on disk for
        inspection (the next record replaces it).
        """
        path = self.path(name)
        self.root.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        if meta:
            _append_tape_row(path, {"meta": dict(meta)})
        recorder = RecordingModel(model, path)
        yield recorder
        if not recorder.queries:
            raise TapeError(f"record {name!r} recorded no exchanges")

    def set(self, name: str, rows: Iterable[dict[str, Any]]) -> Path:
        """Install a tape from response dicts or full tape rows.

        Plain response dicts (built with :func:`text`/:func:`calls`) get no
        fingerprint, so strict replay accepts them for any request. Full rows
        (e.g. from :meth:`entries`) keep their fingerprints. Returns the
        tape path.
        """
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if "meta" in row:
                normalized.append({"meta": row["meta"]})
            elif "response" in row:
                normalized.append(
                    {
                        "fingerprint": str(row.get("fingerprint") or ""),
                        "response": row["response"],
                    }
                )
            else:
                normalized.append({"fingerprint": "", "response": _normalize_response(row)})
        if not any("response" in row for row in normalized):
            raise TapeError(f"refusing to install an empty recording: {name!r}")
        path = self.path(name)
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, default=str) + "\n" for row in normalized),
            encoding="utf-8",
        )
        return path

    def meta(self, name: str) -> dict[str, Any]:
        """Metadata recorded with the tape (e.g. the prompt), or ``{}``."""
        for row in self.entries(name):
            if "meta" in row:
                return dict(row["meta"])
        return {}

    def replay(self, name: str, *, strict: bool = True) -> ReplayModel:
        """Plug recording ``name`` into a run as a :class:`ReplayModel`."""
        if not self.has(name):
            raise TapeError(f"unknown recording: {name!r}")
        return ReplayModel(self.path(name), strict=strict)

    def erase(self, name: str) -> bool:
        """Remove one recording. True when one existed."""
        path = self.path(name)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def clear(self) -> int:
        """Remove every recording; returns how many were removed."""
        removed = 0
        for name in self.names():
            self.erase(name)
            removed += 1
        return removed


# ---------------------------------------------------------------------------
# Warm compile lane
# ---------------------------------------------------------------------------

_SERVER_SCRIPT = Path(__file__).resolve().parent / "_compile_server.py"

# A fresh worker announces readiness (its first stdout line) once its imports
# are done. The allowance bounds that handshake, so the per-compile timeout
# can always bound the compile itself, spawn or no spawn.
_WORKER_STARTUP_ALLOWANCE = 30.0


class CompileServerError(RuntimeError):
    """The warm compile worker cannot be used at all."""


# The outcome of one compile request: the payload, or the reason there is
# none (the worker is killed in both failure cases and lazily restarted).
CompileStatus = Literal["ok", "timeout", "died"]


class _CompileServer:
    """One warm compile-worker subprocess shared by all ``WarmEnvironment``s.

    Speaks newline-delimited JSON with ``tests/_compile_server.py``: the
    worker announces readiness after its imports, then answers one
    ``{"run_dir": ...}`` request per line with one compile payload line.
    A compile that times out or takes the worker down with it (e.g.
    ``os._exit`` in workspace code) costs one worker generation; the next
    compile lazily starts a fresh one. Every queued line is tagged with its
    generation so a stale reader from a killed worker can never be mistaken
    for the current compile's response.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: queue.Queue[tuple[int, str | None]] = queue.Queue()
        self._proc: subprocess.Popen[str] | None = None
        self._generation = 0
        atexit.register(self._stop)

    def compile(
        self,
        run_dir: Path,
        *,
        timeout_seconds: float,
    ) -> tuple[CompileStatus, dict[str, Any] | None]:
        """Compile one run through the warm worker.

        Returns ``("ok", payload)``. A compile that times out or takes the
        worker down with it returns ``("timeout", None)`` or
        ``("died", None)``; the worker is restarted on the next compile.
        Compiles are serialized: one worker serves one compile at a time.
        """
        with self._lock:
            generation = self._send(run_dir)
            deadline = time.monotonic() + timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop()
                    return "timeout", None
                try:
                    item_generation, line = self._lines.get(timeout=remaining)
                except queue.Empty:
                    self._stop()
                    return "timeout", None
                if item_generation != generation:
                    continue  # stale line from a killed worker generation
                if line is None:
                    self._stop()
                    return "died", None
                try:
                    return "ok", json.loads(line)
                except json.JSONDecodeError as exc:
                    self._stop()
                    raise CompileServerError(
                        f"warm compile worker returned invalid JSON: {exc}"
                    ) from exc

    def _send(self, run_dir: Path) -> int:
        """Write one request and return the serving worker's generation.

        Retrying once on a write failure is safe: a broken pipe means the
        worker is dead or dying, so no live compiler can hold the request --
        recompiling on a fresh worker never duplicates a compile.
        """
        request = json.dumps({"run_dir": str(run_dir.resolve())}) + "\n"
        for attempt in range(2):
            proc, just_started = self._ensure_running()
            self._drain_stale_lines()
            if just_started and not self._await_ready():
                self._stop()
                if attempt:
                    raise CompileServerError("warm compile worker failed to start") from None
                continue
            try:
                assert proc.stdin is not None
                proc.stdin.write(request)
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._stop()
                if attempt:
                    raise CompileServerError("warm compile worker is not usable") from None
                continue
            return self._generation
        raise AssertionError("unreachable")

    def _await_ready(self) -> bool:
        """Wait for a fresh worker's readiness marker, bounded by the allowance."""
        deadline = time.monotonic() + _WORKER_STARTUP_ALLOWANCE
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                generation, line = self._lines.get(timeout=remaining)
            except queue.Empty:
                return False
            if generation != self._generation:
                continue
            if line is None:
                return False  # the worker died during startup
            try:
                marker = json.loads(line)
            except json.JSONDecodeError:
                return False
            return marker == {"ready": True}

    def _ensure_running(self) -> tuple[subprocess.Popen[str], bool]:
        if self._proc is None or self._proc.poll() is not None:
            self._start()
            assert self._proc is not None
            return self._proc, True
        return self._proc, False

    def _start(self) -> None:
        self._generation += 1
        self._proc = subprocess.Popen(
            [sys.executable, str(_SERVER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        reader = threading.Thread(
            target=self._read_lines,
            args=(self._proc, self._generation),
            daemon=True,
        )
        reader.start()

    def _read_lines(self, proc: subprocess.Popen[str], generation: int) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._lines.put((generation, line))
        finally:
            self._lines.put((generation, None))  # EOF: the worker exited
            if proc.stdout is not None:
                with contextlib.suppress(OSError):
                    proc.stdout.close()  # the reader owns its pipe

    def _drain_stale_lines(self) -> None:
        while True:
            try:
                self._lines.get_nowait()
            except queue.Empty:
                return

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        if proc.poll() is not None:
            proc.wait()
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


class WarmEnvironment(LocalWorkspace):
    """A ``LocalWorkspace`` that compiles through one shared warm worker.

    Same contract as the cold lane -- every compile runs in a worker
    subprocess with the same timeout and cleanup semantics, and result
    assembly is shared with ``LocalWorkspace`` -- but the worker
    (``tests/_compile_server.py``) stays alive between compiles, so the
    geometry imports are paid once per test session instead of once per
    compile (~3s -> ~0.1s each). A compile that times out or crashes the
    worker gets an error result, and the next compile lazily starts a fresh
    worker. Compiles are serialized through the single shared worker.

    ``compile_count`` tracks how many real compiles ran, which makes
    freshness-cache behavior directly assertable.
    """

    _server: ClassVar[_CompileServer | None] = None
    _server_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.compile_count = 0

    @classmethod
    def _shared_server(cls) -> _CompileServer:
        with cls._server_lock:
            if cls._server is None:
                cls._server = _CompileServer()
            return cls._server

    def _run_worker(self, run_dir: Path) -> CompilePayload:
        self.compile_count += 1
        status, payload = self._shared_server().compile(
            run_dir,
            timeout_seconds=self.config.timeout_seconds,
        )
        if status == "timeout":
            return _error_result(
                error=f"compile timed out after {self.config.timeout_seconds:g}s",
            )
        if payload is None:
            return _error_result(
                error="compile worker exited mid-compile "
                "(workspace code may have exited the process)",
            )
        return _finalize_payload(
            payload,
            stderr="",
            returncode=0 if payload["status"] == "success" else 1,
        )


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


class EventRecorder:
    """Collect agent events and answer common questions about them."""

    def __init__(self) -> None:
        self.events: list[events.Event] = []

    def __call__(self, event: events.Event) -> None:
        self.events.append(event)

    def of(self, kind: type[T]) -> list[T]:
        """All events of one type, in order."""
        return [event for event in self.events if isinstance(event, kind)]

    def tool_finishes(self, name: str | None = None) -> list[events.ToolFinished]:
        """``ToolFinished`` events, optionally filtered to one tool."""
        finishes = self.of(events.ToolFinished)
        if name is None:
            return finishes
        return [event for event in finishes if event.name == name]

    @property
    def finished(self) -> events.RunFinished | None:
        """The terminal event, when the run emitted one last."""
        last = self.events[-1] if self.events else None
        return last if isinstance(last, events.RunFinished) else None


@dataclass
class RunArtifacts:
    """Everything a scenario produced, ready for deep assertions."""

    result: dict[str, Any]
    record: Record
    model: ScenarioModel | Model
    recorder: EventRecorder
    run_dir: Path

    @property
    def workspace(self) -> Path:
        return self.run_dir / "workspace"

    @property
    def conversation(self) -> list[dict[str, Any]]:
        return read_conversation(self.run_dir / "conversation.jsonl")

    def tool_outputs(self) -> list[dict[str, Any]]:
        """Parsed tool result payloads from the persisted conversation."""
        return _tool_outputs(self.conversation)


def run_scenario(
    prompt: str,
    script: Iterable[Step] | None = None,
    *,
    model: ScenarioModel | Model | None = None,
    env: LocalWorkspace | None = None,
    tmp_path: Path | None = None,
    run_id: str = "scenario",
    max_turns: int = 10,
    assert_exhausted: bool = True,
    on_event: Callable[[events.Event], None] | None = None,
) -> RunArtifacts:
    """Run the full agent loop for free and return deep-inspectable artifacts.

    The model is a fresh :class:`ScriptedModel` built from ``script`` unless
    ``model=`` plugs in something else -- e.g. one recording from a
    :class:`ReplayHarness`, or a live :class:`~articraft.Model`. Pass
    ``env`` to choose the compile lane (:class:`WarmEnvironment` for speed)
    or ``tmp_path`` for a plain subprocess ``LocalWorkspace``. By default
    finite harness models must be consumed exactly; pass
    ``assert_exhausted=False`` for open-ended or live runs. Every event also
    flows to ``on_event`` when given (the artifacts recorder always sees it
    too).
    """
    if model is None:
        if script is None:
            raise ValueError("run_scenario needs script= or model=")
        model = ScriptedModel(script)
    elif script is not None:
        raise ValueError("run_scenario takes script= or model=, not both")
    if env is None:
        if tmp_path is None:
            raise ValueError("run_scenario needs env= or tmp_path=")
        env = LocalWorkspace(output_dir=tmp_path)
    recorder = EventRecorder()

    def callback(event: events.Event) -> None:
        recorder(event)
        if on_event is not None:
            on_event(event)

    agent = Agent(model, env, max_turns=max_turns, on_event=callback)
    result = run(agent.run(prompt, run_id=run_id))
    if assert_exhausted:
        check = getattr(model, "assert_exhausted", None)
        if callable(check):
            check()
    run_dir = Path(str(result["run"]))
    return RunArtifacts(
        result=result,
        record=Record.load(run_dir / "record.json"),
        model=model,
        recorder=recorder,
        run_dir=run_dir,
    )


# ---------------------------------------------------------------------------
# Tool fakes shared by agent-loop tests
# ---------------------------------------------------------------------------


def stub_schema(name: str) -> dict[str, Any]:
    """A minimal tool schema for monkeypatched tool tables."""
    return _tool_schema(name, name, {}, [])


def compile_success_tool() -> Tool:
    """A fake ``compile`` tool that always succeeds and mints a USDZ file."""

    async def run_compile(context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        usdz = context.run_dir / "result" / "usdz" / "0000.usdz"
        usdz.parent.mkdir(parents=True, exist_ok=True)
        usdz.write_bytes(b"test-usdz")
        result = fake_compile_payload(usdz=str(usdz))
        context.compile_result = result
        context.successful_compile_result = result
        context.successful_compile_digest = workspace_digest(context.workspace)
        return dict(result)

    return Tool("compile", stub_schema("compile"), run_compile)


# ---------------------------------------------------------------------------
# Canonical workspace sources
# ---------------------------------------------------------------------------

GOOD_MAIN_PY = """
from build123d import Box

from articraft.sdk import ArticulatedObject, TestContext, TestReport



def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("box")
    base = model.part("base")
    base.add(Box(0.2, 0.2, 0.1), name="body")
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    return TestContext(object_model).report()
"""
