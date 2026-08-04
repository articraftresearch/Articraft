"""Run events emitted by the agent harness.

These are plain, stdlib-only dataclasses so the harness can report progress
without depending on any UI library. The CLI renderer consumes them; the agent
core stays free of rich. A consumer dispatches on the event type with
``isinstance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunStarted:
    run_id: str
    model: str
    prompt: str
    reasoning_effort: str = ""
    context_window_tokens: int = 0


@dataclass(frozen=True)
class TurnStarted:
    turn: int


@dataclass(frozen=True)
class AssistantMessage:
    turn: int
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolStarted:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolFinished:
    call_id: str
    name: str
    payload: dict[str, Any]
    duration: float | None = None


@dataclass(frozen=True)
class ContextCompacted:
    tokens_before: int


@dataclass(frozen=True)
class RunFinished:
    status: str
    run: str
    result: str = ""
    error: str = ""
    turns: int = 0
    duration: float | None = None
    cost: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)


Event = (
    RunStarted
    | TurnStarted
    | AssistantMessage
    | ToolStarted
    | ToolFinished
    | ContextCompacted
    | RunFinished
)
