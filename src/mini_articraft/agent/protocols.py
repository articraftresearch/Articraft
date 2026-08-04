"""The agent's swappable pieces.

Three things vary between runs: which model answers, how context is compacted
when it grows too long, and where the work lives. Each is a protocol rather
than a base class so an implementation only has to match the shape -- the
providers, the workspaces, and the test fakes all satisfy these structurally
without importing them.

These live here rather than at the package root because every implementation
and every consumer is inside ``mini_articraft.agent``, and ``Workspace`` has to
name the compiler's payload type to describe what a compile returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mini_articraft.compiler.result import CompilePayload


class Model(Protocol):
    async def query(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...
    async def close(self) -> None: ...


@runtime_checkable
class ContextSummarizer(Protocol):
    async def summarize_context(
        self,
        messages: list[dict[str, Any]],
        *,
        max_output_tokens: int,
    ) -> dict[str, Any]: ...


class Workspace(Protocol):
    """Where the agent's work lives, and therefore where it runs.

    Both halves vary together: whatever machine holds the run directory is the
    machine that has to compile it.
    """

    def create_run(self, run_id: str) -> Path: ...
    def compile_path(self, run_dir: Path | str) -> CompilePayload: ...


__all__ = ["ContextSummarizer", "Model", "Workspace"]
