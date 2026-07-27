from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mini_articraft.api import Generation, Run, RunCancelledError

__version__ = "0.1.0"

package_dir = Path(__file__).resolve().parent


class Model(Protocol):
    async def query(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class Environment(Protocol):
    def create_run(self, run_id: str) -> Path: ...
    def compile_path(self, run_dir: Path | str) -> dict[str, Any]: ...


def __getattr__(name: str) -> Any:
    # Resolved lazily so `import mini_articraft.sdk` in a compile worker does
    # not load the model adapters and their client libraries.
    if name in {"Generation", "Run", "RunCancelledError"}:
        from mini_articraft import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Environment",
    "Generation",
    "Model",
    "Run",
    "RunCancelledError",
    "__version__",
    "package_dir",
]
