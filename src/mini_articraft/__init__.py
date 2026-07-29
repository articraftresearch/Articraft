from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

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


__all__ = ["Environment", "Model", "__version__", "package_dir"]
