"""Generate articulated objects with an agent that writes build123d.

Package metadata and lazy Python API exports live here, so importing
``mini_articraft`` costs almost nothing. The agent's plugin interfaces are in
``mini_articraft.agent``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mini_articraft.api import (
        Event,
        EventHandler,
        GenerationResult,
        GenerationStatus,
        Provider,
        generate,
        generate_async,
    )

__version__ = "0.1.0"

package_dir = Path(__file__).resolve().parent

_LAZY_EXPORTS = (
    "Event",
    "EventHandler",
    "GenerationResult",
    "GenerationStatus",
    "Provider",
    "generate",
    "generate_async",
)


def __getattr__(name: str) -> Any:
    # Resolved lazily so `import mini_articraft.sdk` in a compile worker does
    # not load the model adapters and their client libraries.
    if name in _LAZY_EXPORTS:
        from mini_articraft import api

        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "Event",
    "EventHandler",
    "GenerationResult",
    "GenerationStatus",
    "Provider",
    "__version__",
    "generate",
    "generate_async",
    "package_dir",
]
