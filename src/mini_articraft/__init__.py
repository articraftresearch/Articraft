"""Generate articulated objects with an agent that writes build123d.

Only package metadata lives here, so importing ``mini_articraft`` costs
nothing. The agent's plugin interfaces are in ``mini_articraft.agent``.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

package_dir = Path(__file__).resolve().parent

__all__ = ["__version__", "package_dir"]
