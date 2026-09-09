"""Shared USD reads and tree traversal for the viewer and simulator."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from pxr import Usd


def bodies_scope(prim: Usd.Prim) -> Usd.Prim | None:
    """Read the body scope from current or legacy exports."""
    return prim.GetChild("rigid_bodies") or prim.GetChild("parts") or None


def attribute(prim: Usd.Prim, name: str, default: Any = None) -> Any:
    attr = prim.GetAttribute(name)
    value = attr.Get() if attr else None
    return default if value is None else value


def reversed_tree_edges(edges: Sequence[tuple[str, str]], roots: Iterable[str]) -> Iterator[int]:
    """Indices whose endpoints must swap to point away from each component's root.

    Callers supply only tree edges and handle their own frame, axis, and limit
    conversions. Neither loop constraints nor WORLD joints belong in this walk.
    """
    pending = list(range(len(edges)))
    seen = set(roots)
    while pending:
        reachable = [i for i in pending if edges[i][0] in seen or edges[i][1] in seen]
        if not reachable:
            reachable = [pending[0]]
            seen.add(edges[pending[0]][0])
        for index in reachable:
            parent, child = edges[index]
            if parent not in seen:
                yield index
            seen.update((parent, child))
        pending = [index for index in pending if index not in reachable]
