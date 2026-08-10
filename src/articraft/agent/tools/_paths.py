from __future__ import annotations

from pathlib import Path


def scoped_path(base: Path, raw: str, label: str) -> Path:
    if not raw:
        raise ValueError("path is required")
    base = base.resolve()
    target = Path(raw)
    target = target if target.is_absolute() else base / target
    target = target.resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path must stay inside the {label}") from exc
    return target
