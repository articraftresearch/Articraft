from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from articraft.sdk._mesh.boolean import boolean_difference, boolean_intersection
from articraft.sdk._mesh.core import BoxGeometry, MeshGeometry
from articraft.sdk.errors import ValidationError

ShellSide = Literal["full", "left", "right", "center"]


@dataclass(frozen=True)
class ShellPartitionRegion:
    name: str
    side: ShellSide = "full"
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValidationError("partition region name must be non-empty")
        object.__setattr__(self, "name", name)
        if self.side not in {"full", "left", "right", "center"}:
            raise ValidationError("partition region side must be full, left, right, or center")
        for low_name, high_name in (
            ("x_min", "x_max"),
            ("y_min", "y_max"),
            ("z_min", "z_max"),
        ):
            for field_name in (low_name, high_name):
                value = getattr(self, field_name)
                if value is not None:
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValidationError(f"{field_name} must be finite")
                    object.__setattr__(self, field_name, value)
            low, high = getattr(self, low_name), getattr(self, high_name)
            if low is not None and high is not None and low >= high:
                raise ValidationError(f"{low_name} must be less than {high_name}")


@dataclass(frozen=True)
class ShellPartitionSpec:
    shell: MeshGeometry
    regions: tuple[ShellPartitionRegion, ...]
    splitters: tuple[MeshGeometry, ...] = ()
    remainder_name: str | None = None
    center_gap: float = 0.0
    padding: float = 0.01

    def __post_init__(self) -> None:
        if (
            not isinstance(self.shell, MeshGeometry)
            or not self.shell.vertices
            or not self.shell.faces
            or not self.shell.is_watertight
        ):
            raise ValidationError("partition shell must be a closed MeshGeometry solid")
        regions = tuple(
            region if isinstance(region, ShellPartitionRegion) else ShellPartitionRegion(**region)  # type: ignore[arg-type]
            for region in self.regions
        )
        if not regions:
            raise ValidationError("partition requires at least one region")
        if len({region.name for region in regions}) != len(regions):
            raise ValidationError("partition region names must be unique")
        object.__setattr__(self, "regions", regions)
        if any(
            not isinstance(splitter, MeshGeometry)
            or not splitter.vertices
            or not splitter.faces
            or not splitter.is_watertight
            for splitter in self.splitters
        ):
            raise ValidationError("partition splitters must be closed MeshGeometry solids")
        center_gap, padding = float(self.center_gap), float(self.padding)
        if center_gap < 0.0 or not math.isfinite(center_gap):
            raise ValidationError("center_gap must be finite and non-negative")
        if padding <= 0.0 or not math.isfinite(padding):
            raise ValidationError("padding must be finite and positive")
        object.__setattr__(self, "center_gap", center_gap)
        object.__setattr__(self, "padding", padding)
        if self.remainder_name is not None:
            name = str(self.remainder_name).strip()
            if not name or name in {region.name for region in regions}:
                raise ValidationError("remainder_name must be non-empty and unique")
            object.__setattr__(self, "remainder_name", name)


def _region_box(
    region: ShellPartitionRegion,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    *,
    center_gap: float,
    padding: float,
) -> MeshGeometry:
    minimum, maximum = bounds
    lows = [
        minimum[0] - padding if region.x_min is None else region.x_min,
        minimum[1] - padding if region.y_min is None else region.y_min,
        minimum[2] - padding if region.z_min is None else region.z_min,
    ]
    highs = [
        maximum[0] + padding if region.x_max is None else region.x_max,
        maximum[1] + padding if region.y_max is None else region.y_max,
        maximum[2] + padding if region.z_max is None else region.z_max,
    ]
    if region.side == "left":
        highs[0] = min(highs[0], -center_gap * 0.5)
    elif region.side == "right":
        lows[0] = max(lows[0], center_gap * 0.5)
    elif region.side == "center":
        if center_gap <= 0.0:
            raise ValidationError("center_gap must be positive for a center region")
        lows[0], highs[0] = max(lows[0], -center_gap * 0.5), min(highs[0], center_gap * 0.5)
    if any(low >= high for low, high in zip(lows, highs, strict=True)):
        raise ValidationError(f"partition region {region.name!r} has empty bounds")
    return BoxGeometry(tuple(high - low for low, high in zip(lows, highs, strict=True))).translate(
        *((low + high) * 0.5 for low, high in zip(lows, highs, strict=True))
    )


def partition_shell(
    spec: ShellPartitionSpec | MeshGeometry, /, **overrides: object
) -> dict[str, MeshGeometry]:
    if isinstance(spec, ShellPartitionSpec):
        value = replace(spec, **overrides) if overrides else spec
    else:
        if "regions" not in overrides:
            raise ValidationError("partition_shell(shell, ...) requires regions")
        value = ShellPartitionSpec(shell=spec, **cast(dict[str, Any], overrides))
    working = value.shell.copy()
    for splitter in value.splitters:
        working = boolean_difference(working, splitter)
    bounds = working.bounds
    remaining = working
    parts: dict[str, MeshGeometry] = {}
    for region in value.regions:
        cutter = _region_box(
            region,
            bounds,
            center_gap=value.center_gap,
            padding=value.padding,
        )
        piece = boolean_intersection(remaining, cutter)
        if not piece.vertices:
            raise ValidationError(f"partition region {region.name!r} captured no geometry")
        parts[region.name] = piece
        remaining = boolean_difference(remaining, piece)
    if value.remainder_name is not None:
        parts[value.remainder_name] = remaining
    return parts


__all__ = ["ShellPartitionRegion", "ShellPartitionSpec", "partition_shell"]
