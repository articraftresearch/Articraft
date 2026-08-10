from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias, cast

from articraft.sdk.errors import ValidationError

Vec3: TypeAlias = tuple[float, float, float]


class ArticulationType(StrEnum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"


def _as_vec3(value: Sequence[float], *, field_name: str) -> Vec3:
    if isinstance(value, (str, bytes)):
        raise ValidationError(f"{field_name} must have 3 numeric values")
    try:
        values = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must have 3 numeric values") from exc
    if len(values) != 3:
        raise ValidationError(f"{field_name} must have 3 numeric values")
    if any(not math.isfinite(component) for component in values):
        raise ValidationError(f"{field_name} values must be finite")
    return values


def _as_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    name = value.strip()
    if not name:
        raise ValidationError(f"{field_name} must be non-empty")
    return name


def _coerce_part_name(value: object, *, field_name: str) -> str:
    if isinstance(value, str):
        return _as_name(value, field_name=field_name)
    return _as_name(getattr(value, "name", None), field_name=field_name)


def _coerce_articulation_type(value: ArticulationType | str) -> ArticulationType:
    if isinstance(value, ArticulationType):
        return value
    try:
        return ArticulationType(str(value))
    except ValueError as exc:
        raise ValidationError(f"unknown articulation type: {value!r}") from exc


@dataclass(frozen=True)
class Origin:
    """An articulation frame in its parent part, in meters and radians."""

    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz", _as_vec3(self.xyz, field_name="origin.xyz"))
        object.__setattr__(self, "rpy", _as_vec3(self.rpy, field_name="origin.rpy"))


@dataclass(frozen=True)
class MotionLimits:
    """Limits for one rotational or linear degree of freedom."""

    effort: float = 1.0
    velocity: float = 1.0
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        effort = _positive_finite(self.effort, field_name="motion limit effort")
        velocity = _positive_finite(self.velocity, field_name="motion limit velocity")
        lower = _optional_finite(self.lower, field_name="motion limit lower")
        upper = _optional_finite(self.upper, field_name="motion limit upper")
        if lower is not None and upper is not None and lower > upper:
            raise ValidationError("motion limit lower value cannot exceed upper value")
        object.__setattr__(self, "effort", effort)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(eq=False)
class Articulation:
    name: str
    articulation_type: ArticulationType | str
    parent: str
    child: str
    origin: Origin = field(default_factory=Origin)
    axis: Vec3 = (0.0, 0.0, 1.0)
    motion_limits: MotionLimits | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.name = _as_name(self.name, field_name="articulation name")
        self.articulation_type = _coerce_articulation_type(self.articulation_type)
        self.parent = _coerce_part_name(self.parent, field_name="parent")
        self.child = _coerce_part_name(self.child, field_name="child")
        if self.parent == self.child:
            raise ValidationError(f"articulation {self.name!r} parent and child cannot be the same")
        if not isinstance(self.origin, Origin):
            raise ValidationError(f"articulation {self.name!r} origin must be an Origin")
        self.axis = _as_vec3(self.axis, field_name=f"articulation {self.name!r} axis")
        if self.motion_limits is not None and not isinstance(self.motion_limits, MotionLimits):
            raise ValidationError(
                f"articulation {self.name!r} motion_limits must be MotionLimits or None"
            )

        if self.articulation_type == ArticulationType.FIXED:
            if self.motion_limits is not None:
                raise ValidationError(
                    f"fixed articulation {self.name!r} cannot include motion limits"
                )
            return

        if math.hypot(*self.axis) == 0.0:
            raise ValidationError(f"articulation {self.name!r} axis must be non-zero")

        if self.motion_limits is None:
            raise ValidationError(
                f"articulation {self.name!r} must include motion_limits=MotionLimits(...)"
            )
        if self.articulation_type == ArticulationType.CONTINUOUS:
            if self.motion_limits.lower is not None or self.motion_limits.upper is not None:
                raise ValidationError(
                    f"continuous articulation {self.name!r} cannot include lower or upper limits"
                )
            return

        if self.motion_limits.lower is None or self.motion_limits.upper is None:
            raise ValidationError(
                f"articulation {self.name!r} requires lower and upper motion limits"
            )


def _optional_finite(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field_name=field_name)


def _positive_finite(value: object, *, field_name: str) -> float:
    result = _finite(value, field_name=field_name)
    if result <= 0.0:
        raise ValidationError(f"{field_name} must be positive")
    return result


def _finite(value: object, *, field_name: str) -> float:
    try:
        result = float(cast(str, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{field_name} must be finite")
    return result
