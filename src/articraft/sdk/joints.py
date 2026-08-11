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


@dataclass(frozen=True)
class SpanTo:
    """A prismatic value that follows the gap to a moving anchor.

    A hydraulic ram, a gas strut, or a turnbuckle does not choose its own
    length: it is however long the gap between its two eyes happens to be. Give
    the far eye as a point on another part and the extension follows it.
    """

    part: str
    point: Vec3 = (0.0, 0.0, 0.0)
    rest_length: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _as_vec3(self.point, field_name="span point"))
        object.__setattr__(
            self, "rest_length", _finite(self.rest_length, field_name="span rest length")
        )


@dataclass(frozen=True)
class AimAt:
    """A revolute value that keeps a reference direction pointed at an anchor.

    This is the other half of a ram: the barrel swivels on its mount so it stays
    lined up with the eye that the rod has to reach.
    """

    part: str
    point: Vec3 = (0.0, 0.0, 0.0)
    reference: Vec3 = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _as_vec3(self.point, field_name="aim point"))
        reference = _as_vec3(self.reference, field_name="aim reference")
        if math.hypot(*reference) == 0.0:
            raise ValidationError("aim reference must be non-zero")
        object.__setattr__(self, "reference", reference)


Drive: TypeAlias = SpanTo | AimAt


@dataclass(eq=False)
class Articulation:
    name: str
    articulation_type: ArticulationType | str
    parent: str
    child: str
    origin: Origin = field(default_factory=Origin)
    axis: Vec3 = (0.0, 0.0, 1.0)
    motion_limits: MotionLimits | None = None
    drive: Drive | None = None
    """What decides this joint's value, when the mechanism decides it rather than the author.

    A driven joint is not posed directly. Its value is solved from the rest of
    the model, so a linkage stays assembled through the whole range instead of
    coming apart when only its driving joint is moved.
    """

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

        if self.drive is not None:
            expected = SpanTo if self.articulation_type == ArticulationType.PRISMATIC else AimAt
            if not isinstance(self.drive, expected):
                raise ValidationError(
                    f"articulation {self.name!r} is {self.articulation_type} so its drive must be "
                    f"{expected.__name__}, not {type(self.drive).__name__}"
                )
            if self.drive.part in (self.parent, self.child):
                raise ValidationError(
                    f"articulation {self.name!r} cannot be driven by its own parent or child; "
                    "a drive reads a part the joint does not move"
                )

        if self.articulation_type == ArticulationType.FIXED:
            if self.motion_limits is not None:
                raise ValidationError(
                    f"fixed articulation {self.name!r} cannot include motion limits"
                )
            if self.drive is not None:
                raise ValidationError(f"fixed articulation {self.name!r} cannot be driven")
            return

        if math.hypot(*self.axis) == 0.0:
            raise ValidationError(f"articulation {self.name!r} axis must be non-zero")

        if isinstance(self.drive, AimAt):
            axis = self.axis
            reference = self.drive.reference
            cross = (
                axis[1] * reference[2] - axis[2] * reference[1],
                axis[2] * reference[0] - axis[0] * reference[2],
                axis[0] * reference[1] - axis[1] * reference[0],
            )
            if math.hypot(*cross) <= 1e-9 * math.hypot(*axis) * math.hypot(*reference):
                raise ValidationError(
                    f"articulation {self.name!r} aim reference is parallel to the joint axis, "
                    "so rotating can never aim it; use a reference perpendicular to the axis"
                )

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


def partition_articulations(
    articulations: Sequence[Articulation],
) -> tuple[list[Articulation], list[Articulation]]:
    """Split articulations into the spanning tree and the loop closures.

    The first articulation declared for each child owns the tree edge. Every
    later articulation reaching the same child closes a kinematic loop: a real
    pin in the mechanism that the reduced coordinate tree cannot carry. One
    deterministic rule, shared by validation, kinematics, and export, so they
    can never disagree about which joint is which.
    """

    tree: list[Articulation] = []
    loops: list[Articulation] = []
    owned: set[str] = set()
    for articulation in articulations:
        if articulation.child in owned:
            loops.append(articulation)
        else:
            owned.add(articulation.child)
            tree.append(articulation)
    return tree, loops
