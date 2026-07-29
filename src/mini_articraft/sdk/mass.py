"""Overrides for a part's measured mass properties.

A part is one rigid body, so it carries the physical properties a simulator needs:
mass, where that mass sits, and how it resists rotation. All of them are measured
from the part's shapes and what those shapes are made of -- see ``Material``.

``MassProperties`` exists for the cases measurement cannot reach: geometry that
stands in for something whose real weight you know, or a mass distribution the
shapes do not express.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk.joints import Vec3, _as_vec3


@dataclass(frozen=True, slots=True)
class MassProperties:
    """Physical values that override what the geometry measures.

    Everything here is optional, and anything left out is measured. Reach for
    this only when measurement would be wrong:

    - ``mass`` (kg) sets the part's mass directly, ignoring volume and material.
      Use it when the geometry is a stand-in -- a motor modelled as a plain
      cylinder whose real weight you know.
    - ``density`` (kg/m^3) applies one density to the whole part, for a substance
      the ``Material`` library does not cover.

    ``center_of_mass`` (meters, in the part's frame), ``diagonal_inertia``
    (kg*m^2), and ``principal_axes`` (a quaternion, ``(w, x, y, z)``) replace the
    measured mass distribution.
    """

    density: float | None = None
    mass: float | None = None
    center_of_mass: Vec3 | None = None
    diagonal_inertia: Vec3 | None = None
    principal_axes: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.density is not None and self.mass is not None:
            raise ValidationError(
                "mass properties take either density or mass, not both: mass already "
                "fixes the weight, so a density would have nothing to apply to"
            )
        if self.density is not None:
            object.__setattr__(
                self, "density", _positive(self.density, field_name="mass properties density")
            )
        if self.mass is not None:
            object.__setattr__(
                self, "mass", _positive(self.mass, field_name="mass properties mass")
            )
        if self.center_of_mass is not None:
            object.__setattr__(
                self,
                "center_of_mass",
                _as_vec3(self.center_of_mass, field_name="mass properties center_of_mass"),
            )
        if self.diagonal_inertia is not None:
            inertia = _as_vec3(self.diagonal_inertia, field_name="mass properties diagonal_inertia")
            if any(value <= 0.0 for value in inertia):
                raise ValidationError("mass properties diagonal_inertia must be positive")
            object.__setattr__(self, "diagonal_inertia", inertia)
        if self.principal_axes is not None:
            object.__setattr__(self, "principal_axes", _as_quat(self.principal_axes))


def _positive(value: object, *, field_name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be a number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValidationError(f"{field_name} must be a positive, finite number")
    return number


def _as_quat(value: object) -> tuple[float, float, float, float]:
    try:
        numbers = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ValidationError("mass properties principal_axes must be four numbers") from exc
    if len(numbers) != 4:
        raise ValidationError("mass properties principal_axes must be (w, x, y, z)")
    if not all(number == number and abs(number) != float("inf") for number in numbers):
        raise ValidationError("mass properties principal_axes must be finite")
    norm = sum(number * number for number in numbers) ** 0.5
    if norm <= 0.0:
        raise ValidationError("mass properties principal_axes must not be a zero quaternion")
    return (
        numbers[0] / norm,
        numbers[1] / norm,
        numbers[2] / norm,
        numbers[3] / norm,
    )
