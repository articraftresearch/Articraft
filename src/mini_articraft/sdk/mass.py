"""Mass properties for parts: what a part is made of, and how heavy that makes it.

A part is one rigid body, so it carries the physical properties a simulator needs:
mass, where that mass sits, and how it resists rotation. Authors supply the material
(or a raw density, or an explicit mass); everything else is measured from the geometry
the part already contains.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk.joints import Vec3, _as_vec3

Quat: tuple[float, float, float, float]


class MaterialDensity(StrEnum):
    """Common materials, used to look up a density in kg/m^3."""

    STEEL = "steel"
    ALUMINUM = "aluminum"
    ABS_PLASTIC = "abs_plastic"
    GLASS = "glass"
    HARDWOOD = "hardwood"
    RUBBER = "rubber"

    @property
    def density(self) -> float:
        """Density in kg/m^3."""

        return _DENSITIES[self]


_DENSITIES: dict[MaterialDensity, float] = {
    MaterialDensity.STEEL: 7850.0,
    MaterialDensity.ALUMINUM: 2700.0,
    MaterialDensity.ABS_PLASTIC: 1050.0,
    MaterialDensity.GLASS: 2500.0,
    MaterialDensity.HARDWOOD: 700.0,
    MaterialDensity.RUBBER: 1200.0,
}


@dataclass(frozen=True, slots=True)
class MassProperties:
    """What a part is made of, and any physical values that override measurement.

    Give exactly one of ``material``, ``density``, or ``mass``:

    - ``material`` looks up a density and multiplies it by the part's measured volume.
    - ``density`` (kg/m^3) does the same with a value you choose.
    - ``mass`` (kg) sets the mass directly, ignoring volume.

    ``center_of_mass`` (meters, in the part's frame), ``diagonal_inertia`` (kg*m^2),
    and ``principal_axes`` (a quaternion, ``(w, x, y, z)``) are measured from the
    part's geometry unless you set them here.
    """

    material: MaterialDensity | None = None
    density: float | None = None
    mass: float | None = None
    center_of_mass: Vec3 | None = None
    diagonal_inertia: Vec3 | None = None
    principal_axes: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        given = [
            name
            for name, value in (
                ("material", self.material),
                ("density", self.density),
                ("mass", self.mass),
            )
            if value is not None
        ]
        if len(given) > 1:
            raise ValidationError(
                "mass properties take only one of material, density, or mass "
                f"(got {', '.join(given)})"
            )
        if not given:
            raise ValidationError(
                "mass properties need one of material, density, or mass; "
                f"materials: {', '.join(m.value for m in MaterialDensity)}"
            )

        if self.material is not None and not isinstance(self.material, MaterialDensity):
            raise ValidationError("mass properties material must be a MaterialDensity")
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

    @property
    def resolved_density(self) -> float | None:
        """The density to apply to the measured volume, if mass is not explicit."""

        if self.material is not None:
            return self.material.density
        return self.density


def _positive(value: object, *, field_name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be a number") from exc
    if not number > 0.0 or number != number or number in (float("inf"), float("-inf")):
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
