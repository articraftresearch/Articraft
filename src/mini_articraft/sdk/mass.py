"""A part's mass: how it is measured, and how to override the measurement.

A part is one rigid body, so it carries the physical properties a simulator needs:
mass, where that mass sits, and how it resists rotation. All of them are measured
from the part's shapes and what those shapes are made of -- see ``Material``.

``MassProperties`` exists for the cases measurement cannot reach: geometry that
stands in for something whose real weight you know, or a mass distribution the
shapes do not express.

``resolve_mass`` is the measurement itself -- shapes plus material densities in,
a mass, centre of mass, and inertia out. It lives beside the override so that
"where does a part's mass come from" has one answer and one file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk._values import Vec3, _as_vec3
from mini_articraft.sdk.materials import Material


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


# A part with no measurable solid still needs some inertia to simulate. This stands
# in for a 1 cm radius of gyration; author diagonal_inertia when the part is larger.
_POINT_RADIUS_OF_GYRATION = 0.01


@dataclass(frozen=True, slots=True)
class ResolvedMass:
    """Final mass values for one part, in kilograms and meters."""

    mass: float
    center_of_mass: tuple[float, float, float]
    diagonal_inertia: tuple[float, float, float]
    principal_axes: tuple[float, float, float, float]


def resolve_mass(
    properties: MassProperties | None,
    shapes: list[tuple[trimesh.Trimesh, Material | None]],
    *,
    part_name: str,
) -> ResolvedMass:
    """Measure a part's mass distribution, then apply any authored overrides."""

    properties = properties or MassProperties()
    measured = _measure(shapes, properties, part_name=part_name)

    if measured is None:
        # No measurable solid: an explicit mass still exports, with a point-like
        # inertia the author can override.
        if properties.mass is None:
            raise ValidationError(
                f"part {part_name!r} has no usable volume to apply a density to; "
                "set an explicit mass, or make the geometry a closed solid"
            )
        mass = properties.mass
        center = properties.center_of_mass or (0.0, 0.0, 0.0)
        point_inertia = mass * _POINT_RADIUS_OF_GYRATION**2
        inertia = properties.diagonal_inertia or (point_inertia,) * 3
        axes = properties.principal_axes or (1.0, 0.0, 0.0, 0.0)
        return ResolvedMass(mass, _triple(center), _triple(inertia), axes)

    mass, measured_center, tensor = measured

    if properties.diagonal_inertia is not None:
        # An authored tensor is taken as given, about whichever center is authored.
        center = properties.center_of_mass or measured_center
        axes = properties.principal_axes or (1.0, 0.0, 0.0, 0.0)
        return ResolvedMass(
            float(mass), _triple(center), _triple(properties.diagonal_inertia), axes
        )

    center = properties.center_of_mass or measured_center
    if properties.center_of_mass is not None:
        # USD expects the inertia about the authored center of mass, so shift the
        # measured tensor there instead of exporting a mismatched pair.
        tensor = _shift_inertia(tensor, mass, measured_center, center)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    if float(np.linalg.det(eigenvectors)) < 0.0:
        eigenvectors[:, 0] *= -1.0  # keep a right-handed frame
    inertia = _triple([max(float(value), 1e-12) for value in eigenvalues])
    axes = properties.principal_axes or _quaternion(eigenvectors)
    return ResolvedMass(float(mass), _triple(center), _triple(inertia), axes)


def _measure(
    shapes: list[tuple[trimesh.Trimesh, Material | None]],
    properties: MassProperties,
    *,
    part_name: str,
) -> tuple[float, tuple[float, float, float], np.ndarray] | None:
    """Mass, center of mass, and inertia tensor from geometry and materials.

    Shapes are grouped by material and each group is unioned before it is
    weighed, so geometry that deliberately overlaps -- a handle end embedded in a
    wall -- is not counted twice. Groups are then combined by mass, which is what
    lets one part be steel where it is steel and hardwood where it is hardwood.
    """

    if properties.mass is not None or properties.density is not None:
        # One density (or one weight) for the whole part: union everything, so an
        # explicit override behaves exactly as it did before materials moved to
        # the shape.
        combined = _combine([mesh for mesh, _ in shapes], part_name=part_name)
        volume = float(combined.volume) if combined is not None else 0.0
        if combined is None or volume <= 0.0:
            return None
        density = properties.density
        mass = properties.mass if properties.mass is not None else (density or 0.0) * volume
        tensor = np.asarray(combined.moment_inertia, dtype=float) * (mass / volume)
        return float(mass), _triple(combined.center_mass), tensor

    missing = [index for index, (_, material) in enumerate(shapes) if material is None]
    if missing:
        raise ValidationError(
            f"part {part_name!r} has {len(missing)} shape(s) with no material, so its "
            "mass cannot be measured; pass material= to part.add(), or set an "
            "explicit mass or density in MassProperties"
        )

    groups: dict[Material, list[trimesh.Trimesh]] = {}
    for mesh, material in shapes:
        assert material is not None
        groups.setdefault(material, []).append(mesh)

    total_mass = 0.0
    weighted_center = np.zeros(3, dtype=float)
    parts: list[tuple[float, np.ndarray, np.ndarray]] = []
    for material, meshes in groups.items():
        combined = _combine(meshes, part_name=part_name)
        volume = float(combined.volume) if combined is not None else 0.0
        if combined is None or volume <= 0.0:
            continue
        mass = material.density * volume
        # trimesh reports inertia about the group's own center at unit density.
        tensor = np.asarray(combined.moment_inertia, dtype=float) * material.density
        center = np.asarray(_triple(combined.center_mass), dtype=float)
        total_mass += mass
        weighted_center += mass * center
        parts.append((mass, center, tensor))

    if not parts or total_mass <= 0.0:
        return None

    center_of_mass = weighted_center / total_mass
    tensor = np.zeros((3, 3), dtype=float)
    for mass, center, group_tensor in parts:
        # Parallel-axis each group from its own center onto the part's.
        tensor += _shift_inertia(group_tensor, mass, _triple(center), _triple(center_of_mass))
    return total_mass, _triple(center_of_mass), tensor


def _shift_inertia(
    tensor: np.ndarray,
    mass: float,
    measured_center: tuple[float, float, float],
    target_center: tuple[float, float, float],
) -> np.ndarray:
    """Parallel-axis: re-express an inertia tensor about a different center."""

    offset = np.asarray(target_center, dtype=float) - np.asarray(measured_center, dtype=float)
    return tensor + mass * (float(offset @ offset) * np.eye(3) - np.outer(offset, offset))


def _triple(values) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in values)
    return (x, y, z)


def _combine(meshes: list[trimesh.Trimesh], *, part_name: str) -> trimesh.Trimesh | None:
    """Combine closed shapes without silently counting overlapping volume twice."""

    solids: list[trimesh.Trimesh] = []
    open_shapes = 0
    for mesh in meshes:
        if not mesh.is_watertight:
            open_shapes += 1
            continue
        solid = mesh
        if float(solid.volume) < 0.0:
            # Inverted winding still describes a real solid; flip it rather than
            # discarding the shape's mass.
            solid = solid.copy()
            solid.fix_normals()
        if float(solid.volume) > 0.0:
            solids.append(solid)
    if open_shapes:
        # Dropping a shape would understate the part's mass with no signal, which is
        # the silent wrongness this feature exists to avoid.
        raise ValidationError(
            f"part {part_name!r} has {open_shapes} shape(s) that are not closed solids, "
            "so its mass cannot be measured; close the geometry or set an explicit mass"
        )
    if not solids:
        return None
    if len(solids) == 1:
        return solids[0]
    try:
        # Shapes are deliberately embedded in each other, so a union avoids
        # counting the shared volume twice.
        union = trimesh.boolean.union(solids)
    except Exception as exc:
        raise ValidationError(
            f"part {part_name!r} shapes could not be combined for mass measurement; "
            "simplify the geometry or set explicit mass properties"
        ) from exc
    if union is None or not union.is_watertight or float(union.volume) <= 0.0:
        raise ValidationError(
            f"part {part_name!r} shapes did not produce a closed solid for mass measurement; "
            "simplify the geometry or set explicit mass properties"
        )
    return union


def _quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """A (w, x, y, z) quaternion for a right-handed rotation matrix."""

    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = 0.5 / ((trace + 1.0) ** 0.5)
        w = 0.25 / scale
        x = float(rotation[2, 1] - rotation[1, 2]) * scale
        y = float(rotation[0, 2] - rotation[2, 0]) * scale
        z = float(rotation[1, 0] - rotation[0, 1]) * scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * ((1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) ** 0.5)
        w = float(rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = float(rotation[0, 1] + rotation[1, 0]) / scale
        z = float(rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * ((1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) ** 0.5)
        w = float(rotation[0, 2] - rotation[2, 0]) / scale
        x = float(rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = float(rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = 2.0 * ((1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) ** 0.5)
        w = float(rotation[1, 0] - rotation[0, 1]) / scale
        x = float(rotation[0, 2] + rotation[2, 0]) / scale
        y = float(rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    return (w / norm, x / norm, y / norm, z / norm)
