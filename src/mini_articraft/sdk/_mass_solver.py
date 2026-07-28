"""Measure a part's mass distribution from its geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk.mass import MassProperties
from mini_articraft.sdk.materials import Material

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
    """One solid for the part: a union when it succeeds, else the closed shapes."""

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
        if union is not None and union.is_watertight and float(union.volume) > 0.0:
            return union
    except Exception:
        pass
    # Fall back to treating the shapes as one body: mass and inertia still add up
    # (parallel-axis, via trimesh's concatenation), overlaps just count twice.
    return trimesh.util.concatenate(solids)


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
