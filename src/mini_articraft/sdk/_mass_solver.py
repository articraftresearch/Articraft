"""Measure a part's mass distribution from its geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk.mass import MassProperties

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
    properties: MassProperties,
    meshes: list[trimesh.Trimesh],
    *,
    part_name: str,
) -> ResolvedMass:
    """Combine authored mass properties with the part's measured geometry."""

    combined = _combine(meshes, part_name=part_name)
    volume = float(combined.volume) if combined is not None else 0.0

    density = properties.resolved_density
    if properties.mass is not None:
        mass = properties.mass
    elif volume > 0.0 and density is not None:
        mass = density * volume
    else:
        raise ValidationError(
            f"part {part_name!r} has no usable volume to apply a density to; "
            "set an explicit mass, or make the geometry a closed solid"
        )

    if combined is None or volume <= 0.0:
        # No measurable solid: an explicit mass still exports, with a point-like
        # inertia the author can override.
        center = properties.center_of_mass or (0.0, 0.0, 0.0)
        point_inertia = mass * _POINT_RADIUS_OF_GYRATION**2
        inertia = properties.diagonal_inertia or (point_inertia,) * 3
        axes = properties.principal_axes or (1.0, 0.0, 0.0, 0.0)
        return ResolvedMass(mass, _triple(center), _triple(inertia), axes)

    # trimesh reports inertia for unit density; scale to the resolved mass.
    measured_center = _triple(combined.center_mass)
    scale = mass / volume
    tensor = np.asarray(combined.moment_inertia, dtype=float) * scale
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    if float(np.linalg.det(eigenvectors)) < 0.0:
        eigenvectors[:, 0] *= -1.0  # keep a right-handed frame

    if properties.diagonal_inertia is not None:
        # An authored tensor is taken as given, about whichever center is authored.
        center = properties.center_of_mass or measured_center
        inertia = properties.diagonal_inertia
        axes = properties.principal_axes or (1.0, 0.0, 0.0, 0.0)
        return ResolvedMass(float(mass), _triple(center), _triple(inertia), axes)

    center = properties.center_of_mass or measured_center
    if properties.center_of_mass is not None:
        # USD expects the inertia about the authored center of mass, so shift the
        # measured tensor there instead of exporting a mismatched pair.
        tensor = _shift_inertia(tensor, mass, measured_center, center)
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        if float(np.linalg.det(eigenvectors)) < 0.0:
            eigenvectors[:, 0] *= -1.0
    inertia = _triple([max(float(value), 1e-12) for value in eigenvalues])
    axes = properties.principal_axes or _quaternion(eigenvectors)
    return ResolvedMass(float(mass), _triple(center), _triple(inertia), axes)


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
