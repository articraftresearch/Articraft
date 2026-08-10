from __future__ import annotations

import math

import igl
import manifold3d
import numpy as np
from trimesh import Trimesh

from articraft.sdk._mesh.boolean import (
    _as_manifold,
    _from_manifold,
)
from articraft.sdk._mesh.core import MeshGeometry
from articraft.sdk._mesh.health import (
    MeshHealthIssue,
    _require_healthy_mesh,
    analyze_mesh_health,
)

_AXES = {"x": 0, "y": 1, "z": 2}
_FAR = 1.0e6
_FIELD_CHUNK = 500_000
_MAX_GRID_POINTS = 4_000_000
_BLEND_PROFILES = {"tight": -0.5, "round": 0.0, "soft": 1.0}


class SnapRefused(ValueError):
    """Raised when snapping a piece to touch would move it further than allowed."""


def _closest_points(mesh: Trimesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    squared, _faces, closest = igl.point_mesh_squared_distance(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int64),
    )
    distances = np.sqrt(np.maximum(np.asarray(squared, dtype=np.float64), 0.0))
    return np.asarray(closest, dtype=np.float64), distances


def _require_mesh(geometry: object, label: str) -> MeshGeometry:
    if not isinstance(geometry, MeshGeometry):
        raise TypeError(f"{label} must be MeshGeometry")
    return geometry


def _nearest_gap(anchor: MeshGeometry, piece: MeshGeometry) -> tuple[float, np.ndarray]:
    """Smallest surface gap between two solids and the unit direction piece -> anchor."""
    a_mesh = anchor.to_trimesh()
    b_mesh = piece.to_trimesh()
    points_on_a, dist_b = _closest_points(a_mesh, b_mesh.vertices)
    i = int(np.argmin(dist_b))
    point_a, point_b, dist = points_on_a[i], b_mesh.vertices[i], float(dist_b[i])
    points_on_b, dist_a = _closest_points(b_mesh, a_mesh.vertices)
    j = int(np.argmin(dist_a))
    if dist_a[j] < dist:
        point_a, point_b, dist = a_mesh.vertices[j], points_on_b[j], float(dist_a[j])
    direction = np.asarray(point_a, dtype=np.float64) - np.asarray(point_b, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    unit = direction / norm if norm > 1e-12 else np.zeros(3)
    return dist, unit


def _smooth_max(first: np.ndarray, second: np.ndarray, radius: float, profile: str) -> np.ndarray:
    amount = np.clip(0.5 + 0.5 * (first - second) / radius, 0.0, 1.0)
    blend = amount * (1.0 - amount)
    return (
        second * (1.0 - amount)
        + first * amount
        + radius * blend
        + _BLEND_PROFILES[profile] * 4.0 * radius * blend * blend
    )


def _mesh_field(mesh: Trimesh, points: np.ndarray, band: float) -> np.ndarray:
    values = np.full(len(points), -_FAR, dtype=np.float64)
    minimum, maximum = mesh.bounds
    near = np.all((points >= minimum - band) & (points <= maximum + band), axis=1)
    if np.any(near):
        signed, _faces, _closest, _normals = igl.signed_distance(
            np.ascontiguousarray(points[near], dtype=np.float64),
            np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh.faces, dtype=np.int64),
        )
        # libigl is negative inside, while the blend field is positive inside.
        values[near] = -np.asarray(signed, dtype=np.float64)
    return values


def _grid_points(
    start: int,
    stop: int,
    dimensions: np.ndarray,
    lower: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    flat = np.arange(start, stop, dtype=np.int64)
    yz = int(dimensions[1] * dimensions[2])
    x = flat // yz
    remainder = flat - x * yz
    y = remainder // dimensions[2]
    z = remainder - y * dimensions[2]
    return lower + np.column_stack((x, y, z)) * spacing


def _extract_level_set(
    field: np.ndarray, dimensions: np.ndarray, lower: np.ndarray, spacing: np.ndarray
) -> MeshGeometry:
    nx, ny, nz = (int(value) for value in dimensions)
    point_count = len(field)
    flat = np.arange(point_count, dtype=np.int64)
    x = flat % nx
    y = (flat // nx) % ny
    z = flat // (nx * ny)
    grid_vertices = lower + np.column_stack((x, y, z)) * spacing
    samples = np.ascontiguousarray(field.reshape((nx, ny, nz)).ravel(order="F"))
    vertices, faces, _edges = igl.marching_cubes(
        samples,
        np.ascontiguousarray(grid_vertices, dtype=np.float64),
        nx,
        ny,
        nz,
        0.0,
    )
    solid = manifold3d.Manifold(
        manifold3d.Mesh(
            np.ascontiguousarray(vertices, dtype=np.float32),
            np.ascontiguousarray(faces, dtype=np.uint32)[:, (0, 2, 1)],
        )
    )
    if solid.status() != manifold3d.Error.NoError or solid.is_empty():
        raise ValueError(f"weld level set produced an invalid solid (status={solid.status()})")
    solid = solid.simplify(float(np.min(spacing)) * 0.05)
    return _from_manifold(solid)


def _validate_weld_inputs(geometries: tuple[MeshGeometry, ...]) -> list[Trimesh]:
    meshes: list[Trimesh] = []
    for index, raw_geometry in enumerate(geometries):
        geometry = _require_mesh(raw_geometry, f"geometries[{index}]")
        geometry.validate()
        if not geometry.vertices or not geometry.faces or not geometry.is_watertight:
            raise ValueError(f"geometries[{index}] must be a non-empty closed manifold solid")
        health = analyze_mesh_health(geometry)
        if not health.healthy:
            issues = ", ".join(finding.issue.value for finding in health.findings)
            raise ValueError(f"geometries[{index}] has unhealthy mesh geometry: {issues}")
        mesh = geometry.to_trimesh()
        if _positive_body_count(mesh) != 1:
            raise ValueError(f"geometries[{index}] must contain one connected solid")
        meshes.append(mesh)
    return meshes


def _positive_body_count(mesh: Trimesh) -> int:
    return sum(bool(part.volume > 0.0) for part in mesh.split(only_watertight=True))


def _connection_gaps(
    geometries: tuple[MeshGeometry, ...],
) -> list[tuple[int, int, float]]:
    gaps: list[tuple[int, int, float]] = []
    for first_index, first in enumerate(geometries):
        for second_index in range(first_index + 1, len(geometries)):
            second = geometries[second_index]
            exact = _from_manifold(
                _as_manifold(first, name=f"geometries[{first_index}]")
                + _as_manifold(second, name=f"geometries[{second_index}]")
            )
            gap = (
                0.0
                if _positive_body_count(exact.to_trimesh()) == 1
                else _nearest_gap(first, second)[0]
            )
            gaps.append((first_index, second_index, gap))
    return gaps


def _require_connected_inputs(geometries: tuple[MeshGeometry, ...], *, max_gap: float) -> None:
    connected = {0}
    gaps = _connection_gaps(geometries)
    while True:
        additions = {
            second if first in connected else first
            for first, second, gap in gaps
            if gap <= max_gap and ((first in connected) != (second in connected))
        }
        if not additions:
            break
        connected.update(additions)
    if len(connected) != len(geometries):
        nearest = min(
            gap for first, second, gap in gaps if (first in connected) != (second in connected)
        )
        raise ValueError(
            f"weld inputs are separated by {nearest * 1000.0:.2f} mm; overlap them or "
            "increase max_gap"
        )


def _surface_controls(
    radius: float, tolerance: float | None, profile: str
) -> tuple[float, float, str]:
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    tolerance = radius * 0.25 if tolerance is None else float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    profile = str(profile).strip().lower().replace("-", "_")
    if profile not in _BLEND_PROFILES:
        raise ValueError("profile must be 'tight', 'round', or 'soft'")
    return radius, tolerance, profile


def _smooth_operation(
    meshes: list[Trimesh],
    *,
    radius: float,
    tolerance: float,
    profile: str,
    operation: str,
    trim: Trimesh | None = None,
) -> MeshGeometry:
    if operation == "union":
        minimum = np.min([mesh.bounds[0] for mesh in meshes], axis=0)
        maximum = np.max([mesh.bounds[1] for mesh in meshes], axis=0)
    else:
        minimum, maximum = meshes[0].bounds
    # Keep authored surfaces off the level-set lattice. Exact alignment can create
    # zero-volume corner tetrahedra when a flat face lands directly on the zero level.
    margin = radius + 2.371 * tolerance
    lower = np.asarray(minimum, dtype=np.float64) - margin
    upper = np.asarray(maximum, dtype=np.float64) + margin
    dimensions = np.ceil((upper - lower) / tolerance).astype(np.int64) + 1
    point_count = int(np.prod(dimensions))
    if point_count > _MAX_GRID_POINTS:
        minimum_tolerance = float(np.max(upper - lower) / (math.cbrt(_MAX_GRID_POINTS) - 1.0))
        raise ValueError(
            f"weld tolerance creates {point_count:,} field samples; use tolerance of at "
            f"least {minimum_tolerance:.6g} for this size"
        )
    spacing = (upper - lower) / (dimensions - 1)
    field = np.full(point_count, -_FAR, dtype=np.float64)
    band = radius + 2.0 * tolerance
    for mesh_index, mesh in enumerate(meshes):
        for start in range(0, point_count, _FIELD_CHUNK):
            stop = min(start + _FIELD_CHUNK, point_count)
            points = _grid_points(start, stop, dimensions, lower, spacing)
            values = _mesh_field(mesh, points, band)
            if mesh_index == 0:
                field[start:stop] = values
            elif operation == "difference":
                field[start:stop] = -_smooth_max(-field[start:stop], values, radius, profile)
            else:
                field[start:stop] = _smooth_max(field[start:stop], values, radius, profile)
    if trim is not None:
        for start in range(0, point_count, _FIELD_CHUNK):
            stop = min(start + _FIELD_CHUNK, point_count)
            points = _grid_points(start, stop, dimensions, lower, spacing)
            trim_values = _mesh_field(trim, points, band)
            field[start:stop] = np.minimum(field[start:stop], -trim_values)
    return _extract_level_set(field, dimensions, lower, spacing)


def snap_to(
    anchor: MeshGeometry,
    piece: MeshGeometry,
    *,
    overlap: float = 0.004,
    max_move: float = 0.02,
    axis: str | None = None,
) -> MeshGeometry:
    """Translate ``piece`` toward ``anchor`` until they overlap by about ``overlap``.

    Use this before ``weld`` (or before adding an overlapping shape) when a protrusion
    was placed with a small gap to the form it should meet, so you do not have to hit
    the exact coordinate by hand. Like the other mesh transforms it moves ``piece`` in
    place and returns it; snap the piece BEFORE you add it, and use the returned value,
    so every later check and articulation sees the real snapped position.

    ``snap_to`` only translates the whole piece, so it fits a single freely-placeable
    attachment (a boss, a foot, a spout root). It cannot help a piece whose position is
    fixed by a mechanism (a hinge barrel on its axis) or one that must meet the body at
    more than one place (a handle with two ends) -- move those by fixing their own shape
    or path instead.

    It raises ``SnapRefused`` when closing the gap would move the piece further than
    ``max_move``; a large required move means the piece is constrained or misplaced, and
    silently teleporting it would break the design. Pass ``axis`` (``"x"``, ``"y"``, or
    ``"z"``) to constrain the motion to one direction, e.g. to close a vertical gap
    without shifting a piece off a shared axis.
    """
    anchor = _require_mesh(anchor, "anchor")
    piece = _require_mesh(piece, "piece")
    overlap = float(overlap)
    if overlap < 0.0:
        raise ValueError("overlap must be non-negative")
    max_move = float(max_move)
    if max_move <= 0.0:
        raise ValueError("max_move must be positive")

    dist, unit = _nearest_gap(anchor, piece)
    if axis is not None:
        if axis not in _AXES:
            raise ValueError(f"axis must be one of x, y, z; got {axis!r}")
        k = _AXES[axis]
        sign = np.sign(unit[k]) or 1.0
        unit = np.zeros(3)
        unit[k] = sign

    move = (dist + overlap) * unit
    magnitude = float(np.linalg.norm(move))
    if magnitude > max_move:
        raise SnapRefused(
            f"closing the gap needs {magnitude * 1000:.1f} mm, over max_move "
            f"{max_move * 1000:.0f} mm; the piece is likely constrained or misplaced -- "
            f"fix its placement or shape instead of snapping"
        )
    return piece.translate(float(move[0]), float(move[1]), float(move[2]))


def weld(
    *geometries: MeshGeometry,
    radius: float = 0.006,
    tolerance: float | None = None,
    profile: str = "round",
    max_gap: float = 0.0,
    trim: MeshGeometry | None = None,
) -> MeshGeometry:
    """Fuse solids into one smoothly blended mesh.

    ``radius`` controls the reach of the transition. ``tolerance`` controls the generated
    triangle size and defaults to one quarter of the radius. ``profile`` can be ``tight``,
    ``round``, or ``soft``. Inputs must overlap unless ``max_gap`` allows a small bridge.

    When the body is a hollow shell and a protrusion pokes through the wall into the
    cavity, pass ``trim`` (the solid that fills the cavity) to difference that stub away
    after the smooth union.
    """
    if len(geometries) < 2:
        raise ValueError("weld needs at least two geometries")
    radius, tolerance, profile = _surface_controls(radius, tolerance, profile)
    max_gap = float(max_gap)
    if not math.isfinite(max_gap) or max_gap < 0.0:
        raise ValueError("max_gap must be finite and non-negative")

    meshes = _validate_weld_inputs(geometries)
    _require_connected_inputs(geometries, max_gap=max_gap)
    trim_mesh = None
    if trim is not None:
        trim_geometry = _require_mesh(trim, "trim")
        trim_mesh = _validate_weld_inputs((trim_geometry,))[0]
    result = _smooth_operation(
        meshes,
        radius=radius,
        tolerance=tolerance,
        profile=profile,
        operation="union",
        trim=trim_mesh,
    )
    if not result.vertices:
        raise ValueError("weld produced an empty solid")
    if _positive_body_count(result.to_trimesh()) != 1:
        raise ValueError(
            "weld could not form one connected solid; increase radius or overlap the inputs"
        )
    return _require_healthy_mesh(result, operation="weld")


def smooth_difference(
    geometry: MeshGeometry,
    *cutters: MeshGeometry,
    radius: float = 0.006,
    tolerance: float | None = None,
    profile: str = "round",
    max_gap: float = 0.0,
) -> MeshGeometry:
    """Subtract one or more solids with smooth generated transitions.

    The cutters must overlap ``geometry`` or connect to it through the other cutters.
    ``max_gap`` can allow a nearby cutter to form a shallow blended recess without an
    exact intersection.
    """
    geometry = _require_mesh(geometry, "geometry")
    if not cutters:
        raise ValueError("smooth_difference needs at least one cutter")
    radius, tolerance, profile = _surface_controls(radius, tolerance, profile)
    max_gap = float(max_gap)
    if not math.isfinite(max_gap) or max_gap < 0.0:
        raise ValueError("max_gap must be finite and non-negative")
    inputs = (geometry, *cutters)
    meshes = _validate_weld_inputs(inputs)
    try:
        _require_connected_inputs(inputs, max_gap=max_gap)
    except ValueError as exc:
        raise ValueError(
            "smooth_difference cutters do not reach the geometry; overlap them or increase max_gap"
        ) from exc
    result = _smooth_operation(
        meshes,
        radius=radius,
        tolerance=tolerance,
        profile=profile,
        operation="difference",
    )
    if not result.vertices:
        raise ValueError("smooth_difference removed the entire solid")
    return _require_healthy_mesh(
        result,
        operation="smooth_difference",
        allowed_issues=(MeshHealthIssue.MULTIPLE_COMPONENTS,),
    )


__all__ = ["SnapRefused", "smooth_difference", "snap_to", "weld"]
