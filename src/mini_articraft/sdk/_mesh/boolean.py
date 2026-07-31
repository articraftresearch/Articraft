from __future__ import annotations

from collections.abc import Iterable

import manifold3d
import numpy as np

from mini_articraft.sdk._mesh.core import (
    LoftGeometry,
    MeshGeometry,
    Vec2,
    Vec3,
    _ensure_ccw,
    _profile_2d,
)
from mini_articraft.sdk._mesh.health import MeshHealthIssue, _require_healthy_mesh

_BOOLEAN_ALLOWED_ISSUES = (MeshHealthIssue.MULTIPLE_COMPONENTS,)


def _as_manifold(geometry: MeshGeometry, *, name: str) -> manifold3d.Manifold:
    if not isinstance(geometry, MeshGeometry):
        raise TypeError(f"{name} must be MeshGeometry")
    vertices, faces = geometry._mesh_arrays()
    if not len(vertices) or not len(faces):
        raise ValueError(f"{name} must be a non-empty closed manifold solid")
    mesh = manifold3d.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.uint32),
    )
    result = manifold3d.Manifold(mesh)
    if result.status() != manifold3d.Error.NoError:
        raise ValueError(
            f"{name} must be a closed manifold solid for boolean operations "
            f"(status={result.status()})"
        )
    return result


def _from_manifold(manifold: manifold3d.Manifold) -> MeshGeometry:
    if manifold.is_empty():
        return MeshGeometry()
    mesh = manifold.to_mesh()
    vertices = np.asarray(mesh.vert_properties, dtype=np.float64)
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return MeshGeometry(
        vertices=[(float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in vertices],
        faces=[(int(face[0]), int(face[1]), int(face[2])) for face in faces],
    )


def boolean_union(a: MeshGeometry, b: MeshGeometry) -> MeshGeometry:
    result = _from_manifold(_as_manifold(a, name="a") + _as_manifold(b, name="b"))
    return _require_healthy_mesh(
        result,
        operation="boolean_union",
        allowed_issues=_BOOLEAN_ALLOWED_ISSUES,
    )


def boolean_difference(a: MeshGeometry, b: MeshGeometry) -> MeshGeometry:
    result = _from_manifold(_as_manifold(a, name="a") - _as_manifold(b, name="b"))
    if not result.vertices:
        raise ValueError(
            "boolean_difference produced an empty solid: 'a' is entirely inside 'b', so "
            "nothing is left. Check the two shapes' sizes and positions — a conforming "
            "cut needs 'a' to poke through 'b', not sit fully inside it."
        )
    return _require_healthy_mesh(result, operation="boolean_difference")


def boolean_intersection(a: MeshGeometry, b: MeshGeometry) -> MeshGeometry:
    result = _from_manifold(_as_manifold(a, name="a") ^ _as_manifold(b, name="b"))
    if not result.vertices:
        raise ValueError(
            "boolean_intersection produced an empty solid: the inputs do not overlap. "
            "Position them so they intersect before intersecting."
        )
    return _require_healthy_mesh(result, operation="boolean_intersection")


def _boolean_union_many(geometries: Iterable[MeshGeometry]) -> MeshGeometry:
    values = list(geometries)
    if not values:
        return MeshGeometry()
    result = _as_manifold(values[0], name="geometries[0]")
    for index, geometry in enumerate(values[1:], start=1):
        result += _as_manifold(geometry, name=f"geometries[{index}]")
    return _require_healthy_mesh(_from_manifold(result), operation="boolean_union")


def cut_opening_on_face(
    shell_geometry: MeshGeometry,
    *,
    face: str,
    opening_profile: Iterable[Vec2],
    depth: float,
    offset: Vec2 = (0.0, 0.0),
    taper: float = 0.0,
) -> MeshGeometry:
    """Merge an opening throat into an already-open shell face.

    This deliberately does not subtract a closed face. It mirrors the original
    Articraft helper for authored shells whose opening boundary already exists.
    """

    if not isinstance(shell_geometry, MeshGeometry):
        raise TypeError("shell_geometry must be MeshGeometry")
    if not shell_geometry.vertices:
        raise ValueError("shell_geometry has no vertices")
    depth = float(depth)
    if depth <= 0.0:
        raise ValueError("depth must be positive")
    face = face.strip().lower()
    if face not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError("face must be one of +x, -x, +y, -y, +z, or -z")
    taper = float(taper)
    if abs(taper) >= 0.95:
        raise ValueError("abs(taper) must be less than 0.95")
    profile = _ensure_ccw(_profile_2d(opening_profile))
    center = (
        sum(point[0] for point in profile) / len(profile),
        sum(point[1] for point in profile) / len(profile),
    )
    scale = 1.0 - taper
    inner = [
        (
            center[0] + (point[0] - center[0]) * scale,
            center[1] + (point[1] - center[1]) * scale,
        )
        for point in profile
    ]
    bounds = shell_geometry.bounds
    axis = "xyz".index(face[1])
    sign = 1.0 if face[0] == "+" else -1.0
    plane = bounds[1 if sign > 0.0 else 0][axis]
    offset = (float(offset[0]), float(offset[1]))

    def map_point(point: Vec2, local_depth: float) -> Vec3:
        u, v = point[0] + offset[0], point[1] + offset[1]
        coordinate = plane - sign * local_depth
        if axis == 0:
            return (coordinate, u, v)
        if axis == 1:
            return (u, coordinate, v)
        return (u, v, coordinate)

    shell_geometry.merge(
        LoftGeometry(
            [
                [map_point(point, 0.0) for point in profile],
                [map_point(point, depth) for point in inner],
            ],
            cap=False,
            closed=True,
        )
    )
    return shell_geometry


__all__ = [
    "boolean_difference",
    "boolean_intersection",
    "boolean_union",
    "cut_opening_on_face",
]
