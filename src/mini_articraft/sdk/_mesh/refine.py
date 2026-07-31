from __future__ import annotations

import math
from numbers import Integral
from typing import cast

import numpy as np
import trimesh

from mini_articraft.sdk._mesh.core import MeshGeometry


def _require_mesh(mesh: object) -> MeshGeometry:
    if not isinstance(mesh, MeshGeometry):
        raise TypeError("mesh must be MeshGeometry")
    mesh.validate()
    if not mesh.vertices or not mesh.faces:
        raise ValueError("mesh must contain vertices and faces")
    return mesh


def _non_negative_integer(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _arrays(mesh: MeshGeometry) -> tuple[np.ndarray, np.ndarray]:
    value = _require_mesh(mesh).to_trimesh(process=False)
    value.remove_unreferenced_vertices()
    return (
        np.asarray(value.vertices, dtype=np.float64),
        np.asarray(value.faces, dtype=np.int64),
    )


def _from_arrays(vertices: np.ndarray, faces: np.ndarray) -> MeshGeometry:
    return MeshGeometry(
        vertices=[(float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in vertices],
        faces=[(int(face[0]), int(face[1]), int(face[2])) for face in faces],
    )


def _boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    if not len(boundary):
        return np.empty(0, dtype=np.int64)
    return np.unique(boundary.reshape(-1))


def refine_mesh(
    mesh: MeshGeometry,
    *,
    max_edge_length: float,
    max_iterations: int = 10,
) -> MeshGeometry:
    """Split triangles until every edge is at most ``max_edge_length`` long.

    The new vertices stay on the input triangles, so refinement adds density
    without smoothing or changing the represented surface. The input mesh is
    not changed.
    """
    max_edge_length = float(max_edge_length)
    if max_edge_length <= 0.0 or not math.isfinite(max_edge_length):
        raise ValueError("max_edge_length must be finite and positive")
    max_iterations = _non_negative_integer(max_iterations, "max_iterations")
    vertices, faces = _arrays(mesh)
    refined_vertices, refined_faces = cast(
        "tuple[np.ndarray, np.ndarray]",
        trimesh.remesh.subdivide_to_size(
            vertices,
            faces,
            max_edge=max_edge_length,
            max_iter=max_iterations,
            return_index=False,
        ),
    )
    return _from_arrays(refined_vertices, refined_faces)


def subdivide_mesh(
    mesh: MeshGeometry,
    *,
    levels: int = 1,
    smooth: bool = False,
) -> MeshGeometry:
    """Split every triangle into four triangles for each subdivision level.

    With ``smooth=False``, new vertices stay on the input triangles and the
    surface shape is unchanged. With ``smooth=True``, Loop subdivision moves
    vertices to make the surface smoother. Loop subdivision can round sharp
    edges and reduce volume. The input mesh is not changed.
    """
    levels = _non_negative_integer(levels, "levels")
    value = _require_mesh(mesh)
    if levels == 0:
        return value.copy()
    vertices, faces = _arrays(value)
    if smooth:
        vertices, faces = trimesh.remesh.subdivide_loop(
            vertices,
            faces,
            iterations=levels,
        )
    else:
        for _ in range(levels):
            vertices, faces = cast(
                "tuple[np.ndarray, np.ndarray]",
                trimesh.remesh.subdivide(vertices, faces, return_index=False),
            )
    return _from_arrays(vertices, faces)


def smooth_mesh(
    mesh: MeshGeometry,
    *,
    iterations: int = 10,
    preserve_boundary: bool = True,
) -> MeshGeometry:
    """Smooth vertex positions with a Taubin filter that limits shrinkage.

    The face topology is unchanged. When ``preserve_boundary`` is true,
    vertices on edges used by only one triangle stay fixed. The input mesh is
    not changed.
    """
    iterations = _non_negative_integer(iterations, "iterations")
    value = _require_mesh(mesh)
    if iterations == 0:
        return value.copy()
    vertices, faces = _arrays(value)
    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pinned = _boundary_vertices(result) if preserve_boundary else None
    laplacian = trimesh.smoothing.laplacian_calculation(
        result,
        pinned_vertices=pinned if pinned is not None and len(pinned) else None,
    )
    trimesh.smoothing.filter_taubin(
        result,
        iterations=iterations,
        laplacian_operator=laplacian,
    )
    return _from_arrays(
        np.asarray(result.vertices, dtype=np.float64),
        np.asarray(result.faces, dtype=np.int64),
    )


__all__ = ["refine_mesh", "smooth_mesh", "subdivide_mesh"]
