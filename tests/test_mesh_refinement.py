from __future__ import annotations

import math

import numpy as np
import pytest

from articraft.sdk import BoxGeometry, MeshGeometry
from articraft.sdk.mesh import refine_mesh, smooth_mesh, subdivide_mesh


def _bumpy_patch() -> MeshGeometry:
    vertices = [
        (x * 0.01, y * 0.01, 0.004 if (x, y) == (1, 1) else 0.0) for y in range(3) for x in range(3)
    ]
    faces: list[tuple[int, int, int]] = []
    for y in range(2):
        for x in range(2):
            start = y * 3 + x
            faces.extend(
                (
                    (start, start + 1, start + 4),
                    (start, start + 4, start + 3),
                )
            )
    return MeshGeometry(vertices, faces)


def test_refine_mesh_limits_edge_length_without_changing_surface() -> None:
    source = BoxGeometry((0.1, 0.1, 0.1))
    source_vertices = list(source.vertices)
    refined = refine_mesh(source, max_edge_length=0.04)
    result = refined.to_trimesh()

    assert len(refined.faces) > len(source.faces)
    assert float(np.max(result.edges_unique_length)) <= 0.04 + 1e-12
    assert refined.bounds[0] == pytest.approx(source.bounds[0])
    assert refined.bounds[1] == pytest.approx(source.bounds[1])
    assert result.volume == pytest.approx(source.to_trimesh().volume)
    assert refined.is_watertight
    assert source.vertices == source_vertices


def test_subdivide_mesh_supports_exact_and_smooth_subdivision() -> None:
    source = BoxGeometry((0.1, 0.1, 0.1))
    exact = subdivide_mesh(source, levels=2)
    rounded = subdivide_mesh(source, smooth=True)

    assert len(exact.faces) == len(source.faces) * 16
    assert exact.bounds[0] == pytest.approx(source.bounds[0])
    assert exact.bounds[1] == pytest.approx(source.bounds[1])
    assert exact.to_trimesh().volume == pytest.approx(source.to_trimesh().volume)
    assert exact.is_watertight

    assert len(rounded.faces) == len(source.faces) * 4
    assert rounded.is_watertight
    assert rounded.to_trimesh().volume < source.to_trimesh().volume


def test_smooth_mesh_moves_interior_vertices_and_preserves_open_boundary() -> None:
    source = _bumpy_patch()
    smoothed = smooth_mesh(source, iterations=4)
    free_boundary = smooth_mesh(source, iterations=4, preserve_boundary=False)
    boundary = (0, 1, 2, 3, 5, 6, 7, 8)

    assert smoothed.faces == source.faces
    for index in boundary:
        assert smoothed.vertices[index] == pytest.approx(source.vertices[index])
    assert 0.0 < smoothed.vertices[4][2] < source.vertices[4][2]
    assert free_boundary.vertices[0] != pytest.approx(source.vertices[0])
    assert source.vertices[4][2] == pytest.approx(0.004)


def test_zero_iterations_return_independent_copies() -> None:
    source = _bumpy_patch()
    subdivided = subdivide_mesh(source, levels=0)
    smoothed = smooth_mesh(source, iterations=0)

    assert subdivided is not source
    assert smoothed is not source
    assert subdivided.vertices == source.vertices
    assert smoothed.faces == source.faces


def test_refinement_parameters_are_validated() -> None:
    mesh = BoxGeometry((0.1, 0.1, 0.1))

    with pytest.raises(ValueError, match="max_edge_length"):
        refine_mesh(mesh, max_edge_length=math.nan)
    with pytest.raises(ValueError, match="max_iterations"):
        refine_mesh(mesh, max_edge_length=0.01, max_iterations=-1)
    with pytest.raises(TypeError, match="levels"):
        subdivide_mesh(mesh, levels=1.5)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="iterations"):
        smooth_mesh(mesh, iterations=-1)
    with pytest.raises(ValueError, match="vertices and faces"):
        refine_mesh(MeshGeometry(), max_edge_length=0.01)
