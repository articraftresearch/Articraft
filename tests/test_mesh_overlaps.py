from __future__ import annotations

import subprocess
import sys

import trimesh

from articraft.sdk import mesh_overlaps


def _box(extents: tuple[float, float, float], centre: tuple[float, float, float]):
    return trimesh.creation.box(extents, trimesh.transformations.translation_matrix(centre))


def test_two_cabinets_pushed_into_each_other_at_a_corner_overlap() -> None:
    (found,) = mesh_overlaps(
        {
            "left": [_box((0.6, 0.6, 0.9), (0.0, 0.0, 0.45))],
            "right": [_box((0.6, 0.6, 0.9), (0.56, 0.56, 0.45))],
        }
    )
    assert {found.part_a, found.part_b} == {"left", "right"}


def test_a_kettle_resting_on_a_worktop_does_not_overlap() -> None:
    worktop = _box((1.0, 0.6, 0.04), (0.0, 0.0, 0.88))
    kettle = _box((0.15, 0.15, 0.25), (0.0, 0.0, 1.025))
    assert mesh_overlaps({"counter": [worktop], "kettle": [kettle]}) == []


def test_a_part_buried_inside_another_overlaps() -> None:
    assert mesh_overlaps(
        {"crate": [_box((1.0, 1.0, 1.0), (0, 0, 0))], "screw": [_box((0.1, 0.1, 0.1), (0, 0, 0))]}
    )


def test_meshes_of_one_part_never_count_against_each_other() -> None:
    meshes = [_box((0.6, 0.6, 0.9), (0, 0, 0)), _box((0.6, 0.6, 0.9), (0.3, 0, 0))]
    assert mesh_overlaps({"table": meshes}) == []


def test_importing_mesh_overlaps_loads_no_geometry_kernel() -> None:
    code = "import sys; from articraft.sdk import mesh_overlaps; print('build123d' in sys.modules)"
    ran = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert ran.stdout.strip() == "False"
