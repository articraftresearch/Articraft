from __future__ import annotations

import numpy as np
import pytest

from mini_articraft.sdk import (
    BoxGeometry,
    CylinderGeometry,
    SphereGeometry,
    analyze_mesh_health,
)
from mini_articraft.sdk.mesh import (
    SnapRefused,
    boolean_difference,
    smooth_difference,
    snap_to,
    weld,
)


def test_weld_two_overlapping_boxes_is_one_solid() -> None:
    a = BoxGeometry((0.05, 0.02, 0.02))
    b = BoxGeometry((0.02, 0.05, 0.02)).translate(0.02, 0.0, 0.0)
    result = weld(a, b, tolerance=0.002)
    assert result.vertices and result.faces
    assert result.is_watertight
    # a single connected component (one fused solid, not two)
    assert result.to_trimesh().body_count == 1


def test_weld_profile_controls_the_smooth_transition() -> None:
    a = BoxGeometry((0.05, 0.02, 0.02))
    b = BoxGeometry((0.02, 0.05, 0.02)).translate(0.02, 0.0, 0.0)
    tight = weld(a, b, radius=0.004, tolerance=0.002, profile="tight")
    soft = weld(a, b, radius=0.004, tolerance=0.002, profile="soft")

    assert tight.is_watertight and soft.is_watertight
    assert tight.to_trimesh().volume < soft.to_trimesh().volume
    assert tight.bounds[1][2] < soft.bounds[1][2]


def test_weld_can_bridge_only_an_allowed_small_gap() -> None:
    first = BoxGeometry((0.02, 0.02, 0.02))
    second = BoxGeometry((0.02, 0.02, 0.02)).translate(0.021, 0.0, 0.0)

    with pytest.raises(ValueError, match="increase max_gap"):
        weld(first, second, radius=0.008, tolerance=0.002)

    bridged = weld(
        first,
        second,
        radius=0.008,
        tolerance=0.002,
        max_gap=0.0011,
    )
    assert bridged.is_watertight
    assert bridged.to_trimesh().body_count == 1


def test_weld_trim_removes_stub_in_hollow_cavity() -> None:
    outer = BoxGeometry((0.10, 0.08, 0.08))
    cavity = BoxGeometry((0.08, 0.06, 0.06))
    shell = boolean_difference(outer, cavity)
    prot = (
        CylinderGeometry(0.01, 0.06, radial_segments=24)
        .rotate_y(np.pi / 2)
        .translate(0.045, 0.0, 0.0)
    )
    without_trim = weld(shell, prot, tolerance=0.0025)
    with_trim = weld(shell, prot, tolerance=0.0025, trim=cavity)
    assert with_trim.to_trimesh().volume < without_trim.to_trimesh().volume
    assert with_trim.is_watertight
    assert analyze_mesh_health(with_trim).healthy


def test_weld_requires_two_solids() -> None:
    with pytest.raises(ValueError):
        weld(BoxGeometry((0.02, 0.02, 0.02)))


def test_weld_rejects_non_meshgeometry() -> None:
    with pytest.raises(TypeError):
        weld(BoxGeometry((0.02, 0.02, 0.02)), np.zeros((3, 3)))  # pyright: ignore[reportArgumentType]


def test_weld_validates_surface_controls() -> None:
    first = BoxGeometry((0.02, 0.02, 0.02))
    second = SphereGeometry(0.012).translate(0.01, 0.0, 0.0)

    with pytest.raises(ValueError, match="profile"):
        weld(first, second, profile="puffy")
    with pytest.raises(ValueError, match="tolerance"):
        weld(first, second, tolerance=0.0)


def test_smooth_difference_rounds_a_cut_with_configurable_fullness() -> None:
    base = BoxGeometry((0.06, 0.05, 0.03))
    cutter = CylinderGeometry(0.012, 0.06, radial_segments=32)
    sharp = boolean_difference(base, cutter)
    tight = smooth_difference(
        base,
        cutter,
        radius=0.004,
        tolerance=0.002,
        profile="tight",
    )
    soft = smooth_difference(
        base,
        cutter,
        radius=0.004,
        tolerance=0.002,
        profile="soft",
    )

    assert tight.is_watertight and soft.is_watertight
    assert tight.to_trimesh().body_count == 1
    assert soft.to_trimesh().volume < tight.to_trimesh().volume
    assert tight.to_trimesh().volume < sharp.to_trimesh().volume


def test_smooth_difference_validates_cutter_reach() -> None:
    base = BoxGeometry((0.04, 0.04, 0.02))
    distant = SphereGeometry(0.005).translate(0.1, 0.0, 0.0)

    with pytest.raises(ValueError, match="at least one cutter"):
        smooth_difference(base)
    with pytest.raises(ValueError, match="do not reach"):
        smooth_difference(base, distant, tolerance=0.002)


def test_snap_to_closes_a_small_gap() -> None:
    body = SphereGeometry(0.05, width_segments=32, height_segments=16)
    spout = (
        CylinderGeometry(0.012, 0.05, radial_segments=24)
        .rotate_y(np.pi / 2)
        .translate(0.085, 0.0, 0.0)
    )
    moved = snap_to(body, spout, overlap=0.004, max_move=0.05)
    fused = weld(body, moved, tolerance=0.002)
    assert fused.is_watertight
    assert fused.to_trimesh().body_count == 1


def test_snap_to_refuses_a_move_beyond_max() -> None:
    barrel = CylinderGeometry(0.008, 0.02, radial_segments=16).rotate_y(np.pi / 2)
    cap = SphereGeometry(0.02, width_segments=16, height_segments=8).translate(0.0, 0.03, 0.0)
    with pytest.raises(SnapRefused):
        snap_to(barrel, cap, overlap=0.003, max_move=0.005)


def test_snap_to_axis_constrains_direction() -> None:
    body = BoxGeometry((0.06, 0.06, 0.02))
    piece = BoxGeometry((0.01, 0.01, 0.02)).translate(0.02, 0.05, 0.0)
    before = piece.to_trimesh().vertices.mean(axis=0)
    moved = snap_to(body, piece, overlap=0.002, max_move=0.06, axis="y")
    after = moved.to_trimesh().vertices.mean(axis=0)
    assert after[0] == pytest.approx(before[0], abs=1e-9)  # x unchanged
    assert after[2] == pytest.approx(before[2], abs=1e-9)  # z unchanged
    assert after[1] < before[1]  # moved in -y toward the body
