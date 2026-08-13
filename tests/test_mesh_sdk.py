from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from build123d import Box, Pos

from articraft.errors import ValidationError
from articraft.sdk import (
    ArcPipeGeometry,
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    DomeGeometry,
    ExtrudeGeometry,
    ExtrudeWithHolesGeometry,
    LatheGeometry,
    LoftGeometry,
    MeshGeometry,
    PipeGeometry,
    RoundedBoxGeometry,
    SphereGeometry,
    SuperellipsoidGeometry,
    SweepGeometry,
    TorusGeometry,
    WirePolylineGeometry,
    analyze_mesh_health,
)
from articraft.sdk.mesh import (
    LoftSection,
    SectionLoftSpec,
    ShellPartitionRegion,
    ShellPartitionSpec,
    SweepSection,
    WirePath,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    build123d_to_mesh,
    cut_opening_on_face,
    partition_shell,
    repair_loft,
    resample_side_sections,
    rounded_rect_profile,
    sample_arc_3d,
    sample_catmull_rom_spline_2d,
    sample_catmull_rom_spline_3d,
    sample_cubic_bezier_spline_2d,
    sample_cubic_bezier_spline_3d,
    section_loft,
    split_superellipse_side_loft,
    superellipse_profile,
    superellipse_side_loft,
    sweep_profile_along_spline,
    tube_from_spline_points,
    tube_network_from_paths,
    wire_from_points,
)


def test_mesh_geometry_validates_edits_and_transforms() -> None:
    geometry = MeshGeometry()
    a = geometry.add_vertex(0.0, 0.0, 0.0)
    b = geometry.add_vertex(1.0, 0.0, 0.0)
    c = geometry.add_vertex(0.0, 1.0, 0.0)
    geometry.add_face(a, b, c)
    geometry.validate()

    transformed = geometry.copy().scale(2.0, 3.0, 4.0).rotate_z(math.pi / 2).translate(1, 2, 3)
    assert transformed.vertices[1] == pytest.approx((1.0, 4.0, 3.0))
    assert geometry.vertices[1] == (1.0, 0.0, 0.0)

    merged = geometry.copy().merge(geometry)
    assert len(merged.vertices) == 6
    assert merged.faces[-1] == (3, 4, 5)

    with pytest.raises(ValidationError, match="outside"):
        MeshGeometry(vertices=[(0.0, 0.0, 0.0)], faces=[(0, 1, 2)])
    with pytest.raises(ValueError, match="finite"):
        geometry.add_vertex(math.nan, 0.0, 0.0)


def test_mesh_geometry_rotation_round_trip_and_obj_output() -> None:
    pivoted = MeshGeometry(vertices=[(2.0, 0.0, 0.0)]).rotate(
        (0.0, 0.0, 1.0), math.pi / 2, origin=(1.0, 0.0, 0.0)
    )
    around_x = MeshGeometry(vertices=[(0.0, 1.0, 0.0)]).rotate_x(math.pi / 2)
    around_y = MeshGeometry(vertices=[(0.0, 0.0, 1.0)]).rotate_y(math.pi / 2)
    source = BoxGeometry((1.0, 2.0, 3.0))
    round_trip = MeshGeometry.from_trimesh(source.to_trimesh())

    assert pivoted.vertices[0] == pytest.approx((1.0, 1.0, 0.0))
    assert around_x.vertices[0] == pytest.approx((0.0, 0.0, 1.0))
    assert around_y.vertices[0] == pytest.approx((1.0, 0.0, 0.0))
    assert round_trip.bounds[0] == pytest.approx(source.bounds[0])
    assert round_trip.bounds[1] == pytest.approx(source.bounds[1])
    assert round_trip.to_obj().startswith("o mesh\nv ")


@pytest.mark.parametrize(
    "geometry",
    [
        BoxGeometry((0.1, 0.2, 0.3)),
        CylinderGeometry(0.05, 0.2),
        ConeGeometry(0.05, 0.2),
        SphereGeometry(0.1),
        DomeGeometry((0.1, 0.08, 0.06)),
        CapsuleGeometry(0.03, 0.15),
        RoundedBoxGeometry((0.1, 0.08, 0.06), 0.01, tolerance=0.0015),
        SuperellipsoidGeometry((0.1, 0.08, 0.06)),
        TorusGeometry(0.1, 0.02),
    ],
)
def test_primitive_builders_make_watertight_solids(geometry: MeshGeometry) -> None:
    assert geometry.vertices
    assert geometry.faces
    assert geometry.is_watertight


def test_smooth_form_primitives_expose_shape_and_density_controls() -> None:
    rounded = RoundedBoxGeometry(
        (0.1, 0.06, 0.04),
        0.008,
        tolerance=0.002,
        angular_tolerance=0.2,
    )
    ellipsoid = SuperellipsoidGeometry(
        (0.05, 0.03, 0.02),
        radial_segments=32,
        height_segments=16,
    )
    boxy = SuperellipsoidGeometry(
        (0.05, 0.03, 0.02),
        latitude_exponent=0.45,
        longitude_exponent=0.45,
        radial_segments=32,
        height_segments=16,
    )

    assert rounded.bounds[0] == pytest.approx((-0.05, -0.03, -0.02))
    assert rounded.bounds[1] == pytest.approx((0.05, 0.03, 0.02))
    assert ellipsoid.is_watertight and boxy.is_watertight
    assert ellipsoid.to_trimesh().volume == pytest.approx(
        4.0 * math.pi * 0.05 * 0.03 * 0.02 / 3.0,
        rel=0.03,
    )
    assert boxy.to_trimesh().volume > ellipsoid.to_trimesh().volume
    with pytest.raises(ValueError, match="less than half"):
        RoundedBoxGeometry((0.1, 0.06, 0.04), 0.02)
    with pytest.raises(ValueError, match="latitude_exponent"):
        SuperellipsoidGeometry(0.05, latitude_exponent=0.0)


def test_profile_and_spline_helpers_are_deterministic() -> None:
    rounded = rounded_rect_profile(0.2, 0.1, 0.02, corner_segments=3)
    superellipse = superellipse_profile(0.2, 0.1, exponent=3.0, segments=20)
    catmull_2d = sample_catmull_rom_spline_2d(
        [(0.0, 0.0), (0.1, 0.2), (0.2, 0.0)], samples_per_segment=4
    )
    catmull_3d = sample_catmull_rom_spline_3d(
        [(0.0, 0.0, 0.0), (0.1, 0.0, 0.1), (0.2, 0.0, 0.0)],
        samples_per_segment=4,
    )
    bezier_2d = sample_cubic_bezier_spline_2d(
        [(0.0, 0.0), (0.05, 0.1), (0.15, 0.1), (0.2, 0.0)],
        samples_per_segment=4,
    )
    bezier_3d = sample_cubic_bezier_spline_3d(
        [(0.0, 0.0, 0.0), (0.05, 0.0, 0.1), (0.15, 0.0, 0.1), (0.2, 0.0, 0.0)],
        samples_per_segment=4,
    )
    arc = sample_arc_3d(
        start_point=(1.0, 0.0, 0.0),
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        angle=math.pi / 2,
        segments=4,
    )

    assert len(rounded) == 13
    assert len(superellipse) == 20
    assert catmull_2d[0] == (0.0, 0.0)
    assert catmull_3d[-1] == (0.2, 0.0, 0.0)
    assert bezier_2d[-1] == (0.2, 0.0)
    assert bezier_3d[-1] == (0.2, 0.0, 0.0)
    assert arc[-1] == pytest.approx((0.0, 1.0, 0.0), abs=1e-8)


def test_lathe_shell_rejects_round_cap_between_axis_points() -> None:
    outer = [(0.0, -0.12), (0.08, -0.08), (0.09, 0.1)]
    inner = [(0.0, -0.09), (0.06, -0.06), (0.07, 0.08)]

    flat = LatheGeometry.from_shell_profiles(outer, inner, start_cap="flat")

    assert flat.is_watertight
    with pytest.raises(ValueError, match="cannot connect two points on the rotation axis"):
        LatheGeometry.from_shell_profiles(outer, inner, start_cap="round")


def test_lathe_loft_and_extrusions_build_expected_solids() -> None:
    lathe = LatheGeometry([(0.0, -0.1), (0.08, -0.1), (0.08, 0.1), (0.0, 0.1)])
    shell = LatheGeometry.from_shell_profiles(
        [(0.08, -0.1), (0.09, 0.1)],
        [(0.06, -0.08), (0.07, 0.08)],
        end_cap="round",
    )
    loft = LoftGeometry(
        [
            [(-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.1, 0.1, 0.0), (-0.1, 0.1, 0.0)],
            [(-0.05, -0.05, 0.2), (0.05, -0.05, 0.2), (0.05, 0.05, 0.2), (-0.05, 0.05, 0.2)],
        ]
    )
    extrude = ExtrudeGeometry.from_z0(rounded_rect_profile(0.2, 0.1, 0.01), 0.03)
    with_hole = ExtrudeWithHolesGeometry(
        [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
        [[(-0.03, -0.03), (-0.03, 0.03), (0.03, 0.03), (0.03, -0.03)]],
        0.02,
    )

    for geometry in (lathe, shell, loft, extrude, with_hole):
        assert geometry.is_watertight
    assert extrude.bounds[0][2] == pytest.approx(0.0)
    assert with_hole.to_trimesh().volume == pytest.approx((0.04 - 0.0036) * 0.02)


def test_sweep_pipe_wire_and_spline_builders() -> None:
    square = rounded_rect_profile(0.01, 0.006, 0.001)
    path = [(0.0, 0.0, 0.0), (0.04, 0.0, 0.04), (0.08, 0.0, 0.0)]
    wire_path = (
        WirePath((0.0, 0.0, 0.0))
        .line_by(0.02, 0.0, 0.0)
        .bezier_to((0.03, 0.0, 0.03), (0.05, 0.0, 0.03), (0.06, 0.0, 0.0))
    )
    geometries = (
        SweepGeometry(square, [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1)], cap=True),
        PipeGeometry(square, path, cap=True),
        ArcPipeGeometry(
            square,
            start_point=(0.1, 0.0, 0.0),
            center=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            angle=math.pi / 2,
            cap=True,
        ),
        WirePolylineGeometry(path, radius=0.003, cap_ends=True, corner_radius=0.005),
        wire_from_points(wire_path.to_points(), radius=0.003, cap_ends=True),
        tube_from_spline_points(path, radius=0.003),
        sweep_profile_along_spline(path, profile=square),
    )

    assert all(geometry.is_watertight for geometry in geometries)


def test_sweep_orients_profiles_and_caps_concave_profiles() -> None:
    square = [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)]
    horizontal = SweepGeometry(square, [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)], cap=True)
    offset = PipeGeometry(
        [(0.5, -0.01), (1.5, -0.01), (1.5, 0.01), (0.5, 0.01)],
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1)],
        cap=True,
    )
    concave = PipeGeometry(
        [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0)],
        [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)],
        cap=True,
    )

    assert horizontal.is_watertight
    assert horizontal.to_trimesh().volume > 0.0
    assert offset.bounds[0][0] == pytest.approx(0.5)
    assert offset.bounds[1][0] == pytest.approx(1.5)
    assert concave.is_watertight
    assert concave.to_trimesh().volume == pytest.approx(10.0)


def test_sweep_sections_change_profile_scale_rotation_and_offset() -> None:
    base = [(-0.01, -0.006), (0.01, -0.006), (0.01, 0.006), (-0.01, 0.006)]
    middle = rounded_rect_profile(0.03, 0.01, 0.003, corner_segments=2)
    sweep = PipeGeometry(
        base,
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.05), (0.0, 0.0, 0.10)],
        cap=True,
        sections=(
            SweepSection(0.0, scale=0.5),
            SweepSection(
                0.5,
                profile=tuple(middle),
                scale=(1.2, 0.8),
                rotation=math.pi / 4,
                offset=(0.004, -0.002),
            ),
            SweepSection(1.0, scale=1.5),
        ),
    )

    ring_size = len(middle)
    first = sweep.vertices[:ring_size]
    middle_ring = sweep.vertices[ring_size : ring_size * 2]
    last = sweep.vertices[ring_size * 2 : ring_size * 3]

    assert sweep.is_watertight
    assert max(point[0] for point in last) - min(point[0] for point in last) == pytest.approx(0.03)
    assert max(point[0] for point in first) - min(point[0] for point in first) == pytest.approx(
        0.01
    )
    assert (
        max(point[0] for point in middle_ring) + min(point[0] for point in middle_ring)
    ) / 2 == pytest.approx(0.004)
    assert (
        max(point[1] for point in middle_ring) + min(point[1] for point in middle_ring)
    ) / 2 == pytest.approx(-0.002)


def test_sweep_materializes_sections_between_sparse_path_points() -> None:
    profile = [(-0.01, -0.006), (0.01, -0.006), (0.01, 0.006), (-0.01, 0.006)]
    sweep = PipeGeometry(
        profile,
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1)],
        cap=True,
        sections=(SweepSection(0.5, scale=2.0),),
    )
    middle_ring = sweep.vertices[len(profile) : 2 * len(profile)]

    assert len(sweep.vertices) == 3 * len(profile)
    assert max(point[0] for point in middle_ring) - min(point[0] for point in middle_ring) == (
        pytest.approx(0.04)
    )
    assert sweep.is_watertight


def test_sweep_sections_support_smooth_tension_control() -> None:
    profile = [(-0.01, -0.006), (0.01, -0.006), (0.01, 0.006), (-0.01, 0.006)]
    path = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0125),
        (0.0, 0.0, 0.05),
        (0.0, 0.0, 0.1),
    ]
    sections = (
        SweepSection(0.0, scale=1.0),
        SweepSection(0.5, scale=2.0),
        SweepSection(1.0, scale=1.0),
    )
    linear = PipeGeometry(profile, path, cap=True, sections=sections)
    smooth = PipeGeometry(
        profile,
        path,
        cap=True,
        sections=sections,
        section_interpolation="catmull_rom",
        section_tension=0.0,
    )
    ring_size = len(profile)
    linear_ring = linear.vertices[ring_size : 2 * ring_size]
    smooth_ring = smooth.vertices[ring_size : 2 * ring_size]
    linear_width = max(point[0] for point in linear_ring) - min(point[0] for point in linear_ring)
    smooth_width = max(point[0] for point in smooth_ring) - min(point[0] for point in smooth_ring)

    assert smooth_width != pytest.approx(linear_width)
    assert smooth.is_watertight


def test_sweep_builds_round_caps_and_limits_path_segment_length() -> None:
    profile = [(-0.01, -0.006), (0.01, -0.006), (0.01, 0.006), (-0.01, 0.006)]
    rounded = PipeGeometry(
        profile,
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1)],
        cap=True,
        cap_style="round",
        cap_segments=5,
        cap_length=0.012,
        max_segment_length=0.02,
    )

    assert rounded.is_watertight
    assert rounded.bounds[0][2] == pytest.approx(-0.012)
    assert rounded.bounds[1][2] == pytest.approx(0.112)
    assert len(rounded.vertices) == 6 * len(profile) + 2 * (4 * len(profile) + 1)


def test_closed_sweep_uses_one_shared_seam_and_checks_section_match() -> None:
    path = [
        (0.05 * math.cos(angle), 0.05 * math.sin(angle), 0.006 * math.sin(2.0 * angle))
        for angle in (2.0 * math.pi * index / 16 for index in range(16))
    ]
    profile = rounded_rect_profile(0.006, 0.004, 0.001, corner_segments=2)
    sweep = PipeGeometry(profile, path, path_closed=True)

    assert len(sweep.vertices) == len(path) * len(profile)
    assert sweep.is_watertight
    assert sweep.to_trimesh().body_count == 1

    with pytest.raises(ValueError, match="matching section profiles"):
        PipeGeometry(
            profile,
            path,
            path_closed=True,
            sections=(SweepSection(0.0), SweepSection(1.0, scale=1.2)),
        )


def test_sweep_fixed_up_frame_keeps_profile_vertical() -> None:
    profile = [(-0.004, -0.002), (0.004, -0.002), (0.004, 0.002), (-0.004, 0.002)]
    sweep = PipeGeometry(
        profile,
        [(0.0, 0.0, 0.0), (0.05, 0.02, 0.0), (0.10, 0.0, 0.0)],
        frame_mode="fixed_up",
        up_hint=(0.0, 0.0, 1.0),
    )
    ring_size = len(profile)
    for index in range(3):
        ring = sweep.vertices[index * ring_size : (index + 1) * ring_size]
        assert max(point[2] for point in ring) - min(point[2] for point in ring) == pytest.approx(
            0.004
        )


def test_tube_network_unions_crossing_paths() -> None:
    network = tube_network_from_paths(
        [
            [(-0.05, 0.0, 0.0), (0.05, 0.0, 0.0)],
            [(0.0, -0.05, 0.0), (0.0, 0.05, 0.0)],
        ],
        radius=0.004,
        radial_segments=8,
    )

    assert network.is_watertight
    assert network.to_trimesh().body_count == 1


def test_tube_network_honors_corner_and_cap_options() -> None:
    path = [[(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.05, 0.05, 0.0)]]
    miter = tube_network_from_paths(
        path,
        radius=0.004,
        radial_segments=8,
        corner_mode="miter",
        corner_radius=0.01,
    )
    fillet = tube_network_from_paths(
        path,
        radius=0.004,
        radial_segments=8,
        corner_mode="fillet",
        corner_radius=0.01,
        corner_segments=5,
    )
    open_network = tube_network_from_paths(
        path,
        radius=0.004,
        radial_segments=8,
        cap_ends=False,
    )

    assert len(fillet.vertices) > len(miter.vertices)
    assert not open_network.is_watertight


def test_mesh_booleans_return_closed_solids_and_reject_open_meshes() -> None:
    a = BoxGeometry((1.0, 1.0, 1.0))
    b = BoxGeometry((1.0, 1.0, 1.0)).translate(0.5, 0.0, 0.0)

    union = boolean_union(a, b)
    difference = boolean_difference(a, b)
    intersection = boolean_intersection(a, b)

    assert union.to_trimesh().volume == pytest.approx(1.5)
    assert difference.to_trimesh().volume == pytest.approx(0.5)
    assert intersection.to_trimesh().volume == pytest.approx(0.5)
    with pytest.raises(ValueError, match="closed manifold"):
        boolean_union(a, CylinderGeometry(0.2, 1.0, closed=False))
    with pytest.raises(ValueError, match="non-empty closed manifold"):
        boolean_union(a, MeshGeometry())


def test_cached_mesh_properties_refresh_after_direct_mutation() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0))

    assert geometry.bounds[1][0] == pytest.approx(0.5)
    assert geometry.is_watertight

    geometry.vertices[0] = (2.0, -0.5, -0.5)
    geometry.faces.pop()

    assert geometry.bounds[1][0] == pytest.approx(2.0)
    assert geometry.to_trimesh().vertices[:, 0].max() == pytest.approx(2.0)
    assert not geometry.is_watertight


def test_degenerate_boolean_results_raise_actionable_errors() -> None:
    small = BoxGeometry((0.1, 0.1, 0.1))
    big = BoxGeometry((1.0, 1.0, 1.0))
    apart = BoxGeometry((0.1, 0.1, 0.1)).translate(5.0, 0.0, 0.0)

    # 'small' fully inside 'big' -> difference vanishes; a real conform needs overlap-through.
    with pytest.raises(ValueError, match="empty solid"):
        boolean_difference(small, big)
    # non-overlapping inputs -> intersection is empty.
    with pytest.raises(ValueError, match="do not overlap"):
        boolean_intersection(small, apart)


def test_boolean_difference_rejects_disconnected_output() -> None:
    solid = BoxGeometry((1.0, 0.2, 0.2))
    splitter = BoxGeometry((0.1, 0.4, 0.4))

    with pytest.raises(ValueError, match=r"multiple_components=2.*near"):
        boolean_difference(solid, splitter)


def test_build123d_conversion_keeps_shape_location() -> None:
    geometry = build123d_to_mesh(Pos(2.0, 3.0, 4.0) * Box(1.0, 2.0, 3.0))

    assert geometry.bounds[0] == pytest.approx((1.5, 2.0, 2.5))
    assert geometry.bounds[1] == pytest.approx((2.5, 4.0, 5.5))
    assert geometry.is_watertight


def test_side_loft_split_and_resample_helpers() -> None:
    sections = [
        (-0.1, -0.02, 0.02, 0.08),
        (0.0, -0.03, 0.03, 0.1),
        (0.1, -0.02, 0.02, 0.07),
    ]
    whole = superellipse_side_loft(sections, segments=20)
    rear, front, seam = split_superellipse_side_loft(sections, split_y=0.03, segments=20)
    dense = resample_side_sections(sections, samples_per_span=3, smooth_passes=1)

    assert whole.is_watertight
    assert rear.is_watertight and front.is_watertight
    assert seam[0] == pytest.approx(0.03)
    assert len(dense) == 7


def test_section_loft_supports_path_symmetry_and_repair() -> None:
    spec = SectionLoftSpec(
        sections=(
            LoftSection(
                ((0.0, -0.04, 0.0), (0.05, -0.04, 0.0), (0.05, 0.04, 0.0), (0.0, 0.04, 0.0))
            ),
            LoftSection(
                ((0.0, -0.02, 0.1), (0.03, -0.02, 0.1), (0.03, 0.02, 0.1), (0.0, 0.02, 0.1))
            ),
            LoftSection(
                ((0.0, -0.01, 0.2), (0.02, -0.01, 0.2), (0.02, 0.01, 0.2), (0.0, 0.01, 0.2))
            ),
        ),
        path=((0.0, 0.0, 0.0), (0.02, 0.0, 0.1), (0.0, 0.0, 0.2)),
        symmetry="mirror_yz",
    )
    geometry = section_loft(spec)
    dirty = MeshGeometry(
        vertices=[*geometry.vertices, geometry.vertices[0]],
        faces=[*geometry.faces, geometry.faces[0]],
    )
    repaired = repair_loft(dirty)

    assert geometry.bounds[0][0] < -0.04
    assert geometry.bounds[1][0] > 0.04
    assert analyze_mesh_health(geometry).healthy
    assert repaired.is_watertight
    assert len(repaired.vertices) < len(dirty.vertices)
    assert repair_loft(dirty, repair="off").faces == dirty.faces


def test_loft_interpolates_smoothly_between_authored_sections() -> None:
    profiles = [
        [(-0.04, -0.03, 0.0), (0.04, -0.03, 0.0), (0.04, 0.03, 0.0), (-0.04, 0.03, 0.0)],
        [(-0.08, -0.02, 0.1), (0.08, -0.02, 0.1), (0.08, 0.02, 0.1), (-0.08, 0.02, 0.1)],
        [(-0.03, -0.04, 0.2), (0.03, -0.04, 0.2), (0.03, 0.04, 0.2), (-0.03, 0.04, 0.2)],
    ]
    linear = LoftGeometry(profiles, interpolation="linear", samples_per_span=4)
    smooth = LoftGeometry(profiles, interpolation="catmull_rom", samples_per_span=4)

    assert linear.is_watertight and smooth.is_watertight
    assert len(smooth.vertices) == 9 * 4
    assert smooth.vertices[4 * 4] == profiles[1][0]
    assert smooth.vertices[2 * 4] != pytest.approx(linear.vertices[2 * 4])


def test_loft_controls_smooth_parameterization_tension_and_round_caps() -> None:
    profiles = [
        [(-0.03, -0.02, 0.0), (0.03, -0.02, 0.0), (0.03, 0.02, 0.0), (-0.03, 0.02, 0.0)],
        [(-0.06, -0.01, 0.02), (0.06, -0.01, 0.02), (0.06, 0.01, 0.02), (-0.06, 0.01, 0.02)],
        [(-0.02, -0.04, 0.2), (0.02, -0.04, 0.2), (0.02, 0.04, 0.2), (-0.02, 0.04, 0.2)],
    ]
    uniform = LoftGeometry(
        profiles,
        interpolation="catmull_rom",
        samples_per_span=4,
        parameterization="uniform",
    )
    centripetal = LoftGeometry(
        profiles,
        interpolation="catmull_rom",
        samples_per_span=4,
        parameterization="centripetal",
    )
    tensioned = LoftGeometry(
        profiles,
        interpolation="catmull_rom",
        samples_per_span=4,
        parameterization="centripetal",
        tension=0.8,
    )
    rounded = LoftGeometry(
        profiles,
        cap_style="round",
        cap_segments=5,
        cap_length=0.012,
    )

    assert uniform.is_watertight and centripetal.is_watertight and tensioned.is_watertight
    assert centripetal.vertices[4] != pytest.approx(uniform.vertices[4])
    assert tensioned.vertices[4] != pytest.approx(centripetal.vertices[4])
    assert rounded.is_watertight
    assert rounded.bounds[0][2] == pytest.approx(-0.012)
    assert rounded.bounds[1][2] == pytest.approx(0.212)


def test_section_loft_can_orient_sections_to_a_guide_path() -> None:
    outline = ((-0.02, -0.01), (0.02, -0.01), (0.02, 0.01), (-0.02, 0.01))
    sections = tuple(
        LoftSection(tuple((x, y, z) for x, y in outline)) for z in (0.0, 0.025, 0.05, 0.075, 0.1)
    )
    path = ((0.0, 0.0, 0.0), (0.04, 0.0, 0.05), (0.0, 0.0, 0.1))
    geometry = section_loft(
        SectionLoftSpec(
            sections,
            path=path,
            orient_to_path=True,
            frame_mode="parallel_transport",
        )
    )
    first_ring = geometry.to_trimesh().vertices[:4]
    first_normal = np.cross(first_ring[1] - first_ring[0], first_ring[2] - first_ring[0])
    first_normal /= np.linalg.norm(first_normal)
    first_tangent = np.asarray(path[1]) - np.asarray(path[0])
    first_tangent /= np.linalg.norm(first_tangent)

    assert geometry.is_watertight
    assert np.mean(geometry.to_trimesh().vertices[2 * 4 : 3 * 4], axis=0) == pytest.approx(path[1])
    assert abs(float(np.dot(first_normal, first_tangent))) == pytest.approx(1.0)


def test_loft_can_close_around_a_curved_section_path() -> None:
    profiles = []
    major_radius, minor_radius = 0.06, 0.01
    for theta in (2.0 * math.pi * index / 8 for index in range(8)):
        radial = (math.cos(theta), math.sin(theta), 0.0)
        center = (major_radius * radial[0], major_radius * radial[1], 0.0)
        profiles.append(
            [
                (
                    center[0] + minor_radius * math.cos(phi) * radial[0],
                    center[1] + minor_radius * math.cos(phi) * radial[1],
                    minor_radius * math.sin(phi),
                )
                for phi in (2.0 * math.pi * sample / 12 for sample in range(12))
            ]
        )
    loft = LoftGeometry(
        profiles,
        close_path=True,
        interpolation="catmull_rom",
        samples_per_span=2,
    )

    assert len(loft.vertices) == 8 * 2 * 12
    assert loft.is_watertight
    assert loft.to_trimesh().body_count == 1


def test_section_loft_can_preserve_authored_point_correspondence() -> None:
    lower = LoftSection(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)))
    shifted_upper = LoftSection(
        ((1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (-1.0, -1.0, 1.0))
    )
    aligned = section_loft(SectionLoftSpec((lower, shifted_upper)))
    preserved = section_loft(
        SectionLoftSpec((lower, shifted_upper), align_sections=False, repair="off")
    )

    assert aligned.vertices[4] == (-1.0, -1.0, 1.0)
    assert preserved.vertices[4] == (1.0, -1.0, 1.0)


def test_section_loft_aligns_cyclic_sections_and_preserves_open_boundaries() -> None:
    lower = LoftSection(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)))
    upper = LoftSection(((1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (-1.0, -1.0, 1.0)))

    solid = section_loft(SectionLoftSpec((lower, upper)))
    open_loft = section_loft(SectionLoftSpec((lower, upper), cap=False))

    assert solid.to_trimesh().volume == pytest.approx(4.0)
    assert solid.is_watertight
    assert not open_loft.is_watertight


def test_shell_partition_and_face_opening_helpers() -> None:
    spec = ShellPartitionSpec(
        shell=BoxGeometry((0.2, 0.3, 0.1)),
        regions=(
            ShellPartitionRegion("left", side="left", y_max=0.0),
            ShellPartitionRegion("right", side="right", y_max=0.0),
        ),
        remainder_name="body",
        center_gap=0.01,
    )
    parts = partition_shell(spec)
    shell = BoxGeometry((0.2, 0.3, 0.1))
    original_faces = len(shell.faces)
    cut_opening_on_face(
        shell,
        face="+z",
        opening_profile=rounded_rect_profile(0.05, 0.03, 0.005),
        depth=0.02,
    )

    assert set(parts) == {"left", "right", "body"}
    assert parts["left"].bounds[1][0] <= -0.005 + 1e-6
    assert parts["right"].bounds[0][0] >= 0.005 - 1e-6
    assert len(shell.faces) > original_faces
    with pytest.raises(ValidationError, match="unique"):
        ShellPartitionSpec(
            shell=BoxGeometry((1.0, 1.0, 1.0)),
            regions=(ShellPartitionRegion("same"), ShellPartitionRegion("same")),
        )
    with pytest.raises(ValidationError, match="closed MeshGeometry solid"):
        ShellPartitionSpec(
            shell=CylinderGeometry(1.0, 1.0, closed=False),
            regions=(ShellPartitionRegion("body"),),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": -0.1},
        {"radius": 0.1, "corner_radius": -0.1},
        {"radius": 0.1, "corner_segments": 1},
        {"radius": 0.1, "min_segment_length": 0.0},
        {"radius": 0.1, "up_hint": (math.nan, 0.0, 1.0)},
    ],
)
def test_wire_polyline_rejects_invalid_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        WirePolylineGeometry([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)], **kwargs)
