from __future__ import annotations

import warnings

import pytest
from build123d import Box, Pos

from articraft.errors import ValidationError
from articraft.sdk import (
    AllowedMeshIssues,
    AllowedOverlap,
    BoxGeometry,
    FailureKind,
    JointAxis,
    JointDOF,
    JointFrame,
    MeshGeometry,
    MeshHealthIssue,
    RigidBodyAssembly,
    SphereGeometry,
    TestContext,
)


def add_box(part, name: str, *, size: float = 1.0, x: float = 0.0) -> None:
    part.add(Pos(X=x) * Box(size, size, size), name=name)


def fixed(model: RigidBodyAssembly, name: str, parent, child, xyz=(0.0, 0.0, 0.0)):
    """A joint with no free axes is a fixed joint."""
    return model.joint(name, parent.at(JointFrame(xyz=xyz)), child.at(JointFrame()))


def test_report_records_warnings_and_shape_scoped_allowances() -> None:
    model = RigidBodyAssembly("report")
    base = model.rigid_body("base")
    add_box(base, "outer")
    insert = model.rigid_body("insert")
    add_box(insert, "inner", size=0.25)
    ctx = TestContext(model)

    ctx.check("custom pass", True)
    ctx.warn("nonblocking note")
    ctx.allow_isolated_part("base", reason="display stand")
    ctx.allow_overlap(
        "base",
        "insert",
        shape_a=" outer ",
        shape_b=" inner ",
        reason="nested shape",
    )

    report = ctx.report()
    assert report.passed
    assert report.warnings == ("nonblocking note",)
    assert report.allowed_overlaps == (
        AllowedOverlap("base", "insert", "nested shape", "outer", "inner"),
    )

    with pytest.raises(TypeError, match="shape_a"):
        ctx.allow_overlap(  # pyright: ignore[reportCallIssue]
            "base", "insert", reason="too broad"
        )


def test_mesh_health_allowance_is_exact_and_requires_a_reason() -> None:
    model = RigidBodyAssembly("mesh_health")
    base = model.rigid_body("base")
    box = BoxGeometry((1.0, 1.0, 1.0))
    open_box = MeshGeometry(box.vertices, box.faces[:-1])
    base.add(open_box, name="intentional-sheet")

    blocked = TestContext(model)
    assert not blocked.fail_if_mesh_unhealthy()
    assert blocked.report().failures[0].kind is FailureKind.MESH_HEALTH

    allowed = TestContext(model)
    allowed.allow_mesh_issues(
        "base",
        shape="intentional-sheet",
        issues=(MeshHealthIssue.BOUNDARY_EDGES,),
        reason="This named mesh is an intentional open surface.",
    )
    assert allowed.fail_if_mesh_unhealthy()
    assert allowed.report().allowed_mesh_issues == (
        AllowedMeshIssues(
            "base",
            "intentional-sheet",
            (MeshHealthIssue.BOUNDARY_EDGES,),
            "This named mesh is an intentional open surface.",
        ),
    )

    wrong_issue = TestContext(model)
    wrong_issue.allow_mesh_issues(
        "base",
        shape="intentional-sheet",
        issues=(MeshHealthIssue.SLIVER_FACES,),
        reason="Only slivers are intentional.",
    )
    assert not wrong_issue.fail_if_mesh_unhealthy()

    with pytest.raises(ValueError, match="non-empty reason"):
        allowed.allow_mesh_issues(
            "base",
            shape="intentional-sheet",
            issues=(MeshHealthIssue.BOUNDARY_EDGES,),
            reason="",
        )


def test_named_shape_queries_and_world_bounds() -> None:
    model = RigidBodyAssembly("queries")
    root = model.rigid_body("root")
    add_box(root, "left", x=-1.0)
    add_box(root, "right", x=1.0)
    ctx = TestContext(model)

    assert ctx.shape_world_bounds("root", "left") == ((-1.5, -0.5, -0.5), (-0.5, 0.5, 0.5))
    assert ctx.part_world_bounds("root") == ((-1.5, -0.5, -0.5), (1.5, 0.5, 0.5))
    distance = ctx.distance_between("root", "root", shape_a="left", shape_b="right")
    assert distance.distance == pytest.approx(1.0)

    with pytest.raises(ValidationError, match="unknown shape"):
        ctx.shape_world_bounds("root", "missing")


def test_complete_physics_state_is_authoritative_for_geometry_queries() -> None:
    model = RigidBodyAssembly("state")
    body = model.rigid_body("body")
    add_box(body, "shape")
    state = model.physics_state({body: JointFrame(xyz=(2.0, 0.0, 0.0))})
    context = TestContext(model)

    with context.state(state):
        assert context.part_world_bounds(body) == ((1.5, -0.5, -0.5), (2.5, 0.5, 0.5))

    assert context.part_world_bounds(body) == ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))


def test_world_bounds_transform_large_mesh_without_runtime_warnings() -> None:
    model = RigidBodyAssembly("large_mesh_transform")
    root = model.rigid_body("root")
    sphere = SphereGeometry(1.0, width_segments=32, height_segments=16)
    assert len(sphere.vertices) >= 512
    root.add(sphere, name="sphere")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        bounds = TestContext(model).part_world_bounds("root")

    assert bounds[0][2] == pytest.approx(-1.0)
    assert bounds[1][2] == pytest.approx(1.0)


def test_build123d_shape_bounds_refresh_after_location_mutation() -> None:
    model = RigidBodyAssembly("mutable_shape")
    root = model.rigid_body("root")
    shape = Box(1.0, 1.0, 1.0)
    root.add(shape, name="body")
    ctx = TestContext(model)

    assert ctx.shape_world_bounds("root", "body")[0][0] == pytest.approx(-0.5)
    shape.locate(Pos(X=2.0))

    assert ctx.shape_world_bounds("root", "body")[0][0] == pytest.approx(1.5)


def test_mesh_collision_cache_refreshes_after_geometry_mutation() -> None:
    model = RigidBodyAssembly("mutable_mesh")
    root = model.rigid_body("root")
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    root.add(geometry, name="body")
    context = TestContext(model)

    assert context.shape_world_bounds("root", "body")[0][0] == pytest.approx(-0.5)
    geometry.translate(2.0, 0.0, 0.0)

    assert context.shape_world_bounds("root", "body")[0][0] == pytest.approx(1.5)


def test_exact_checks_target_named_shapes() -> None:
    model = RigidBodyAssembly("exact")
    root = model.rigid_body("root")
    add_box(root, "outer", size=3.0)
    add_box(root, "left", size=0.5, x=-1.0)
    add_box(root, "left_copy", size=0.5, x=-1.0)
    add_box(root, "right", size=0.5, x=1.0)
    ctx = TestContext(model)

    assert ctx.expect_within("root", "root", inner_shape="left", outer_shape="outer", axes="xyz")
    assert ctx.expect_distance("root", "root", shape_a="left", shape_b="right", min_distance=1.49)
    assert ctx.expect_no_collision("root", "root", shape_a="left", shape_b="right")
    assert ctx.expect_collision("root", "root", shape_a="left", shape_b="left_copy")
    assert ctx.report().passed


def test_shape_scoped_projection_checks_have_distinct_default_names() -> None:
    model = RigidBodyAssembly("named_checks")
    root = model.rigid_body("root")
    add_box(root, "left", size=0.5, x=-1.0)
    add_box(root, "center", size=0.5, x=0.0)
    add_box(root, "right", size=0.5, x=1.0)
    ctx = TestContext(model)

    assert not ctx.expect_gap(
        "root",
        "root",
        axis="x",
        positive_shape="left",
        negative_shape="center",
        min_gap=0.1,
    )
    assert not ctx.expect_gap(
        "root",
        "root",
        axis="x",
        positive_shape="center",
        negative_shape="right",
        min_gap=0.1,
    )

    names = [failure.name for failure in ctx.report().failures]
    assert len(set(names)) == 2
    assert "positive_shape=left" in names[0]
    assert "positive_shape=center" in names[1]


def test_pose_changes_prismatic_part_transform_and_restores() -> None:
    model = RigidBodyAssembly("pose")
    base = model.rigid_body("base")
    add_box(base, "body")
    slider = model.rigid_body("slider")
    # Clear of the base at rest, so sliding back is what brings them together.
    add_box(slider, "body", x=1.5)
    model.joint(
        "slide",
        base.at(JointFrame()),
        slider.at(JointFrame()),
        dofs=(JointDOF(JointAxis.TRANS_X, limits=(-1.5, 0.0)),),
    )
    ctx = TestContext(model)

    rest = ctx.part_world_position("slider")
    assert not ctx.expect_collision("base", "slider", shape_a="body", shape_b="body")
    with ctx.pose({"slide": -1.25}):
        posed = ctx.part_world_position("slider")
        assert ctx.expect_collision("base", "slider", shape_a="body", shape_b="body")
    assert ctx.part_world_position("slider") == rest
    assert posed[0] < rest[0]


def test_pose_rejects_an_unknown_joint_position() -> None:
    model = RigidBodyAssembly("pose")
    base = model.rigid_body("base")
    add_box(base, "body")
    with (
        pytest.raises(ValidationError, match="unknown joint position"),
        TestContext(model).pose(missing=1.0),
    ):
        pass


def test_pose_accepts_qualified_d6_coordinates() -> None:
    model = RigidBodyAssembly("d6_pose")
    base = model.rigid_body("base")
    add_box(base, "body")
    moving = model.rigid_body("moving")
    add_box(moving, "body")
    model.joint(
        "motion",
        base.at(),
        moving.at(),
        dofs=(
            JointDOF(JointAxis.TRANS_X, limits=(-0.5, 0.5)),
            JointDOF(JointAxis.ROT_Z, limits=(-0.5, 0.5)),
        ),
    )
    model.articulation("main", root=base)
    context = TestContext(model)

    with context.pose({"motion.transX": 0.2, "motion.rotZ": 0.1}):
        assert context.part_world_position(moving)[0] == pytest.approx(0.2)

    # A multi-DOF joint has no single value; the error must teach the
    # qualified spelling.
    with (
        pytest.raises(ValidationError, match=r"motion\.rotZ"),
        context.pose({"motion": 0.2}),
    ):
        pass


def test_scoped_allowance_does_not_hide_another_shape_pair() -> None:
    model = RigidBodyAssembly("allowance")
    parent = model.rigid_body("parent")
    add_box(parent, "allowed_parent", x=-1.0)
    add_box(parent, "blocked_parent", x=1.0)
    child = model.rigid_body("child")
    add_box(child, "allowed_child", x=-1.0)
    add_box(child, "blocked_child", x=1.0)
    fixed(model, "mount", parent, child)
    ctx = TestContext(model)
    ctx.allow_overlap(
        parent,
        child,
        shape_a="allowed_parent",
        shape_b="allowed_child",
        reason="captured insert",
    )

    assert not ctx.fail_if_parts_overlap_in_current_pose(overlap_tol=0.001)
    failure = ctx.report().failures[0]
    assert "blocked_parent" in failure.details
    assert "blocked_child" in failure.details


def test_adjacent_contact_and_tiny_penetration_pass_physical_thresholds() -> None:
    for index, offset in enumerate((1.0, 0.996)):
        model = RigidBodyAssembly(f"contact_{index}")
        parent = model.rigid_body("parent")
        add_box(parent, "body")
        child = model.rigid_body("child")
        add_box(child, "body")
        fixed(model, "mount", parent, child, xyz=(0.0, 0.0, offset))

        assert TestContext(model).fail_if_parts_overlap_in_current_pose()


def test_coplanar_contact_with_large_watertight_bounds_passes() -> None:
    model = RigidBodyAssembly("watertight_contact")
    parent = model.rigid_body("parent")
    parent.add(Box(1, 1, 1) + Pos(X=2, Z=2) * Box(1, 1, 1), name="body")
    child = model.rigid_body("child")
    child.add(Pos(Z=1) * Box(1, 1, 1), name="body")
    fixed(model, "mount", parent, child)

    assert TestContext(model).fail_if_parts_overlap_in_current_pose()


def test_coplanar_contact_with_large_open_mesh_bounds_passes() -> None:
    model = RigidBodyAssembly("open_mesh_contact")
    parent = model.rigid_body("parent")
    parent.add(
        MeshGeometry(
            vertices=[
                (-0.5, -0.5, 0.5),
                (0.5, -0.5, 0.5),
                (0.5, 0.5, 0.5),
                (-0.5, 0.5, 0.5),
                (1.5, -0.5, 2.5),
                (2.5, -0.5, 2.5),
                (2.0, 0.5, 2.5),
            ],
            faces=[(0, 1, 2), (0, 2, 3), (4, 5, 6)],
        ),
        name="body",
    )
    child = model.rigid_body("child")
    child.add(Pos(Z=1) * Box(1, 1, 1), name="body")
    fixed(model, "mount", parent, child)

    assert TestContext(model).fail_if_parts_overlap_in_current_pose()


def test_adjacent_large_penetration_blocks() -> None:
    model = RigidBodyAssembly("penetration")
    parent = model.rigid_body("parent")
    add_box(parent, "body")
    child = model.rigid_body("child")
    add_box(child, "body")
    fixed(model, "mount", parent, child, xyz=(0.0, 0.0, 0.98))

    ctx = TestContext(model)
    assert not ctx.fail_if_parts_overlap_in_current_pose()
    assert "shape_a='body'" in ctx.report().failures[0].details


def test_physical_isolation_ignores_the_articulation_graph() -> None:
    model = RigidBodyAssembly("isolated")
    base = model.rigid_body("base")
    add_box(base, "body")
    floating = model.rigid_body("floating")
    add_box(floating, "body")
    fixed(model, "mount", base, floating, xyz=(3.0, 0.0, 0.0))

    ctx = TestContext(model)
    assert not ctx.fail_if_isolated_parts()
    assert "floating_group=['floating']" in ctx.report().failures[0].details


def test_an_entire_floating_group_must_be_allowed() -> None:
    model = RigidBodyAssembly("floating_group")
    base = model.rigid_body("base")
    add_box(base, "body")
    first = model.rigid_body("first")
    add_box(first, "body")
    second = model.rigid_body("second")
    add_box(second, "body")
    fixed(model, "base_to_first", base, first, xyz=(3.0, 0.0, 0.0))
    fixed(model, "first_to_second", first, second, xyz=(1.0, 0.0, 0.0))

    partial = TestContext(model)
    partial.allow_isolated_part("first", reason="display group")
    assert not partial.fail_if_isolated_parts()
    assert "nearest_root_part='base' nearest_gap=2m" in partial.report().failures[0].details

    complete = TestContext(model)
    complete.allow_isolated_part("first", reason="display group")
    complete.allow_isolated_part("second", reason="display group")
    assert complete.fail_if_isolated_parts()
    assert complete.report().warnings


def test_disconnected_geometry_warns_by_default_and_can_be_authored_as_blocking() -> None:
    model = RigidBodyAssembly("disconnected")
    base = model.rigid_body("base")
    add_box(base, "left", x=-1.0)
    add_box(base, "right", x=1.0)

    warning_ctx = TestContext(model)
    assert warning_ctx.warn_if_part_contains_disconnected_geometry_islands()
    assert warning_ctx.report().passed
    assert "Disconnected geometry islands" in warning_ctx.report().warnings[0]

    blocking_ctx = TestContext(model)
    assert not blocking_ctx.fail_if_part_contains_disconnected_geometry_islands()
    assert not blocking_ctx.report().passed
    assert blocking_ctx.report().failures[0].kind is FailureKind.DISCONNECTED_GEOMETRY


def test_failing_checks_record_machine_readable_kinds() -> None:
    model = RigidBodyAssembly("kinds")
    base = model.rigid_body("base")
    add_box(base, "body")
    lid = model.rigid_body("lid")
    add_box(lid, "body")
    fixed(model, "mount", base, lid, xyz=(0.0, 0.0, 0.98))

    ctx = TestContext(model)
    assert not ctx.fail_if_parts_overlap_in_current_pose()
    assert not ctx.expect_no_collision("base", "lid")
    assert not ctx.check("custom_check", False, "authored detail")
    failures = ctx.report().failures
    assert [failure.kind for failure in failures] == [
        FailureKind.OVERLAP,
        FailureKind.OVERLAP,
        FailureKind.AUTHORED,
    ]


def test_contact_and_isolation_checks_record_kinds() -> None:
    model = RigidBodyAssembly("gap_kinds")
    base = model.rigid_body("base")
    add_box(base, "body")
    floating = model.rigid_body("floating")
    add_box(floating, "body")
    fixed(model, "mount", base, floating, xyz=(3.0, 0.0, 0.0))

    ctx = TestContext(model)
    assert not ctx.expect_contact("base", "floating")
    assert not ctx.fail_if_isolated_parts()
    failures = ctx.report().failures
    assert [failure.kind for failure in failures] == [
        FailureKind.CONTACT,
        FailureKind.ISOLATED_PART,
    ]


def test_nested_solid_shapes_are_connected_geometry() -> None:
    model = RigidBodyAssembly("nested")
    base = model.rigid_body("base")
    add_box(base, "outer", size=2.0)
    add_box(base, "insert", size=0.5)
    ctx = TestContext(model)

    assert ctx.warn_if_part_contains_disconnected_geometry_islands()
    assert ctx.report().warnings == ()


def test_absurd_dimensions_and_scale_outliers_are_warnings() -> None:
    model = RigidBodyAssembly("scale")
    base = model.rigid_body("base")
    add_box(base, "normal", size=1.0)
    add_box(base, "absurd", size=2001.0, x=3000.0)
    ctx = TestContext(model)

    assert ctx.warn_if_absurd_dimensions()
    assert ctx.report().passed
    assert "absurd dimension" in ctx.report().warnings[0]

    relative = RigidBodyAssembly("relative_scale")
    detailed = relative.rigid_body("body")
    add_box(detailed, "detail_a", size=0.001, x=-1.0)
    add_box(detailed, "detail_b", size=0.001, x=1.0)
    add_box(detailed, "body", size=0.2)
    relative_ctx = TestContext(relative)

    assert relative_ctx.warn_if_absurd_dimensions()
    assert "extreme scale outlier" in relative_ctx.report().warnings[0]


def _swing_arm() -> RigidBodyAssembly:
    model = RigidBodyAssembly("swing_arm")
    base = model.rigid_body("base")
    add_box(base, "body")
    arm = model.rigid_body("arm")
    add_box(arm, "body", x=3.0)
    model.joint(
        "hinge",
        base.at(),
        arm.at(),
        dofs=(JointDOF(JointAxis.ROT_Z, limits=(-1.0, 1.0)),),
    )
    model.articulation("main", root=base, joints=["hinge"])
    return model


def test_failed_pose_leaves_an_enclosing_state_intact() -> None:
    model = RigidBodyAssembly("state_guard")
    body = model.rigid_body("body")
    add_box(body, "shape")
    moved = model.physics_state({body: JointFrame(xyz=(2.0, 0.0, 0.0))})
    context = TestContext(model)

    with context.state(moved):
        with (
            pytest.raises(ValidationError, match="unknown joint position"),
            context.pose(bogus=1.0),
        ):
            pass
        assert context.part_world_position("body")[0] == pytest.approx(2.0)


def test_nested_pose_spellings_share_one_coordinate() -> None:
    """ "hinge" and "hinge.rotZ" are the same DOF; the innermost value wins."""

    import math as _math

    context = TestContext(_swing_arm())

    with context.pose(hinge=0.3), context.pose({"hinge.rotZ": 0.5}), context.pose(hinge=0.7):
        tip = context.part_world_point("arm", (3.0, 0.0, 0.0))
        assert tip[0] == pytest.approx(3.0 * _math.cos(0.7))


def test_pose_rejects_out_of_limit_values_at_entry() -> None:
    context = TestContext(_swing_arm())

    with pytest.raises(ValidationError, match="outside limits"), context.pose(hinge=5.0):
        pass


def test_pose_sweeps_record_unreachable_loop_poses() -> None:
    """A sweep past a ring's reachable range is a recorded failure, not a crash."""

    model = _swing_arm()
    base = model.get_rigid_body("base")
    arm = model.get_rigid_body("arm")
    model.joint(
        "second_pin",
        base.at(JointFrame(xyz=(1.0, 0.0, 0.0))),
        arm.at(JointFrame(xyz=(1.0, 0.0, 0.0))),
        dofs=(JointDOF(JointAxis.ROT_Z),),
    )
    context = TestContext(model)

    poses = context.sample_joint("hinge")
    assert not context.expect_no_collision_at_poses("base", "arm", poses=poses)

    failures = context.report().failures
    assert failures and "unreachable" in failures[0].details


def test_sample_dof_selects_one_axis_on_multi_dof_joint() -> None:
    model = RigidBodyAssembly("removable_lid")
    pot = model.rigid_body("pot")
    add_box(pot, "body")
    lid = model.rigid_body("lid")
    add_box(lid, "body")
    model.joint(
        "lid_free",
        pot.at(),
        lid.at(),
        dofs=tuple(JointDOF(axis) for axis in JointAxis),
    )
    model.articulation("main", root=pot, joints=("lid_free",))
    context = TestContext(model)

    poses = context.sample_dof(
        "lid_free",
        "lid_free.transZ",
        positions=(0.0, 0.04, 0.1),
    )

    assert [pose.as_dict() for pose in poses] == [
        {"lid_free.transZ": 0.0},
        {"lid_free.transZ": 0.04},
        {"lid_free.transZ": 0.1},
    ]
    assert poses[-1].label == "lid_free.transZ=0.1"
    with pytest.raises(ValidationError, match="belongs to joint"):
        context.sample_dof("lid_free", "other.transZ", positions=(0.0,))
    with pytest.raises(ValidationError, match="unknown joint axis"):
        context.sample_dof("lid_free", "scaleX", positions=(0.0,))


def test_separation_check_handles_multi_dof_joints() -> None:
    model = RigidBodyAssembly("multi_dof_baseline")
    base = model.rigid_body("base")
    add_box(base, "body")
    moving = model.rigid_body("moving")
    add_box(moving, "body")
    model.joint(
        "wrist",
        base.at(),
        moving.at(),
        dofs=(
            JointDOF(JointAxis.ROT_X, limits=(-0.4, 0.4)),
            JointDOF(JointAxis.ROT_Y, limits=(-0.4, 0.4)),
        ),
    )
    model.articulation("main", root=base, joints=["wrist"])
    context = TestContext(model)

    assert context.fail_if_articulation_separates_child()


def test_open_mesh_buried_in_a_solid_reads_as_collision() -> None:
    model = RigidBodyAssembly("buried")
    outer = model.rigid_body("outer")
    add_box(outer, "solid")
    inner = model.rigid_body("inner")
    sheet = BoxGeometry((0.2, 0.2, 0.2))
    inner.add(MeshGeometry(sheet.vertices, sheet.faces[:-1]), name="open_sheet")
    fixed(model, "mount", outer, inner)
    context = TestContext(model)

    assert context.expect_collision("outer", "inner")
    assert not context.expect_no_collision("outer", "inner")
