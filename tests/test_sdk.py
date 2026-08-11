from __future__ import annotations

import math

import pytest
from build123d import Box, Pos
from build123d.topology import Shape

import articraft.sdk as sdk
from articraft.sdk import (
    JointAxis,
    JointDOF,
    JointFrame,
    MeshGeometry,
    RigidBody,
    RigidBodyAssembly,
    ValidationError,
)


def box() -> Shape:
    return Box(0.1, 0.1, 0.1)


def add_box(model: RigidBodyAssembly, name: str) -> RigidBody:
    part = model.rigid_body(name)
    part.add(box(), name="body")
    return part


def tetrahedron() -> MeshGeometry:
    return MeshGeometry(
        vertices=[
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.0, 0.1, 0.0),
            (0.0, 0.0, 0.1),
        ],
        faces=[(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
    )


def test_object_uses_meters_without_a_units_option() -> None:
    model = RigidBodyAssembly("meter_model")

    assert model.meters_per_unit == 1.0
    with pytest.raises(TypeError):
        RigidBodyAssembly("old_units", units="millimeters")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        RigidBodyAssembly("injected", parts=[])  # type: ignore[call-arg]


def test_part_accepts_multiple_named_shapes_and_preserves_local_placement() -> None:
    model = RigidBodyAssembly("mixer")
    body = model.rigid_body("body")
    shell = Pos(0.5, 0.0, 0.0) * Box(0.2, 0.1, 0.1)
    trim = Box(0.05, 0.1, 0.1)

    assert body.add(shell, name="shell", color=(0.7, 0.1, 0.1)) is shell
    body.add(trim, name="trim", color=(0.8, 0.8, 0.8, 0.5))

    stored_shell = body.get_shape("shell")
    assert stored_shell is shell
    assert isinstance(stored_shell, Shape)
    assert pytest.approx(0.4) == stored_shell.bounding_box().min.X
    entries = list(body._iter_shapes())
    assert [entry.name for entry in entries] == ["shell", "trim"]
    assert entries[0].color == (0.7, 0.1, 0.1, 1.0)
    assert entries[1].color == (0.8, 0.8, 0.8, 0.5)


def test_part_accepts_mesh_geometry() -> None:
    model = RigidBodyAssembly("mesh_model")
    body = model.rigid_body("body")
    mesh = tetrahedron()

    body.add(mesh, name="procedural")

    assert body.get_shape("procedural") is mesh
    model.validate()


def test_shape_names_are_required_and_unique_within_each_part() -> None:
    model = RigidBodyAssembly("names")
    left = model.rigid_body("left")
    right = model.rigid_body("right")
    left.add(box(), name="body")
    right.add(box(), name="body")

    with pytest.raises(TypeError):
        left.add(box())  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match=r"shape name.*non-empty"):
        left.add(box(), name="  ")
    with pytest.raises(ValidationError, match="duplicate shape name"):
        left.add(box(), name="body")
    with pytest.raises(ValidationError, match="unknown shape"):
        left.get_shape("missing")


@pytest.mark.parametrize(
    "color, message",
    [
        ((0.1, 0.2), "3 or 4"),
        ((0.1, 0.2, math.nan), "finite"),
        ((1.1, 0.2, 0.3), "between 0.0 and 1.0"),
    ],
)
def test_shape_colors_are_validated(color: tuple[float, ...], message: str) -> None:
    part = RigidBodyAssembly("color").rigid_body("body")

    with pytest.raises(ValidationError, match=message):
        part.add(box(), name="painted", color=color)


def test_parts_and_geometry_must_be_nonempty() -> None:
    model = RigidBodyAssembly("empty_part")
    part = model.rigid_body("body")

    with pytest.raises(ValidationError, match="at least one shape"):
        model.validate()
    with pytest.raises(ValidationError, match="non-empty"):
        part.add(Shape(), name="empty")
    with pytest.raises(ValidationError, match="build123d Shape or MeshGeometry"):
        part.add(object(), name="wrong")  # type: ignore[arg-type]


def test_mesh_edits_are_revalidated_with_the_model() -> None:
    model = RigidBodyAssembly("edited_mesh")
    part = model.rigid_body("body")
    mesh = tetrahedron()
    part.add(mesh, name="mesh")
    mesh.vertices[0] = (math.nan, 0.0, 0.0)

    with pytest.raises(ValidationError, match="finite"):
        model.validate()


def test_public_joint_grammar_covers_fixed_hinge_free_and_slide() -> None:
    model = RigidBodyAssembly("mechanism")
    root = add_box(model, "root")
    fixed = add_box(model, "fixed")
    hinge = add_box(model, "hinge")
    rotor = add_box(model, "rotor")
    slider = add_box(model, "slider")

    model.joint(
        "root_to_fixed",
        body0=root,
        frame0=JointFrame(),
        body1=fixed,
        frame1=JointFrame(),
    )
    revolute = model.joint(
        "fixed_to_hinge",
        body0=fixed,
        frame0=JointFrame(xyz=(0.0, 0.0, 0.2)),
        body1=hinge,
        frame1=JointFrame(),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=(-math.pi / 4.0, math.pi / 4.0)),),
    )
    model.joint(
        "hinge_to_rotor",
        body0=hinge,
        frame0=JointFrame(),
        body1=rotor,
        frame1=JointFrame(),
        dofs=(JointDOF(JointAxis.ROT_Z),),
    )
    model.joint(
        "rotor_to_slider",
        body0=rotor,
        frame0=JointFrame(),
        body1=slider,
        frame1=JointFrame(),
        dofs=(JointDOF(JointAxis.TRANS_X, limits=(-0.02, 0.2)),),
    )

    model.articulation(
        "main",
        root=root,
        joints=["root_to_fixed", "fixed_to_hinge", "hinge_to_rotor", "rotor_to_slider"],
    )
    model.validate()

    assert revolute.frame0.xyz == (0.0, 0.0, 0.2)
    assert revolute.dofs[0].limits == (-math.pi / 4.0, math.pi / 4.0)
    assert model.get_joint("fixed_to_hinge") is revolute
    assert model.get_joint("root_to_fixed").is_fixed
    assert model.get_joint("hinge_to_rotor").dofs[0].limits is None
    assert model.get_joint("rotor_to_slider").is_prismatic


def test_joint_frame_and_dof_values_must_be_usable() -> None:
    with pytest.raises(ValidationError, match="3 numeric values"):
        JointFrame(xyz="123")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="finite"):
        JointFrame(xyz=(math.inf, 0.0, 0.0))
    with pytest.raises(ValidationError, match="finite"):
        JointDOF(JointAxis.ROT_Z, limits=(0.0, math.nan))
    with pytest.raises(ValidationError, match="cannot exceed"):
        JointDOF(JointAxis.ROT_Z, limits=(1.0, -1.0))
    with pytest.raises(ValidationError, match="unknown joint axis"):
        JointDOF("spin")  # type: ignore[arg-type]


def test_the_graph_model_rules_are_validated() -> None:
    model = RigidBodyAssembly("invalid_motion")
    root = add_box(model, "root")
    child = add_box(model, "child")

    with pytest.raises(ValidationError, match="must contain the zero configuration"):
        JointDOF(JointAxis.ROT_Z, limits=(0.5, 1.0))
    with pytest.raises(ValidationError, match="duplicate DOF axes"):
        model.joint(
            "twice",
            body0=root,
            frame0=JointFrame(),
            body1=child,
            frame1=JointFrame(),
            dofs=(JointDOF(JointAxis.ROT_Z), JointDOF(JointAxis.ROT_Z)),
        )
    with pytest.raises(ValidationError, match="endpoints cannot be the same"):
        model.joint(
            "itself",
            body0=root,
            frame0=JointFrame(),
            body1=root,
            frame1=JointFrame(),
        )


def test_the_graph_must_be_one_connected_whole() -> None:
    disconnected = RigidBodyAssembly("two_islands")
    add_box(disconnected, "left")
    add_box(disconnected, "right")
    with pytest.raises(ValidationError, match="one connected joint graph"):
        disconnected.validate()


def test_a_second_joint_into_a_body_closes_a_loop() -> None:
    """Two joints reaching one body is a ring, not an error."""

    model = RigidBodyAssembly("rooted_loop")
    base = add_box(model, "base")
    upper = add_box(model, "upper")
    brace = add_box(model, "brace")
    model.joint("base_upper", body0=base, frame0=JointFrame(), body1=upper, frame1=JointFrame())
    model.joint("base_brace", body0=base, frame0=JointFrame(), body1=brace, frame1=JointFrame())
    model.joint("upper_brace", body0=upper, frame0=JointFrame(), body1=brace, frame1=JointFrame())
    model.articulation("main", root=base, joints=["base_upper", "base_brace"])

    resolved = model.resolve()

    assert resolved.has_closed_loops
    excluded = [item.joint.name for item in resolved.joints if item.exclude_from_articulation]
    assert excluded == ["upper_brace"]


def test_an_articulation_must_select_a_tree() -> None:
    model = RigidBodyAssembly("not_a_tree")
    base = add_box(model, "base")
    upper = add_box(model, "upper")
    brace = add_box(model, "brace")
    model.joint("base_upper", body0=base, frame0=JointFrame(), body1=upper, frame1=JointFrame())
    model.joint("base_brace", body0=base, frame0=JointFrame(), body1=brace, frame1=JointFrame())
    model.joint("upper_brace", body0=upper, frame0=JointFrame(), body1=brace, frame1=JointFrame())
    # all three edges is the ring itself, not a spanning tree
    model.articulation("main", root=base, joints=["base_upper", "base_brace", "upper_brace"])

    with pytest.raises(ValidationError, match="must form a tree"):
        model.validate()


def test_duplicate_and_unknown_names_are_rejected() -> None:
    model = RigidBodyAssembly("names")
    root = add_box(model, "root")
    child = add_box(model, "child")

    with pytest.raises(ValidationError, match="duplicate rigid body name"):
        model.rigid_body("root")
    model.joint("connection", body0=root, frame0=JointFrame(), body1=child, frame1=JointFrame())
    with pytest.raises(ValidationError, match="duplicate joint name"):
        model.joint("connection", body0=root, frame0=JointFrame(), body1=child, frame1=JointFrame())
    with pytest.raises(ValidationError, match="unknown"):
        model.joint("missing", body0=root, frame0=JointFrame(), body1="absent", frame1=JointFrame())


def test_old_frame_and_joint_helpers_are_not_public() -> None:
    model = RigidBodyAssembly("new_api")

    assert not hasattr(sdk, "Frame")
    assert not hasattr(model, "fixed")
    assert not hasattr(model, "revolute")


def test_a_redundant_loop_still_poses_from_joint_values() -> None:
    """A bail handle hangs on two coaxial pivots, which is a ring carrying one DOF.

    The spanning tree decides the pose and the second pivot agrees with it, so
    refusing to pose every ring outright would rule out the commonest hinge in
    everyday objects.
    """

    model = RigidBodyAssembly("kettle")
    body = add_box(model, "body")
    handle = add_box(model, "handle")
    swing = (JointDOF(JointAxis.ROT_X, limits=(-1.2, 1.2)),)
    model.joint(
        "left_pivot",
        body0=body,
        frame0=JointFrame(xyz=(-0.1, 0.0, 0.1)),
        body1=handle,
        frame1=JointFrame(xyz=(-0.1, 0.0, 0.0)),
        dofs=swing,
    )
    model.joint(
        "right_pivot",
        body0=body,
        frame0=JointFrame(xyz=(0.1, 0.0, 0.1)),
        body1=handle,
        frame1=JointFrame(xyz=(0.1, 0.0, 0.0)),
        dofs=swing,
    )
    model.articulation("main", root=body, joints=["left_pivot"])
    resolved = model.resolve()

    assert resolved.has_closed_loops
    state = resolved.forward_kinematics({"left_pivot.rotX": 0.6})

    assert state.dof_positions["left_pivot.rotX"] == pytest.approx(0.6)
    # The redundant pivot is carried along rather than fought.
    assert state.dof_positions["right_pivot.rotX"] == pytest.approx(0.6)


def test_a_ring_the_tree_cannot_satisfy_names_the_joint_that_broke() -> None:
    model = RigidBodyAssembly("four_bar")
    ground = add_box(model, "ground")
    left = add_box(model, "left")
    coupler = add_box(model, "coupler")
    swing = (JointDOF(JointAxis.ROT_Y, limits=(-0.6, 0.6)),)
    model.joint(
        "ground_left",
        body0=ground,
        frame0=JointFrame(),
        body1=left,
        frame1=JointFrame(),
        dofs=swing,
    )
    model.joint(
        "left_coupler",
        body0=left,
        frame0=JointFrame(xyz=(0.0, 0.0, 0.08)),
        body1=coupler,
        frame1=JointFrame(),
        dofs=swing,
    )
    model.joint(
        "ground_coupler",
        # Coincident at rest: the coupler origin rides 0.08 up on the crank.
        body0=ground,
        frame0=JointFrame(xyz=(0.12, 0.0, 0.08)),
        body1=coupler,
        frame1=JointFrame(xyz=(0.12, 0.0, 0.0)),
        dofs=swing,
    )
    model.articulation("main", root=ground, joints=["ground_left", "left_coupler"])
    resolved = model.resolve()

    with pytest.raises(ValidationError, match="ground_coupler"):
        resolved.forward_kinematics({"ground_left.rotY": 0.4})
