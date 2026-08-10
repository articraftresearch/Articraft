from __future__ import annotations

import math

import pytest
from build123d import Box, Pos
from build123d.topology import Shape

import mini_articraft.sdk as sdk
from mini_articraft.sdk import (
    RigidBodyAssembly,
    JointAxis,
    JointDOF,
    JointFrame,
    MeshGeometry,
    RigidBody,
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


def test_public_articulation_grammar_supports_all_four_types() -> None:
    model = RigidBodyAssembly("mechanism")
    root = add_box(model, "root")
    fixed = add_box(model, "fixed")
    hinge = add_box(model, "hinge")
    rotor = add_box(model, "rotor")
    slider = add_box(model, "slider")

    model.articulation("root_to_fixed", .FIXED, root, fixed)
    revolute = model.joint(
     "fixed_to_hinge",
     body0=fixed,
     frame0=JointFrame(),
     body1=hinge,
     frame1=JointFrame(),
     dofs=(JointDOF(JointAxis.ROT_Y, limits=(-math.pi / 4.0, math.pi / 4.0)),),
 )
    model.articulation(
        "hinge_to_rotor",
        "continuous",
        hinge,
        rotor,
        motion_limits=MotionLimits(effort=2.0, velocity=3.0),
    )
    model.joint(
        "rotor_to_slider",
        body0=rotor,
        frame0=JointFrame(),
        body1=slider,
        frame1=JointFrame(),
        dofs=(JointDOF(JointAxis.TRANS_X, limits=(-0.02, 0.2)),),
    )

    model.validate()
    assert revolute.origin.xyz == (0.0, 0.0, 0.2)
    assert revolute.motion_limits == MotionLimits(lower=-math.pi / 4.0, upper=math.pi / 4.0)
    assert model.get_articulation("fixed_to_hinge") is revolute
    assert not hasattr(model, "joints")


def test_articulation_values_must_be_finite() -> None:
    with pytest.raises(ValidationError, match="3 numeric values"):
        Origin(xyz="123")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="finite"):
        Origin(xyz=(math.inf, 0.0, 0.0))
    with pytest.raises(ValidationError, match="finite"):
        MotionLimits(velocity=math.nan)
    with pytest.raises(ValidationError, match="positive"):
        MotionLimits(effort=0.0)
    with pytest.raises(ValidationError, match="cannot exceed"):
        MotionLimits(lower=1.0, upper=-1.0)


def test_articulation_type_specific_rules_are_validated() -> None:
    model = RigidBodyAssembly("invalid_motion")
    root = add_box(model, "root")
    child = add_box(model, "child")

    with pytest.raises(ValidationError, match="must include motion_limits"):
        model.articulation("hinge", .REVOLUTE, root, child)
    with pytest.raises(ValidationError, match="requires lower and upper"):
        model.joint(
            "hinge",
            body0=root,
            frame0=JointFrame(),
            body1=child,
            frame1=JointFrame(),
            dofs=(JointDOF(JointAxis.ROT_Z),),
        )
    with pytest.raises(ValidationError, match="axis must be non-zero"):
        model.articulation(
            "hinge",
            .REVOLUTE,
            root,
            child,
            axis=(0.0, 0.0, 0.0),
            motion_limits=MotionLimits(lower=0.0, upper=1.0),
        )
    with pytest.raises(ValidationError, match=r"continuous.*cannot include"):
        model.joint(
            "rotor",
            body0=root,
            frame0=JointFrame(),
            body1=child,
            frame1=JointFrame(),
            dofs=(JointDOF(JointAxis.ROT_Z, limits=(0.0, 1.0)),),
        )
    with pytest.raises(ValidationError, match=r"fixed.*cannot include"):
        model.joint(
            "fixed",
            body0=root,
            frame0=JointFrame(),
            body1=child,
            frame1=JointFrame(),
            dofs=(),
        )


def test_articulation_tree_requires_one_root_and_one_parent_per_child() -> None:
    disconnected = RigidBodyAssembly("two_roots")
    add_box(disconnected, "left")
    add_box(disconnected, "right")
    with pytest.raises(ValidationError, match="exactly one root"):
        disconnected.validate()

    duplicate_parent = RigidBodyAssembly("duplicate_parent")
    left = add_box(duplicate_parent, "left")
    right = add_box(duplicate_parent, "right")
    child = add_box(duplicate_parent, "child")
    duplicate_parent.articulation("left_child", "fixed", left, child)
    duplicate_parent.articulation("right_child", "fixed", right, child)
    with pytest.raises(ValidationError, match="multiple parent articulations"):
        duplicate_parent.validate()


def test_duplicate_and_unknown_names_are_rejected() -> None:
    model = RigidBodyAssembly("names")
    root = add_box(model, "root")
    child = add_box(model, "child")

    with pytest.raises(ValidationError, match="duplicate part name"):
        model.rigid_body("root")
    model.articulation("connection", "fixed", root, child)
    with pytest.raises(ValidationError, match="duplicate articulation name"):
        model.articulation("connection", "fixed", root, child)
    with pytest.raises(ValidationError, match="unknown part"):
        model.articulation("missing", "fixed", root, "missing")


def test_old_frame_and_joint_helpers_are_not_public() -> None:
    model = RigidBodyAssembly("new_api")

    assert not hasattr(sdk, "Frame")
    assert not hasattr(model, "fixed")
    assert not hasattr(model, "revolute")
