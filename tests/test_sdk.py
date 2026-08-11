from __future__ import annotations

import math

import pytest
from build123d import Box, Pos
from build123d.topology import Shape

import articraft.sdk as sdk
from articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    MeshGeometry,
    MotionLimits,
    Origin,
    Part,
    ValidationError,
)


def box() -> Shape:
    return Box(0.1, 0.1, 0.1)


def add_box(model: ArticulatedObject, name: str) -> Part:
    part = model.part(name)
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
    model = ArticulatedObject("meter_model")

    assert model.meters_per_unit == 1.0
    with pytest.raises(TypeError):
        ArticulatedObject("old_units", units="millimeters")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ArticulatedObject("injected", parts=[])  # type: ignore[call-arg]


def test_part_accepts_multiple_named_shapes_and_preserves_local_placement() -> None:
    model = ArticulatedObject("mixer")
    body = model.part("body")
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
    model = ArticulatedObject("mesh_model")
    body = model.part("body")
    mesh = tetrahedron()

    body.add(mesh, name="procedural")

    assert body.get_shape("procedural") is mesh
    model.validate()


def test_shape_names_are_required_and_unique_within_each_part() -> None:
    model = ArticulatedObject("names")
    left = model.part("left")
    right = model.part("right")
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
    part = ArticulatedObject("color").part("body")

    with pytest.raises(ValidationError, match=message):
        part.add(box(), name="painted", color=color)


def test_parts_and_geometry_must_be_nonempty() -> None:
    model = ArticulatedObject("empty_part")
    part = model.part("body")

    with pytest.raises(ValidationError, match="at least one shape"):
        model.validate()
    with pytest.raises(ValidationError, match="non-empty"):
        part.add(Shape(), name="empty")
    with pytest.raises(ValidationError, match="build123d Shape or MeshGeometry"):
        part.add(object(), name="wrong")  # type: ignore[arg-type]


def test_mesh_edits_are_revalidated_with_the_model() -> None:
    model = ArticulatedObject("edited_mesh")
    part = model.part("body")
    mesh = tetrahedron()
    part.add(mesh, name="mesh")
    mesh.vertices[0] = (math.nan, 0.0, 0.0)

    with pytest.raises(ValidationError, match="finite"):
        model.validate()


def test_public_articulation_grammar_supports_all_four_types() -> None:
    model = ArticulatedObject("mechanism")
    root = add_box(model, "root")
    fixed = add_box(model, "fixed")
    hinge = add_box(model, "hinge")
    rotor = add_box(model, "rotor")
    slider = add_box(model, "slider")

    model.articulation("root_to_fixed", ArticulationType.FIXED, root, fixed)
    revolute = model.articulation(
        "fixed_to_hinge",
        ArticulationType.REVOLUTE,
        fixed,
        hinge,
        origin=Origin(xyz=(0.0, 0.0, 0.2), rpy=(0.0, math.pi / 2.0, 0.0)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-math.pi / 4.0, upper=math.pi / 4.0),
    )
    model.articulation(
        "hinge_to_rotor",
        "continuous",
        hinge,
        rotor,
        motion_limits=MotionLimits(effort=2.0, velocity=3.0),
    )
    model.articulation(
        "rotor_to_slider",
        ArticulationType.PRISMATIC,
        rotor,
        slider,
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=-0.02, upper=0.2),
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
    model = ArticulatedObject("invalid_motion")
    root = add_box(model, "root")
    child = add_box(model, "child")

    with pytest.raises(ValidationError, match="must include motion_limits"):
        model.articulation("hinge", ArticulationType.REVOLUTE, root, child)
    with pytest.raises(ValidationError, match="requires lower and upper"):
        model.articulation(
            "hinge",
            ArticulationType.REVOLUTE,
            root,
            child,
            motion_limits=MotionLimits(),
        )
    with pytest.raises(ValidationError, match="axis must be non-zero"):
        model.articulation(
            "hinge",
            ArticulationType.REVOLUTE,
            root,
            child,
            axis=(0.0, 0.0, 0.0),
            motion_limits=MotionLimits(lower=0.0, upper=1.0),
        )
    with pytest.raises(ValidationError, match=r"continuous.*cannot include"):
        model.articulation(
            "rotor",
            ArticulationType.CONTINUOUS,
            root,
            child,
            motion_limits=MotionLimits(lower=0.0, upper=1.0),
        )
    with pytest.raises(ValidationError, match=r"fixed.*cannot include"):
        model.articulation(
            "fixed",
            ArticulationType.FIXED,
            root,
            child,
            motion_limits=MotionLimits(),
        )


def test_articulation_tree_requires_one_root() -> None:
    disconnected = ArticulatedObject("two_roots")
    add_box(disconnected, "left")
    add_box(disconnected, "right")
    with pytest.raises(ValidationError, match="exactly one root"):
        disconnected.validate()

    # A second parent articulation is a loop closure now, not an error, but the
    # spanning tree it leaves behind must still be rooted in exactly one part.
    two_rooted = ArticulatedObject("two_rooted_loop")
    left = add_box(two_rooted, "left")
    right = add_box(two_rooted, "right")
    child = add_box(two_rooted, "child")
    two_rooted.articulation("left_child", "fixed", left, child)
    two_rooted.articulation("right_child", "fixed", right, child)
    with pytest.raises(ValidationError, match="exactly one root"):
        two_rooted.validate()

    rooted = ArticulatedObject("rooted_loop")
    base = add_box(rooted, "base")
    upper = add_box(rooted, "upper")
    brace = add_box(rooted, "brace")
    rooted.articulation("base_upper", "fixed", base, upper)
    rooted.articulation("base_brace", "fixed", base, brace)
    rooted.articulation("upper_brace", "fixed", upper, brace)
    rooted.validate()


def test_duplicate_and_unknown_names_are_rejected() -> None:
    model = ArticulatedObject("names")
    root = add_box(model, "root")
    child = add_box(model, "child")

    with pytest.raises(ValidationError, match="duplicate part name"):
        model.part("root")
    model.articulation("connection", "fixed", root, child)
    with pytest.raises(ValidationError, match="duplicate articulation name"):
        model.articulation("connection", "fixed", root, child)
    with pytest.raises(ValidationError, match="unknown part"):
        model.articulation("missing", "fixed", root, "missing")


def test_old_frame_and_joint_helpers_are_not_public() -> None:
    model = ArticulatedObject("new_api")

    assert not hasattr(sdk, "Frame")
    assert not hasattr(model, "fixed")
    assert not hasattr(model, "revolute")
