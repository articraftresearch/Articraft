from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box

from articraft.sdk.assembly import (
    WORLD,
    Articulation,
    JointAxis,
    JointDOF,
    JointFrame,
    Mat4,
    PhysicsState,
    RigidBodyAssembly,
)
from articraft.sdk.bodies import RigidBody
from articraft.sdk.errors import ValidationError


def add_body(assembly: RigidBodyAssembly, name: str) -> RigidBody:
    body = assembly.rigid_body(name)
    body.add(Box(0.1, 0.1, 0.1), name="shape")
    return body


def four_bar(*, second_closure: bool = False) -> RigidBodyAssembly:
    assembly = RigidBodyAssembly("four_bar")
    ground = add_body(assembly, "ground")
    crank = add_body(assembly, "crank")
    coupler = add_body(assembly, "coupler")
    rocker = add_body(assembly, "rocker")
    revolute = (JointDOF(JointAxis.ROT_Z),)
    ground_crank = assembly.joint("ground_crank", body0=ground, body1=crank, dofs=revolute)
    crank_coupler = assembly.joint(
        "crank_coupler",
        body0=crank,
        frame0=JointFrame(xyz=(1.0, 0.0, 0.0)),
        body1=coupler,
        dofs=revolute,
    )
    coupler_rocker = assembly.joint(
        "coupler_rocker",
        body0=coupler,
        frame0=JointFrame(xyz=(1.0, 0.0, 0.0)),
        body1=rocker,
        dofs=revolute,
    )
    assembly.joint(
        "rocker_ground",
        body0=rocker,
        body1=ground,
        frame1=JointFrame(xyz=(2.0, 0.0, 0.0)),
        dofs=revolute,
    )
    if second_closure:
        assembly.joint(
            "crank_rocker",
            body0=crank,
            frame0=JointFrame(xyz=(2.0, 0.0, 0.0)),
            body1=rocker,
            dofs=revolute,
        )
    assembly.articulation(
        "main",
        root=ground,
        joints=(ground_crank, crank_coupler, coupler_rocker),
    )
    return assembly


def test_joint_dofs_are_usd_d6_axes_and_unlisted_axes_are_locked() -> None:
    assembly = RigidBodyAssembly("slider_hinge")
    base = add_body(assembly, "base")
    moving = add_body(assembly, "moving")
    joint = assembly.joint(
        "motion",
        body0=base,
        body1=moving,
        dofs=(
            JointDOF("rotZ", limits=(-1.0, 1.0)),
            JointDOF(JointAxis.TRANS_X, limits=(-0.2, 0.3)),
        ),
    )
    assembly.articulation("main", root=base)

    resolved = assembly.resolve()

    assert [dof.axis for dof in joint.dofs] == [JointAxis.TRANS_X, JointAxis.ROT_Z]
    assert not resolved.has_closed_loops
    posed = resolved.forward_kinematics({"motion.transX": 0.1, "motion.rotZ": 0.5})
    assert posed.dof_positions["motion.transX"] == pytest.approx(0.1)
    assert posed.dof_positions["motion.rotZ"] == pytest.approx(0.5)


def test_joint_limits_must_include_the_frame_defined_zero_configuration() -> None:
    with pytest.raises(ValidationError, match="contain the zero configuration"):
        JointDOF(JointAxis.ROT_X, limits=(0.1, 1.0))
    with pytest.raises(ValidationError, match="duplicate DOF axes"):
        assembly = RigidBodyAssembly("duplicate")
        a = add_body(assembly, "a")
        b = add_body(assembly, "b")
        assembly.joint(
            "joint",
            body0=a,
            body1=b,
            dofs=(JointDOF("rotX"), JointDOF("rotX")),
        )


def test_ordinary_tree_articulation_is_inferred_and_supports_forward_kinematics() -> None:
    assembly = RigidBodyAssembly("arm")
    base = add_body(assembly, "base")
    link = add_body(assembly, "link")
    tip = add_body(assembly, "tip")
    hinge = assembly.joint(
        "hinge",
        body0=base,
        frame0=JointFrame(xyz=(1.0, 0.0, 0.0)),
        body1=link,
        dofs=(JointDOF(JointAxis.ROT_Z, limits=(-math.pi, math.pi)),),
    )
    assembly.joint(
        "tip_mount",
        body0=link,
        frame0=JointFrame(xyz=(1.0, 0.0, 0.0)),
        body1=tip,
    )
    assembly.articulation("arm", root=base)

    resolved = assembly.resolve()
    posed = resolved.forward_kinematics({hinge.dof_id(hinge.dofs[0]): math.pi / 2})

    assert [item.joint.name for item in resolved.joints if item.articulation] == [
        "hinge",
        "tip_mount",
    ]
    assert np.allclose(posed.matrix(link)[:3, 3], (1.0, 0.0, 0.0))
    assert np.allclose(posed.matrix(tip)[:3, 3], (1.0, 1.0, 0.0), atol=1e-9)


@pytest.mark.parametrize("second_closure", [False, True])
def test_closed_loop_joints_are_derived_as_regular_excluded_constraints(
    second_closure: bool,
) -> None:
    resolved = four_bar(second_closure=second_closure).resolve()

    excluded = [item.joint.name for item in resolved.joints if item.exclude_from_articulation]
    assert excluded == (["rocker_ground", "crank_rocker"] if second_closure else ["rocker_ground"])
    assert resolved.has_closed_loops
    assert np.allclose(resolved.reference_state.matrix("rocker")[:3, 3], (2.0, 0.0, 0.0))

    with pytest.raises(ValidationError, match="closed-loop assemblies"):
        resolved.forward_kinematics({"ground_crank.rotZ": 0.2})


def test_closed_loop_requires_an_explicit_articulation_tree() -> None:
    assembly = four_bar()
    assembly.articulations.clear()
    assembly.articulation("main", root=assembly.get_rigid_body("ground"))

    with pytest.raises(ValidationError, match="requires explicit joints"):
        assembly.resolve()


def test_reference_state_rejects_a_geometrically_open_closure() -> None:
    assembly = four_bar()
    closure = assembly.get_joint("rocker_ground")
    assembly.joints[assembly.joints.index(closure)] = type(closure)(
        name=closure.name,
        body0=closure.body0,
        frame0=closure.frame0,
        body1=closure.body1,
        frame1=JointFrame(xyz=(3.0, 0.0, 0.0)),
        dofs=closure.dofs,
    )

    with pytest.raises(ValidationError, match="violates locked axis"):
        assembly.resolve()


def test_state_body_poses_are_authoritative_and_checked_against_every_joint() -> None:
    resolved = four_bar().resolve()
    valid = resolved.validate_state(PhysicsState(resolved.reference_state.body_poses))
    shifted = np.asarray(valid.body_poses["rocker"], dtype=float)
    shifted[1, 3] = 0.1
    invalid: dict[str, Mat4] = {
        name: np.asarray(matrix, dtype=float) for name, matrix in valid.body_poses.items()
    }
    invalid["rocker"] = shifted

    assert set(valid.body_poses) == {"ground", "crank", "coupler", "rocker"}
    with pytest.raises(ValidationError, match="violates locked axis"):
        resolved.validate_state(PhysicsState(invalid))


def test_world_joint_is_the_fixed_articulation_root() -> None:
    assembly = RigidBodyAssembly("mounted")
    base = add_body(assembly, "base")
    link = add_body(assembly, "link")
    mount = assembly.joint(
        "mount",
        body0=WORLD,
        frame0=JointFrame(xyz=(1.0, 2.0, 3.0)),
        body1=base,
    )
    hinge = assembly.joint("hinge", body0=base, body1=link, dofs=(JointDOF(JointAxis.ROT_Y),))
    assembly.articulation("fixed", root=mount, joints=(mount, hinge))

    resolved = assembly.resolve()

    assert np.allclose(resolved.reference_state.matrix(base)[:3, 3], (1.0, 2.0, 3.0))
    assert resolved.articulations[0].articulation.root is mount


def test_multiple_articulations_do_not_share_bodies_and_cross_joints_are_excluded() -> None:
    assembly = RigidBodyAssembly("two_arms")
    a = add_body(assembly, "a")
    b = add_body(assembly, "b")
    c = add_body(assembly, "c")
    d = add_body(assembly, "d")
    ab = assembly.joint("ab", body0=a, body1=b)
    cd = assembly.joint("cd", body0=c, body1=d)
    assembly.joint("bridge", body0=b, body1=c)
    assembly.articulation("left", root=a, joints=(ab,))
    assembly.articulation("right", root=c, joints=(cd,))

    resolved = assembly.resolve()

    assert resolved.get_joint("bridge").exclude_from_articulation
    with pytest.raises(ValidationError, match="only one articulation"):
        assembly.articulation("overlap", root=b, joints=(ab,))
        assembly.resolve()


def test_maximal_coordinate_assembly_needs_no_articulation() -> None:
    assembly = RigidBodyAssembly("maximal")
    a = add_body(assembly, "a")
    b = add_body(assembly, "b")
    assembly.joint("constraint", body0=a, body1=b, dofs=(JointDOF(JointAxis.ROT_X),))

    resolved = assembly.resolve()

    assert not resolved.articulations
    assert not resolved.joints[0].exclude_from_articulation


def test_invalid_graph_references_and_disconnected_bodies_fail() -> None:
    disconnected = RigidBodyAssembly("disconnected")
    add_body(disconnected, "left")
    add_body(disconnected, "right")
    with pytest.raises(ValidationError, match="connected joint graph"):
        disconnected.resolve()

    assembly = RigidBodyAssembly("foreign")
    local = add_body(assembly, "local")
    foreign = RigidBody("foreign")
    foreign.add(Box(0.1, 0.1, 0.1), name="shape")
    with pytest.raises(ValidationError, match="unknown body1"):
        assembly.joint("bad", body0=local, body1=foreign)


def test_articulation_selection_must_be_a_rooted_tree() -> None:
    assembly = four_bar()
    ground = assembly.get_rigid_body("ground")
    assembly.articulations.clear()
    with pytest.raises(ValidationError, match="must form a tree"):
        assembly.articulation("bad", root=ground, joints=assembly.joints)
        assembly.resolve()

    world_assembly = RigidBodyAssembly("wrong_root")
    body = add_body(world_assembly, "body")
    mount = world_assembly.joint("mount", body0=WORLD, body1=body)
    world_assembly.articulation("bad", root=body, joints=(mount,))
    with pytest.raises(ValidationError, match="use that joint as the articulation root"):
        world_assembly.resolve()


def test_articulation_root_must_belong_to_the_assembly() -> None:
    assembly = RigidBodyAssembly("local")
    add_body(assembly, "body")
    foreign = RigidBody("foreign")
    foreign.add(Box(0.1, 0.1, 0.1), name="shape")
    assembly.articulations.append(Articulation("bad", root=foreign, joints=()))

    with pytest.raises(ValidationError, match="outside this assembly"):
        assembly.resolve()
