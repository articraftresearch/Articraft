from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from build123d import Box
from pxr import Kind, Usd, UsdPhysics

from articraft.sdk.assembly import (
    WORLD,
    JointAxis,
    JointDOF,
    JointFrame,
    RigidBodyAssembly,
)
from articraft.sdk.export import export_assembly


def body(assembly: RigidBodyAssembly, name: str):
    result = assembly.rigid_body(name)
    result.add(Box(0.1, 0.1, 0.1), name="shape")
    return result


def four_bar(*, second_closure: bool = False) -> RigidBodyAssembly:
    assembly = RigidBodyAssembly("four_bar")
    ground = body(assembly, "ground")
    crank = body(assembly, "crank")
    coupler = body(assembly, "coupler")
    rocker = body(assembly, "rocker")
    dofs = (JointDOF(JointAxis.ROT_Z),)
    a = assembly.joint("ground_crank", body0=ground, body1=crank, dofs=dofs)
    b = assembly.joint(
        "crank_coupler",
        body0=crank,
        frame0=JointFrame(xyz=(1.0, 0.0, 0.0)),
        body1=coupler,
        dofs=dofs,
    )
    c = assembly.joint(
        "coupler_rocker",
        body0=coupler,
        frame0=JointFrame(xyz=(1.0, 0.0, 0.0)),
        body1=rocker,
        dofs=dofs,
    )
    assembly.joint(
        "rocker_ground",
        body0=rocker,
        body1=ground,
        frame1=JointFrame(xyz=(2.0, 0.0, 0.0)),
        dofs=dofs,
    )
    if second_closure:
        assembly.joint(
            "crank_rocker",
            body0=crank,
            frame0=JointFrame(xyz=(2.0, 0.0, 0.0)),
            body1=rocker,
            dofs=dofs,
        )
    assembly.articulation("main", root=ground, joints=(a, b, c))
    return assembly


def open_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path))
    assert stage is not None
    return stage


@pytest.mark.parametrize("second_closure", [False, True])
def test_closed_loop_exports_every_joint_and_excludes_only_closures(
    tmp_path: Path, second_closure: bool
) -> None:
    result = export_assembly(four_bar(second_closure=second_closure), tmp_path)
    stage = open_stage(result.usdz)
    root = stage.GetPrimAtPath("/World/four_bar")
    ground = stage.GetPrimAtPath("/World/four_bar/rigid_bodies/ground")
    joints = stage.GetPrimAtPath("/World/four_bar/joints")

    assert Usd.ModelAPI(root).GetKind() == Kind.Tokens.assembly
    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert ground.HasAPI(UsdPhysics.ArticulationRootAPI)
    excluded = {
        joint.GetAttribute("articraft:name").Get(): bool(
            UsdPhysics.Joint(joint).GetExcludeFromArticulationAttr().Get()
        )
        for joint in joints.GetChildren()
    }
    expected = {
        "ground_crank": False,
        "crank_coupler": False,
        "coupler_rocker": False,
        "rocker_ground": True,
    }
    if second_closure:
        expected["crank_rocker"] = True
    assert excluded == expected
    assert result.audit.joint_count == len(expected)
    assert result.audit.articulation_count == 1


def test_manifest_v2_records_graph_frames_dofs_exclusions_and_reference_state(
    tmp_path: Path,
) -> None:
    result = export_assembly(four_bar(), tmp_path)
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert [item["name"] for item in payload["rigid_bodies"]] == [
        "ground",
        "crank",
        "coupler",
        "rocker",
    ]
    closure = next(item for item in payload["joints"] if item["name"] == "rocker_ground")
    assert closure == {
        "name": "rocker_ground",
        "type": "revolute",
        "body0": "rocker",
        "body1": "ground",
        "frame0": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        "frame1": {"xyz": [2.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        "dofs": [{"axis": "rotZ", "limits": None}],
        "articulation": None,
        "exclude_from_articulation": True,
    }
    assert payload["articulations"] == [
        {
            "name": "main",
            "root": {"type": "rigid_body", "name": "ground"},
            "joints": ["ground_crank", "crank_coupler", "coupler_rocker"],
            "rigid_bodies": ["coupler", "crank", "ground", "rocker"],
        }
    ]
    assert set(payload["reference_state"]["body_poses"]) == {
        "ground",
        "crank",
        "coupler",
        "rocker",
    }


def test_fixed_world_joint_is_the_articulation_root_and_has_one_body_target(
    tmp_path: Path,
) -> None:
    assembly = RigidBodyAssembly("mounted")
    base = body(assembly, "base")
    mount = assembly.joint(
        "mount",
        body0=WORLD,
        frame0=JointFrame(xyz=(1.0, 2.0, 3.0)),
        body1=base,
        frame1=JointFrame(rpy=(0.1, 0.2, 0.3)),
    )
    assembly.articulation("fixed", root=mount, joints=(mount,))

    stage = open_stage(export_assembly(assembly, tmp_path).usdz)
    prim = stage.GetPrimAtPath("/World/mounted/joints/mount")
    schema = UsdPhysics.FixedJoint(prim)

    assert prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not schema.GetBody0Rel().GetTargets()
    assert [str(path) for path in schema.GetBody1Rel().GetTargets()] == [
        "/World/mounted/rigid_bodies/base"
    ]
    assert tuple(schema.GetLocalPos0Attr().Get()) == pytest.approx((1.0, 2.0, 3.0))
    assert tuple(schema.GetLocalPos1Attr().Get()) == pytest.approx((0.0, 0.0, 0.0))


def test_simple_joints_use_specialized_native_schemas_and_radian_limits_become_degrees(
    tmp_path: Path,
) -> None:
    assembly = RigidBodyAssembly("specialized")
    a = body(assembly, "a")
    b = body(assembly, "b")
    joint = assembly.joint(
        "hinge",
        body0=a,
        frame0=JointFrame(xyz=(0.1, 0.2, 0.3), rpy=(0.2, 0.1, 0.0)),
        body1=b,
        frame1=JointFrame(xyz=(-0.1, 0.0, 0.2), rpy=(0.0, 0.3, 0.4)),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=(-math.pi / 2, math.pi / 4)),),
    )
    assembly.articulation("main", root=a, joints=(joint,))

    stage = open_stage(export_assembly(assembly, tmp_path).usdz)
    schema = UsdPhysics.RevoluteJoint.Get(stage, "/World/specialized/joints/hinge")

    assert schema
    assert schema.GetAxisAttr().Get() == "Y"
    assert schema.GetLowerLimitAttr().Get() == pytest.approx(-90.0)
    assert schema.GetUpperLimitAttr().Get() == pytest.approx(45.0)
    assert tuple(schema.GetLocalPos0Attr().Get()) == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(schema.GetLocalPos1Attr().Get()) == pytest.approx((-0.1, 0.0, 0.2))


def test_multi_dof_joint_uses_generic_usd_joint_and_per_axis_limit_apis(
    tmp_path: Path,
) -> None:
    assembly = RigidBodyAssembly("d6")
    a = body(assembly, "a")
    b = body(assembly, "b")
    assembly.joint(
        "planar",
        body0=a,
        body1=b,
        dofs=(
            JointDOF(JointAxis.TRANS_X, limits=(-0.2, 0.3)),
            JointDOF(JointAxis.ROT_Z),
        ),
    )

    stage = open_stage(export_assembly(assembly, tmp_path).usdz)
    prim = stage.GetPrimAtPath("/World/d6/joints/planar")

    assert prim.IsA(UsdPhysics.Joint)
    assert not prim.IsA(UsdPhysics.RevoluteJoint)
    limited = UsdPhysics.LimitAPI.Get(prim, "transX")
    locked = UsdPhysics.LimitAPI.Get(prim, "transY")
    assert limited.GetLowAttr().Get() == pytest.approx(-0.2)
    assert limited.GetHighAttr().Get() == pytest.approx(0.3)
    assert locked.GetLowAttr().Get() > locked.GetHighAttr().Get()
    assert not prim.HasAPI(UsdPhysics.LimitAPI, "rotZ")
