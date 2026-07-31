from __future__ import annotations

import json
import math

import pytest
from build123d import Box, Pos
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdValidation

import mini_articraft.sdk.export as export_module
from mini_articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    BoxGeometry,
    MotionLimits,
    Origin,
)
from mini_articraft.sdk.export import export_object


def test_export_writes_rigid_part_bodies_and_named_child_meshes(tmp_path) -> None:
    result = export_object(_hinge(), tmp_path)
    manifest = json.loads(result.manifest.read_text())
    stage = Usd.Stage.Open(str(result.usdz))

    assert result.usdz == tmp_path / "usdz" / "0000.usdz"
    assert manifest["files"] == {"usdz": "usdz/0000.usdz"}
    assert manifest["units"] == "meters"
    assert manifest["meters_per_unit"] == 1.0
    assert [shape["name"] for shape in manifest["parts"][0]["shapes"]] == [
        "shell",
        "trim",
    ]

    assert stage.GetDefaultPrim().GetPath().pathString == "/World"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == "Z"

    base = stage.GetPrimAtPath("/World/hinge/parts/base")
    assert base.GetTypeName() == "Xform"
    assert base.HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/World/hinge/parts/base/shapes").GetTypeName() == "Scope"

    shell = stage.GetPrimAtPath("/World/hinge/parts/base/shapes/shell")
    trim = stage.GetPrimAtPath("/World/hinge/parts/base/shapes/trim")
    assert shell.GetTypeName() == "Mesh"
    assert trim.GetTypeName() == "Mesh"
    assert tuple(
        round(float(value), 6) for value in shell.GetAttribute("primvars:displayColor").Get()[0]
    ) == (
        0.6,
        0.1,
        0.12,
    )
    assert trim.GetAttribute("primvars:displayOpacity").Get() == [0.7]

    joint_prim = stage.GetPrimAtPath("/World/hinge/joints/base_to_door")
    joint = UsdPhysics.RevoluteJoint.Get(stage, joint_prim.GetPath())
    assert joint_prim.GetAttribute("mini_articraft:articulationType").Get() == "revolute"
    assert joint.GetBody0Rel().GetTargets()[0].pathString == "/World/hinge/parts/base"
    assert joint.GetBody1Rel().GetTargets()[0].pathString == "/World/hinge/parts/door"
    assert joint.GetAxisAttr().Get() == "X"
    assert _joint_x_axis(joint) == (-0.074758, 0.71704, 0.693012)
    assert math.isclose(joint.GetUpperLimitAttr().Get(), math.degrees(1.57), abs_tol=1e-5)


def test_export_preserves_numbered_usdz_outputs(tmp_path) -> None:
    first = export_object(_hinge(), tmp_path)
    second = export_object(_hinge(), tmp_path)

    assert first.usdz.name == "0000.usdz"
    assert second.usdz.name == "0001.usdz"
    assert first.usdz.exists()
    assert second.usdz.exists()


def test_export_failure_cleans_temporary_package_and_preserves_prior_result(
    monkeypatch, tmp_path
) -> None:
    first = export_object(_hinge(), tmp_path)
    manifest_before = first.manifest.read_bytes()
    validate_usdz = export_module._validate_usdz

    def fail_validation(_path) -> None:
        raise RuntimeError("injected package validation failure")

    monkeypatch.setattr(export_module, "_validate_usdz", fail_validation)
    with pytest.raises(RuntimeError, match="injected"):
        export_object(_hinge(), tmp_path)

    assert first.manifest.read_bytes() == manifest_before
    assert sorted(path.name for path in first.usdz.parent.glob("*.usdz")) == ["0000.usdz"]
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.tmp.usdz"))

    monkeypatch.setattr(export_module, "_validate_usdz", validate_usdz)
    assert export_object(_hinge(), tmp_path).usdz.name == "0001.usdz"


def test_exported_package_passes_openusd_validators(tmp_path) -> None:
    stage = Usd.Stage.Open(str(export_object(_hinge(), tmp_path).usdz))
    validators = UsdValidation.ValidationRegistry().GetOrLoadValidatorsByName(
        [
            "usdUtilsValidators:UsdzPackageValidator",
            "usdGeomValidators:StageMetadataChecker",
            "usdValidation:CompositionErrorTest",
            "usdPhysicsValidators:RigidBodyChecker",
            "usdPhysicsValidators:PhysicsJointChecker",
            "usdPhysicsValidators:ArticulationChecker",
        ]
    )
    assert UsdValidation.ValidationContext(validators).Validate(stage) == []


def test_export_supports_every_articulation_type_and_matching_joint_frames(tmp_path) -> None:
    model = ArticulatedObject("motions")
    root = model.part("root")
    root.add(Box(0.1, 0.1, 0.1), name="body")
    fixed_part = model.part("fixed")
    fixed_part.add(Box(0.1, 0.1, 0.1), name="body")
    hinge = model.part("hinge")
    hinge.add(Box(0.1, 0.1, 0.1), name="body")
    rotor = model.part("rotor")
    rotor.add(Box(0.1, 0.1, 0.1), name="body")
    slider = model.part("slider")
    slider.add(Box(0.1, 0.1, 0.1), name="body")
    model.articulation(
        "fixed_mount",
        ArticulationType.FIXED,
        root,
        fixed_part,
        origin=Origin(xyz=(0.2, 0.0, 0.0), rpy=(0.1, 0.2, 0.3)),
    )
    model.articulation(
        "hinge_joint",
        ArticulationType.REVOLUTE,
        fixed_part,
        hinge,
        origin=Origin(xyz=(0.0, 0.3, 0.0), rpy=(0.2, 0.0, 0.1)),
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(lower=-0.5, upper=0.75),
    )
    model.articulation(
        "rotor_joint",
        ArticulationType.CONTINUOUS,
        hinge,
        rotor,
        origin=Origin(xyz=(0.0, 0.0, 0.4)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(),
    )
    model.articulation(
        "slider_joint",
        ArticulationType.PRISMATIC,
        rotor,
        slider,
        origin=Origin(xyz=(0.1, 0.2, 0.3), rpy=(0.0, 0.1, 0.0)),
        axis=(1.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.1, upper=0.2),
    )

    stage = Usd.Stage.Open(str(export_object(model, tmp_path).usdz))
    joints = {
        "fixed_mount": UsdPhysics.FixedJoint.Get(stage, "/World/motions/joints/fixed_mount"),
        "hinge_joint": UsdPhysics.RevoluteJoint.Get(stage, "/World/motions/joints/hinge_joint"),
        "rotor_joint": UsdPhysics.RevoluteJoint.Get(stage, "/World/motions/joints/rotor_joint"),
        "slider_joint": UsdPhysics.PrismaticJoint.Get(stage, "/World/motions/joints/slider_joint"),
    }

    assert joints["hinge_joint"].GetLowerLimitAttr().Get() == pytest.approx(
        math.degrees(-0.5), abs=1e-5
    )
    assert joints["hinge_joint"].GetUpperLimitAttr().Get() == pytest.approx(
        math.degrees(0.75), abs=1e-5
    )
    assert not joints["rotor_joint"].GetLowerLimitAttr().HasAuthoredValueOpinion()
    assert not joints["rotor_joint"].GetUpperLimitAttr().HasAuthoredValueOpinion()
    assert joints["slider_joint"].GetLowerLimitAttr().Get() == pytest.approx(-0.1)
    assert joints["slider_joint"].GetUpperLimitAttr().Get() == pytest.approx(0.2)
    for joint in joints.values():
        _assert_joint_frames_meet(stage, joint)


@pytest.mark.parametrize("axis", [(1e-320, 0.0, 0.0), (1e308, 1e308, 0.0)])
def test_export_robustly_normalizes_finite_nonzero_axes(tmp_path, axis) -> None:
    model = ArticulatedObject("axis")
    root = model.part("root")
    root.add(Box(0.1, 0.1, 0.1), name="body")
    child = model.part("child")
    child.add(Box(0.1, 0.1, 0.1), name="body")
    model.articulation(
        "spin",
        ArticulationType.CONTINUOUS,
        root,
        child,
        axis=axis,
        motion_limits=MotionLimits(),
    )

    result = export_object(model, tmp_path)

    assert result.usdz.is_file()


def _hinge() -> ArticulatedObject:
    model = ArticulatedObject("hinge")
    base = model.part("base")
    base.add(Box(1.0, 1.0, 0.2), name="shell", color=(0.6, 0.1, 0.12))
    base.add(
        BoxGeometry((0.2, 0.2, 0.04)).translate(0.0, 0.0, 0.12),
        name="trim",
        color=(0.8, 0.8, 0.82, 0.7),
    )
    door = model.part("door")
    door.add(Pos(Z=0.5) * Box(0.8, 0.1, 1.0), name="panel", color=(0.2, 0.35, 0.8))
    model.articulation(
        "base_to_door",
        ArticulationType.REVOLUTE,
        base,
        door,
        origin=Origin(xyz=(0.0, 0.0, 0.2), rpy=(0.0, 0.2, 0.3)),
        axis=(0.0, 1.0, 1.0),
        motion_limits=MotionLimits(lower=0.0, upper=1.57),
    )
    return model


def _joint_x_axis(joint: UsdPhysics.RevoluteJoint) -> tuple[float, float, float]:
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(joint.GetLocalRot0Attr().Get())
    vector = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetNormalized()
    x, y, z = (round(float(component), 6) for component in vector)
    return (x, y, z)


def _assert_joint_frames_meet(stage: Usd.Stage, joint) -> None:
    positions = []
    directions = []
    for relation, position_attr, rotation_attr in (
        (joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
        (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
    ):
        body = stage.GetPrimAtPath(relation.GetTargets()[0])
        body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()  # pyright: ignore[reportAttributeAccessIssue]
        )
        positions.append(body_world.Transform(Gf.Vec3d(position_attr.Get())))
        local_rotation = Gf.Matrix4d(1.0)
        local_rotation.SetRotate(rotation_attr.Get())
        directions.append(
            body_world.TransformDir(
                local_rotation.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            ).GetNormalized()
        )
    assert tuple(positions[0]) == pytest.approx(tuple(positions[1]), abs=1e-6)
    assert tuple(directions[0]) == pytest.approx(tuple(directions[1]), abs=1e-6)
