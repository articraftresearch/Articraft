"""Gravity and initial body state on the rigid-body graph model.

The same settings #63 gave `RigidBodyAssembly`, carried onto the assembly so the
cut-over does not silently drop them.
"""

from __future__ import annotations

import json
import math

import pytest
from pxr import Usd, UsdPhysics

from mini_articraft.sdk import BodyState, BoxGeometry, PhysicsScene
from mini_articraft.sdk.assembly import JointAxis, JointDOF, JointFrame, RigidBodyAssembly
from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk.export import export_assembly


def _assembly(scene: PhysicsScene | None = None, **body_kwargs) -> RigidBodyAssembly:
    assembly = (
        RigidBodyAssembly("rig") if scene is None else RigidBodyAssembly("rig", scene=scene)
    )
    base = assembly.rigid_body("base")
    base.add(BoxGeometry((0.2, 0.2, 0.05)), name="plate")
    arm = assembly.rigid_body("arm", **body_kwargs)
    arm.add(BoxGeometry((0.05, 0.05, 0.2)), name="bar")
    assembly.joint(
        "hinge",
        body0=base,
        frame0=JointFrame(xyz=(0.0, 0.0, 0.025)),
        body1=arm,
        frame1=JointFrame(xyz=(0.0, 0.0, -0.1)),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=(-1.0, 1.0)),),
    )
    assembly.articulation("main", root=base, joints=["hinge"])
    return assembly


def test_assembly_exports_one_physics_scene_with_its_gravity(tmp_path) -> None:
    result = export_assembly(_assembly(PhysicsScene(magnitude=1.62)), tmp_path)
    stage = Usd.Stage.Open(str(result.usdz))
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]

    assert len(scenes) == 1
    scene = UsdPhysics.Scene(scenes[0])
    assert scene.GetGravityMagnitudeAttr().Get() == pytest.approx(1.62)
    assert tuple(scene.GetGravityDirectionAttr().Get()) == (0.0, 0.0, -1.0)

    manifest = json.loads(result.manifest.read_text())
    assert manifest["scene"] == {
        "gravity_direction": [0.0, 0.0, -1.0],
        "gravity_magnitude": 1.62,
    }


def test_assembly_defaults_to_earth_gravity(tmp_path) -> None:
    result = export_assembly(_assembly(), tmp_path)

    assert json.loads(result.manifest.read_text())["scene"]["gravity_magnitude"] == 9.81


def test_body_state_reaches_usd_with_angles_in_degrees(tmp_path) -> None:
    state = BodyState(kinematic=True, angular_velocity=(0.0, math.pi, 0.0), starts_asleep=True)
    result = export_assembly(_assembly(body_state=state), tmp_path)
    stage = Usd.Stage.Open(str(result.usdz))
    rigid = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/World/rig/rigid_bodies/arm"))

    assert rigid.GetKinematicEnabledAttr().Get() is True
    assert rigid.GetStartsAsleepAttr().Get() is True
    assert tuple(rigid.GetAngularVelocityAttr().Get()) == pytest.approx((0.0, 180.0, 0.0))

    manifest = json.loads(result.manifest.read_text())
    arm = next(b for b in manifest["rigid_bodies"] if b["name"] == "arm")
    assert arm["body_state"]["kinematic"] is True
    assert arm["body_state"]["angular_velocity"] == pytest.approx([0.0, math.pi, 0.0])


def test_assembly_rejects_a_scene_that_is_not_a_physics_scene() -> None:
    with pytest.raises(ValidationError, match="scene must be a PhysicsScene"):
        RigidBodyAssembly("rig", scene="earth")  # pyright: ignore[reportArgumentType]


def test_rigid_body_rejects_a_body_state_that_is_not_one() -> None:
    with pytest.raises(ValidationError, match="body_state must be BodyState"):
        RigidBodyAssembly("rig").rigid_body("base", body_state="asleep")  # pyright: ignore[reportArgumentType]
