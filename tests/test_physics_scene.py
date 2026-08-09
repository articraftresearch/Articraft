from __future__ import annotations

import json
import math

import pytest
from pxr import Usd, UsdPhysics

from mini_articraft.sdk import (
    ArticulatedObject,
    BodyState,
    BoxGeometry,
    PhysicsScene,
)
from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk.export import export_object


def _model(scene: PhysicsScene | None = None, **part_kwargs) -> ArticulatedObject:
    model = ArticulatedObject("cart") if scene is None else ArticulatedObject("cart", scene=scene)
    body = model.part("body", **part_kwargs)
    body.add(BoxGeometry((0.2, 0.2, 0.2)), name="shell")
    model.validate()
    return model


def _rigid_body(usdz) -> tuple[Usd.Stage, UsdPhysics.RigidBodyAPI]:
    # The stage is returned with the schema: a prim goes invalid the moment its
    # stage is collected.
    stage = Usd.Stage.Open(str(usdz))
    return stage, UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/World/cart/parts/body"))


def test_default_scene_is_earth_gravity_down_the_up_axis(tmp_path) -> None:
    result = export_object(_model(), tmp_path)

    manifest = json.loads(result.manifest.read_text())
    assert manifest["scene"] == {
        "gravity_direction": [0.0, 0.0, -1.0],
        "gravity_magnitude": 9.81,
    }


def test_explicit_gravity_is_normalized_and_recorded(tmp_path) -> None:
    # A non-unit direction is a direction, not a magnitude: the length is dropped.
    scene = PhysicsScene(direction=(0.0, -2.0, 0.0), magnitude=1.62)
    result = export_object(_model(scene), tmp_path)

    assert json.loads(result.manifest.read_text())["scene"] == {
        "gravity_direction": [0.0, -1.0, 0.0],
        "gravity_magnitude": 1.62,
    }


def test_the_asset_package_carries_no_physics_scene(tmp_path) -> None:
    # Gravity belongs to the stage that references this asset. A scene prim here
    # would collide with that one, and no body carries a simulation owner.
    result = export_object(_model(PhysicsScene(magnitude=1.62)), tmp_path)
    stage = Usd.Stage.Open(str(result.usdz))

    assert [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)] == []


def test_an_unconfigured_part_authors_no_body_state(tmp_path) -> None:
    # The default is a free body at rest, which is also what USD falls back to.
    # Leaving it unauthored means a scenario layer has no opinion to fight.
    result = export_object(_model(), tmp_path)
    _stage, rigid = _rigid_body(result.usdz)
    prim = rigid.GetPrim()

    assert [name for name in prim.GetAuthoredPropertyNames() if name.startswith("physics:")] == []
    assert rigid.GetRigidBodyEnabledAttr().Get() is True
    assert tuple(rigid.GetVelocityAttr().Get()) == (0.0, 0.0, 0.0)

    # The manifest still describes the model in full: it is metadata, not a
    # layer anything composes over.
    manifest = json.loads(result.manifest.read_text())
    assert manifest["parts"][0]["body_state"] == {
        "enabled": True,
        "kinematic": False,
        "linear_velocity": [0.0, 0.0, 0.0],
        "angular_velocity": [0.0, 0.0, 0.0],
        "starts_asleep": False,
    }


def test_authored_body_state_reaches_usd_with_angles_in_degrees(tmp_path) -> None:
    state = BodyState(
        kinematic=True,
        linear_velocity=(0.5, 0.0, -1.25),
        angular_velocity=(0.0, math.pi, 0.0),
        starts_asleep=True,
    )
    result = export_object(_model(body_state=state), tmp_path)
    _stage, rigid = _rigid_body(result.usdz)

    assert rigid.GetKinematicEnabledAttr().Get() is True
    assert rigid.GetStartsAsleepAttr().Get() is True
    assert tuple(rigid.GetVelocityAttr().Get()) == pytest.approx((0.5, 0.0, -1.25))
    # The SDK authors rad/s; USD stores deg/s.
    assert tuple(rigid.GetAngularVelocityAttr().Get()) == pytest.approx((0.0, 180.0, 0.0))
    # enabled was left alone, so it stays unauthored while the rest are written.
    assert sorted(
        name for name in rigid.GetPrim().GetAuthoredPropertyNames() if name.startswith("physics:")
    ) == [
        "physics:angularVelocity",
        "physics:kinematicEnabled",
        "physics:startsAsleep",
        "physics:velocity",
    ]
    manifest = json.loads(result.manifest.read_text())
    assert manifest["parts"][0]["body_state"]["angular_velocity"] == pytest.approx(
        [0.0, math.pi, 0.0]
    )


def test_disabled_body_exports_as_a_body_the_simulator_does_not_move(tmp_path) -> None:
    result = export_object(_model(body_state=BodyState(enabled=False)), tmp_path)
    _stage, rigid = _rigid_body(result.usdz)

    assert rigid.GetRigidBodyEnabledAttr().Get() is False


def test_scene_and_body_state_reject_unusable_values() -> None:
    with pytest.raises(ValidationError, match="non-zero"):
        PhysicsScene(direction=(0.0, 0.0, 0.0))
    with pytest.raises(ValidationError, match="negative"):
        PhysicsScene(magnitude=-1.0)
    with pytest.raises(ValidationError, match="finite"):
        PhysicsScene(magnitude=math.inf)
    with pytest.raises(ValidationError, match="must be a bool"):
        BodyState(kinematic="yes")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError, match="cannot be kinematic"):
        BodyState(enabled=False, kinematic=True)
    with pytest.raises(ValidationError, match="3 numeric values"):
        BodyState(linear_velocity=(1.0, 2.0))  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError, match="scene must be a PhysicsScene"):
        ArticulatedObject("cart", scene="earth")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError, match="body_state must be BodyState"):
        ArticulatedObject("cart").part("body", body_state="asleep")  # pyright: ignore[reportArgumentType]
