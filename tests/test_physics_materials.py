"""Contact behavior follows from what each shape is made of."""

from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # pyright: ignore[reportAttributeAccessIssue]

from mini_articraft.sdk import ArticulatedObject, BoxGeometry, Material
from mini_articraft.sdk.export import export_object

# The usd-core stubs omit these schemas; bind them once rather than at every use.
_MaterialAPI = UsdPhysics.MaterialAPI  # pyright: ignore[reportAttributeAccessIssue]


def _model() -> ArticulatedObject:
    model = ArticulatedObject("rig")
    part = model.part("frame")
    part.add(
        BoxGeometry((0.30, 0.16, 0.12)).translate(0.0, 0.0, 0.06),
        name="shell",
        material=Material.STEEL,
    )
    part.add(
        BoxGeometry((0.05, 0.02, 0.01)).translate(0.10, 0.0, 0.005),
        name="foot",
        material=Material.RUBBER,
    )
    part.add(
        BoxGeometry((0.05, 0.02, 0.01)).translate(-0.10, 0.0, 0.005),
        name="foot_b",
        material=Material.RUBBER,
    )
    return model


def _open(usdz: str) -> tuple[Usd.Stage, dict[str, Usd.Prim]]:
    stage = Usd.Stage.Open(usdz)
    return stage, {prim.GetName(): prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)}


def _bound_physics_material(stage: Usd.Stage, mesh: Usd.Prim) -> Usd.Prim | None:
    targets = UsdShade.MaterialBindingAPI(mesh).GetDirectBindingRel("physics").GetTargets()
    return stage.GetPrimAtPath(targets[0]) if targets else None


def test_each_collider_carries_its_material_friction(tmp_path: Path) -> None:
    result = export_object(_model(), tmp_path)
    stage, meshes = _open(str(result.usdz))

    shell = _MaterialAPI(_bound_physics_material(stage, meshes["shell"]))
    foot = _MaterialAPI(_bound_physics_material(stage, meshes["foot"]))

    # USD stores these as float32, so compare approximately.
    assert shell.GetStaticFrictionAttr().Get() == pytest.approx(Material.STEEL.static_friction)
    assert shell.GetDynamicFrictionAttr().Get() == pytest.approx(Material.STEEL.dynamic_friction)
    assert shell.GetRestitutionAttr().Get() == pytest.approx(Material.STEEL.restitution)
    # A steel frame on rubber feet grips through the feet, so friction has to bind
    # per shape rather than per part.
    assert foot.GetStaticFrictionAttr().Get() == pytest.approx(Material.RUBBER.static_friction)
    assert foot.GetStaticFrictionAttr().Get() > shell.GetStaticFrictionAttr().Get()


def test_one_physics_material_is_shared_by_every_shape_made_of_it(tmp_path: Path) -> None:
    result = export_object(_model(), tmp_path)
    stage, meshes = _open(str(result.usdz))

    first = _bound_physics_material(stage, meshes["foot"])
    second = _bound_physics_material(stage, meshes["foot_b"])

    assert first is not None and second is not None
    assert first.GetPath() == second.GetPath()


def test_shapes_without_a_material_bind_no_physics_material(tmp_path: Path) -> None:
    model = ArticulatedObject("plain")
    model.part("body").add(BoxGeometry((0.1, 0.1, 0.1)), name="cube")

    result = export_object(model, tmp_path)
    stage, meshes = _open(str(result.usdz))

    # No invented friction: the engine applies its own default instead.
    assert _bound_physics_material(stage, meshes["cube"]) is None


def test_physics_binding_does_not_disturb_the_appearance_binding(tmp_path: Path) -> None:
    result = export_object(_model(), tmp_path)
    stage, meshes = _open(str(result.usdz))
    assert stage  # keep the stage alive: its prims go invalid when it is collected
    shell = meshes["shell"]

    appearance = UsdShade.MaterialBindingAPI(shell).GetDirectBindingRel("").GetTargets()
    physics = UsdShade.MaterialBindingAPI(shell).GetDirectBindingRel("physics").GetTargets()

    assert appearance and physics
    assert appearance[0] != physics[0]


def test_exported_stage_passes_openusd_physics_validation(tmp_path: Path) -> None:
    from pxr import UsdValidation

    result = export_object(_model(), tmp_path)
    stage = Usd.Stage.Open(str(result.usdz))
    validators = UsdValidation.ValidationRegistry().GetOrLoadValidatorsByName(
        [
            "usdPhysicsValidators:ColliderChecker",
            "usdPhysicsValidators:RigidBodyChecker",
            "usdPhysicsValidators:PhysicsJointChecker",
            "usdPhysicsValidators:ArticulationChecker",
        ]
    )
    errors = UsdValidation.ValidationContext(validators).Validate(stage)

    assert [str(error) for error in errors] == []
