"""Contact behavior follows from what each shape is made of."""

from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # pyright: ignore[reportAttributeAccessIssue]

from mini_articraft.sdk import RigidBodyAssembly, BoxGeometry, Material
from mini_articraft.sdk.export import export_assembly

# The usd-core stubs omit these schemas; bind them once rather than at every use.
_MaterialAPI = UsdPhysics.MaterialAPI  # pyright: ignore[reportAttributeAccessIssue]


def _model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("rig")
    part = model.rigid_body("frame")
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
    result = export_assembly(_model(), tmp_path)
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
    result = export_assembly(_model(), tmp_path)
    stage, meshes = _open(str(result.usdz))

    first = _bound_physics_material(stage, meshes["foot"])
    second = _bound_physics_material(stage, meshes["foot_b"])

    assert first is not None and second is not None
    assert first.GetPath() == second.GetPath()


def test_shapes_without_a_material_bind_no_physics_material(tmp_path: Path) -> None:
    model = RigidBodyAssembly("plain")
    model.rigid_body("body").add(BoxGeometry((0.1, 0.1, 0.1)), name="cube")

    result = export_assembly(model, tmp_path)
    stage, meshes = _open(str(result.usdz))

    # No invented friction: the engine applies its own default instead.
    assert _bound_physics_material(stage, meshes["cube"]) is None


def test_physics_binding_does_not_disturb_the_appearance_binding(tmp_path: Path) -> None:
    result = export_assembly(_model(), tmp_path)
    stage, meshes = _open(str(result.usdz))
    assert stage  # keep the stage alive: its prims go invalid when it is collected
    shell = meshes["shell"]

    appearance = UsdShade.MaterialBindingAPI(shell).GetDirectBindingRel("").GetTargets()
    physics = UsdShade.MaterialBindingAPI(shell).GetDirectBindingRel("physics").GetTargets()

    assert appearance and physics
    assert appearance[0] != physics[0]


def test_exported_stage_passes_openusd_physics_validation(tmp_path: Path) -> None:
    from pxr import UsdValidation

    result = export_assembly(_model(), tmp_path)
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


def test_a_coating_moves_friction_to_the_surface_without_moving_mass(tmp_path: Path) -> None:
    """A rubber grip on a steel bar is heavy like steel and grippy like rubber."""
    from mini_articraft.sdk.export import _resolve_part_mass

    model = RigidBodyAssembly("gripped")
    part = model.rigid_body("bar")
    part.add(
        BoxGeometry((0.02, 0.02, 0.30)),
        name="bar",
        material=Material.STEEL,
        coating=Material.RUBBER,
    )

    resolved = _resolve_part_mass(part, 0.0005)
    assert resolved is not None
    # Mass stays with the material underneath.
    assert resolved.mass == pytest.approx(0.02 * 0.02 * 0.30 * Material.STEEL.density)

    result = export_assembly(model, tmp_path)
    stage, meshes = _open(str(result.usdz))
    physics = _MaterialAPI(_bound_physics_material(stage, meshes["bar"]))

    # Friction follows the surface.
    assert physics.GetStaticFrictionAttr().Get() == pytest.approx(Material.RUBBER.static_friction)


def test_a_coating_also_supplies_the_look(tmp_path: Path) -> None:
    model = RigidBodyAssembly("plated")
    part = model.rigid_body("knob")
    part.add(
        BoxGeometry((0.05, 0.05, 0.05)),
        name="knob",
        material=Material.ABS_PLASTIC,
        coating=Material.STEEL,
    )

    shape = next(part._iter_shapes())

    # Chrome-plated plastic: weighs like plastic, slides and looks like metal.
    assert shape.material is Material.ABS_PLASTIC
    assert shape.surface_material is Material.STEEL
    assert shape.display_material is not None
    assert shape.display_material.metallic == 1.0


def test_textured_shapes_keep_their_physics_material(monkeypatch, tmp_path: Path) -> None:
    """The textured export path returns early and once skipped friction entirely."""
    from mini_articraft.sdk import ambientcg

    maps = tmp_path / "maps"
    maps.mkdir()
    color = maps / "Metal009_1K-JPG_Color.jpg"
    color.write_bytes(b"image")
    texture_set = ambientcg.TextureSet("Metal009", "1K", color)
    monkeypatch.setattr(
        ambientcg,
        "fetch_material",
        lambda kind: (texture_set, ambientcg.MaterialSpec("Metal009")),
    )

    model = RigidBodyAssembly("textured")
    model.rigid_body("body").add(BoxGeometry((0.1, 0.1, 0.1)), name="cube", material=Material.STEEL)

    result = export_assembly(model, tmp_path / "out", textured=True)
    stage, meshes = _open(str(result.usdz))
    physics = _bound_physics_material(stage, meshes["cube"])

    assert physics is not None
    assert _MaterialAPI(physics).GetStaticFrictionAttr().Get() == pytest.approx(
        Material.STEEL.static_friction
    )
