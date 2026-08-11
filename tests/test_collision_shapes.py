from __future__ import annotations

from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

from articraft.sdk import (
    BoxGeometry,
    CylinderGeometry,
    JointAxis,
    JointDOF,
    JointFrame,
    Material,
    RigidBodyAssembly,
    SphereGeometry,
)
from articraft.sdk.export import export_assembly
from articraft.sdk.mesh import boolean_difference

# The usd-core stubs omit these schemas; bind them once instead of repeating the
# suppression at every call site.
_CollisionAPI = UsdPhysics.CollisionAPI  # pyright: ignore[reportAttributeAccessIssue]
_MeshCollisionAPI = UsdPhysics.MeshCollisionAPI  # pyright: ignore[reportAttributeAccessIssue]


def _open_meshes(usdz: str) -> tuple[Usd.Stage, dict[str, Usd.Prim]]:
    """Return the stage alongside its meshes.

    The caller must keep the stage alive: prims from a stage that has been
    collected are invalid, and every schema access on them raises.
    """
    stage = Usd.Stage.Open(usdz)
    return stage, {prim.GetName(): prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)}


def _approximation(prim: Usd.Prim) -> str:
    return _MeshCollisionAPI(prim).GetApproximationAttr().Get()


def test_every_shape_is_a_collider(tmp_path: Path) -> None:
    """Collision geometry is one-to-one with display geometry, with nothing to declare."""
    model = RigidBodyAssembly("plain")
    part = model.rigid_body("body")
    part.add(BoxGeometry((0.2, 0.2, 0.1)), name="shell")
    part.add(SphereGeometry(0.04), name="knob")

    result = export_assembly(model, tmp_path)
    stage, meshes = _open_meshes(str(result.usdz))
    assert stage

    assert set(meshes) == {"shell", "knob"}
    assert all(prim.HasAPI(_CollisionAPI) for prim in meshes.values())


def test_convex_geometry_collides_as_a_hull(tmp_path: Path) -> None:
    model = RigidBodyAssembly("convex")
    model.rigid_body("body").add(CylinderGeometry(0.05, 0.1), name="drum")

    result = export_assembly(model, tmp_path)
    stage, meshes = _open_meshes(str(result.usdz))
    assert stage

    # A hull is exact for convex geometry, and cheaper than decomposing it.
    assert _approximation(meshes["drum"]) == "convexHull"


def test_concave_geometry_collides_as_a_decomposition(tmp_path: Path) -> None:
    model = RigidBodyAssembly("concave")
    shell = boolean_difference(
        CylinderGeometry(0.055, 0.10, radial_segments=48).translate(0.0, 0.0, 0.05),
        CylinderGeometry(0.051, 0.10, radial_segments=48).translate(0.0, 0.0, 0.054),
    )
    model.rigid_body("body").add(shell, name="shell")

    result = export_assembly(model, tmp_path)
    stage, meshes = _open_meshes(str(result.usdz))
    assert stage

    # A hull of a hollow shell fills the cavity, so a lid could never close into it.
    assert _approximation(meshes["shell"]) == "convexDecomposition"


def test_shapes_with_materials_are_colliders_too(tmp_path: Path) -> None:
    """A shape that takes the material-bound export path must not skip collision."""
    model = RigidBodyAssembly("finished")
    model.rigid_body("body").add(
        BoxGeometry((0.1, 0.1, 0.1)),
        name="cube",
        material=Material.STEEL,
    )

    result = export_assembly(model, tmp_path)
    stage, meshes = _open_meshes(str(result.usdz))
    assert stage

    assert meshes["cube"].HasAPI(_CollisionAPI)


def test_exported_stage_passes_openusd_physics_validation(tmp_path: Path) -> None:
    """The colliders we author must satisfy OpenUSD's own physics validators.

    Run against a whole articulated model rather than a lone part, so the
    collider, rigid body, and joint checkers all see something to inspect.
    """
    from pxr import UsdValidation

    model = RigidBodyAssembly("rig")
    base = model.rigid_body("base")
    base.add(BoxGeometry((0.2, 0.2, 0.02)), name="plate", material=Material.STEEL)
    arm = model.rigid_body("arm")
    arm.add(
        BoxGeometry((0.03, 0.15, 0.03)).translate(0.0, 0.075, 0.03),
        name="beam",
        material=Material.ALUMINUM,
    )
    model.joint(
        "pivot",
        body0=base,
        frame0=JointFrame(),
        body1=arm,
        frame1=JointFrame(),
        dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 1.5)),),
    )

    result = export_assembly(model, tmp_path)
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
