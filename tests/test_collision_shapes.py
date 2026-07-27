from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from mini_articraft.sdk import (
    ArticulatedObject,
    BoxGeometry,
    CollisionApproximation,
    CylinderGeometry,
    ShapeRole,
    TestContext,
)
from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk.export import export_object


def _model() -> ArticulatedObject:
    model = ArticulatedObject("rig")
    part = model.part("body")
    part.add(BoxGeometry((0.2, 0.2, 0.1)), name="shell")
    part.add(
        CylinderGeometry(0.05, 0.2),
        name="proxy",
        role=ShapeRole.COLLISION,
        collision_approximation=CollisionApproximation.CONVEX_HULL,
    )
    part.add(BoxGeometry((0.02, 0.02, 0.02)), name="badge", role=ShapeRole.VISUAL)
    return model


# The usd-core stubs omit these schemas; bind them once instead of repeating the
# suppression at every call site.
_CollisionAPI = UsdPhysics.CollisionAPI  # pyright: ignore[reportAttributeAccessIssue]
_MeshCollisionAPI = UsdPhysics.MeshCollisionAPI  # pyright: ignore[reportAttributeAccessIssue]
_Imageable = UsdGeom.Imageable  # pyright: ignore[reportAttributeAccessIssue]


def _open_meshes(usdz: str) -> tuple[Usd.Stage, dict[str, Usd.Prim]]:
    """Return the stage alongside its meshes.

    The caller must keep the stage alive: prims from a stage that has been
    collected are invalid, and every schema access on them raises.
    """
    stage = Usd.Stage.Open(usdz)
    return stage, {prim.GetName(): prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)}


def test_shapes_are_drawn_and_collidable_by_default() -> None:
    model = ArticulatedObject("plain")
    model.part("body").add(BoxGeometry((0.1, 0.1, 0.1)), name="cube")

    shape = next(model.get_part("body")._iter_shapes())

    assert shape.role is ShapeRole.VISUAL_AND_COLLISION
    assert shape.collision_approximation is CollisionApproximation.CONVEX_DECOMPOSITION


def test_export_authors_collision_apis_per_role(tmp_path: Path) -> None:
    result = export_object(_model(), tmp_path)
    stage, meshes = _open_meshes(str(result.usdz))
    assert stage

    assert meshes["shell"].HasAPI(_CollisionAPI)
    assert meshes["proxy"].HasAPI(_CollisionAPI)
    # A visual-only shape must not be a collider, or the simulator collides with
    # decoration the author explicitly excluded.
    assert not meshes["badge"].HasAPI(_CollisionAPI)


def test_export_records_the_requested_approximation(tmp_path: Path) -> None:
    result = export_object(_model(), tmp_path)
    stage, meshes = _open_meshes(str(result.usdz))
    assert stage

    def approximation(name: str) -> str:
        return _MeshCollisionAPI(meshes[name]).GetApproximationAttr().Get()

    assert approximation("shell") == "convexDecomposition"
    assert approximation("proxy") == "convexHull"


def test_collision_only_shapes_export_invisible(tmp_path: Path) -> None:
    result = export_object(_model(), tmp_path)
    stage, meshes = _open_meshes(str(result.usdz))
    assert stage

    # The collider has to stay on the stage to resolve, but nothing should draw it.
    assert _Imageable(meshes["proxy"]).ComputeVisibility() == "invisible"
    assert _Imageable(meshes["shell"]).ComputeVisibility() == "inherited"
    assert _Imageable(meshes["badge"]).ComputeVisibility() == "inherited"


def test_manifest_records_role_and_approximation(tmp_path: Path) -> None:
    import json

    result = export_object(_model(), tmp_path)
    manifest = json.loads(Path(result.manifest).read_text())
    shapes = {shape["name"]: shape for shape in manifest["parts"][0]["shapes"]}

    assert shapes["shell"]["role"] == "visual_and_collision"
    assert shapes["shell"]["collision_approximation"] == "convexDecomposition"
    assert shapes["proxy"]["collision_approximation"] == "convexHull"
    assert shapes["badge"]["role"] == "visual"
    assert shapes["badge"]["collision_approximation"] is None


def test_a_part_with_only_visual_shapes_fails_the_check() -> None:
    model = ArticulatedObject("ghost")
    model.part("body").add(BoxGeometry((0.1, 0.1, 0.1)), name="cube", role=ShapeRole.VISUAL)

    ctx = TestContext(model)
    ctx.fail_if_parts_have_no_collider()
    report = ctx.report()

    assert not report.passed
    assert "body" in report.failures[0].details


def test_a_collision_only_shape_satisfies_the_check() -> None:
    model = ArticulatedObject("proxied")
    part = model.part("body")
    part.add(BoxGeometry((0.1, 0.1, 0.1)), name="pretty", role=ShapeRole.VISUAL)
    part.add(BoxGeometry((0.1, 0.1, 0.1)), name="collider", role=ShapeRole.COLLISION)

    ctx = TestContext(model)
    ctx.fail_if_parts_have_no_collider()

    assert ctx.report().passed


def test_invalid_role_and_approximation_are_rejected() -> None:
    part = ArticulatedObject("bad").part("body")

    with pytest.raises(ValidationError, match="role must be a ShapeRole"):
        part.add(BoxGeometry((0.1, 0.1, 0.1)), name="a", role="collision")  # pyright: ignore
    with pytest.raises(ValidationError, match="must be a CollisionApproximation"):
        part.add(
            BoxGeometry((0.1, 0.1, 0.1)),
            name="b",
            collision_approximation="convexHull",  # pyright: ignore
        )


def test_exported_stage_passes_openusd_physics_validation(tmp_path: Path) -> None:
    """The colliders we author must satisfy OpenUSD's own physics validators.

    Run against a whole articulated model rather than a lone part, so the
    collider, rigid body, and joint checkers all see something to inspect.
    """
    from pxr import UsdValidation

    from mini_articraft.sdk import (
        ArticulationType,
        MassProperties,
        MaterialDensity,
        MotionLimits,
        Origin,
    )

    model = ArticulatedObject("rig")
    base = model.part("base", mass_properties=MassProperties(material=MaterialDensity.STEEL))
    base.add(BoxGeometry((0.2, 0.2, 0.02)), name="plate")
    arm = model.part("arm", mass_properties=MassProperties(material=MaterialDensity.ALUMINUM))
    arm.add(BoxGeometry((0.03, 0.15, 0.03)).translate(0.0, 0.075, 0.03), name="beam")
    arm.add(
        CylinderGeometry(0.02, 0.05),
        name="hub",
        role=ShapeRole.COLLISION,
        collision_approximation=CollisionApproximation.CONVEX_HULL,
    )
    model.articulation(
        "pivot",
        ArticulationType.REVOLUTE,
        base,
        arm,
        origin=Origin(xyz=(0.0, 0.0, 0.02)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(effort=5.0, velocity=2.0, lower=0.0, upper=1.5),
    )

    result = export_object(model, tmp_path)
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
