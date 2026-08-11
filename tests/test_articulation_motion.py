from __future__ import annotations

from articraft.sdk import (
    BoxGeometry,
    FailureKind,
    JointAxis,
    JointDOF,
    JointFrame,
    RigidBodyAssembly,
    TestContext,
)


def _hinged_lid(pivot: tuple[float, float, float]) -> RigidBodyAssembly:
    """A base slab with a lid resting on it, hinged about X at `pivot`.

    The lid geometry lives in the joint frame, so it is authored relative to
    `pivot` such that at rest it seats on the base top (world z=0.01) regardless of
    where the pivot is. A pivot on the contact plane keeps the lid seated as it
    swings; a pivot above the contact plane lifts the lid clear as it rotates.
    """
    model = RigidBodyAssembly("hinge_test")
    base = model.rigid_body("base")
    base.add(BoxGeometry((0.10, 0.10, 0.02)), name="base_slab")  # top at z=0.01
    world_center = (0.0, 0.0, 0.02)
    offset = tuple(world_center[i] - pivot[i] for i in range(3))
    lid = model.rigid_body("lid")
    lid.add(BoxGeometry((0.10, 0.10, 0.02)).translate(*offset), name="lid_slab")
    model.joint(
        "lid_hinge",
        body0=base,
        frame0=JointFrame(),
        body1=lid,
        frame1=JointFrame(),
        dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 1.2)),),
    )
    return model


def test_separation_check_passes_hinge_at_the_contact_edge() -> None:
    # Pivot on the rear contact edge: the lid flips but its rear edge stays seated.
    model = _hinged_lid((0.0, -0.05, 0.01))
    ctx = TestContext(model)
    ctx.fail_if_articulation_separates_child()
    assert ctx.report().passed


def test_separation_check_fails_a_lid_that_lifts_off() -> None:
    # Pivot above the contact plane: rotating lifts the lid clear of the base.
    model = _hinged_lid((0.0, -0.05, 0.06))
    ctx = TestContext(model)
    ctx.fail_if_articulation_separates_child()
    report = ctx.report()
    assert not report.passed
    assert "lid_hinge" in report.failures[0].details
    assert report.failures[0].kind is FailureKind.ARTICULATION_SEPARATION


def test_separation_check_ignores_prismatic_liftoff() -> None:
    # A prismatic lift-off is meant to separate; it must not be flagged.
    model = RigidBodyAssembly("liftoff")
    base = model.rigid_body("base")
    base.add(BoxGeometry((0.10, 0.10, 0.02)), name="base_slab")
    body = model.rigid_body("body")
    body.add(BoxGeometry((0.08, 0.08, 0.10)).translate(0.0, 0.0, 0.06), name="body_box")
    model.joint(
        "lift",
        body0=base,
        frame0=JointFrame(),
        body1=body,
        frame1=JointFrame(),
        dofs=(JointDOF(JointAxis.TRANS_Z, limits=(0.0, 0.10)),),
    )
    ctx = TestContext(model)
    ctx.fail_if_articulation_separates_child()
    assert ctx.report().passed
