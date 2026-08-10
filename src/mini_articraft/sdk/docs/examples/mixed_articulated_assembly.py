from __future__ import annotations

from build123d import Box

from mini_articraft.sdk import (
    BoxGeometry,
    JointAxis,
    JointDOF,
    JointFrame,
    RigidBodyAssembly,
    TestContext,
    TestReport,
)


def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("mixed_assembly")

    base = model.rigid_body("base")
    base.add(Box(0.30, 0.22, 0.10), name="plinth", color=(0.2, 0.22, 0.25))

    arm = model.rigid_body("arm")
    arm_mesh = BoxGeometry((0.04, 0.04, 0.20)).translate(0.0, 0.0, 0.10)
    arm.add(arm_mesh, name="upright", color=(0.78, 0.48, 0.12, 1.0))

    # The frames meet: the joint sits on top of the plinth, and at the arm's foot.
    model.joint(
        "base_to_arm",
        body0=base,
        frame0=JointFrame(xyz=(0.0, 0.0, 0.05)),
        body1=arm,
        frame1=JointFrame(xyz=(0.0, 0.0, 0.0)),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=(-0.8, 0.8)),),
    )
    model.articulation("main", root=base, joints=["base_to_arm"])
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.expect_contact("base", "arm", shape_a="plinth", shape_b="upright")
    return ctx.report()
