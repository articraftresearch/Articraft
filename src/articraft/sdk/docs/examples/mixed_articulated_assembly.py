from __future__ import annotations

from build123d import Box

from articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    BoxGeometry,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("mixed_assembly")

    base = model.part("base")
    base.add(Box(0.30, 0.22, 0.10), name="plinth", color=(0.2, 0.22, 0.25))

    arm = model.part("arm")
    arm_mesh = BoxGeometry((0.04, 0.04, 0.20)).translate(0.0, 0.0, 0.10)
    arm.add(arm_mesh, name="upright", color=(0.78, 0.48, 0.12, 1.0))

    model.articulation(
        "base_to_arm",
        ArticulationType.REVOLUTE,
        base,
        arm,
        origin=Origin(xyz=(0.0, 0.0, 0.05)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.8, upper=0.8),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.expect_contact("base", "arm", shape_a="plinth", shape_b="upright")
    return ctx.report()
