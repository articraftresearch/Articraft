"""Canonical articulated example: a box whose lid opens on a real hinge.

Two parts, one REVOLUTE articulation, and an authored contact check. The
hinge axis sits exactly on the edge where the lid meets the base, so the
parts stay in contact through the whole motion range -- the property the
compiler's articulation-separation check verifies.

Compile it:  python -m mini_articraft.environments.worker <run_dir>
"""

from __future__ import annotations

from build123d import Box

from mini_articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    Material,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("hinged_box")

    base = model.part("base")
    # Saying what a shape is made of settles its mass, how it behaves on contact,
    # and how it looks. Aluminum comes out light and metallic without asking.
    base.add(Box(0.10, 0.08, 0.04), name="body", material=Material.ALUMINUM)

    lid = model.part("lid")
    lid.add(
        # Part geometry is authored in the part's LOCAL frame; the hinge
        # origin (0, -0.04, 0.02) maps it into the parent. Local (0, 0.04,
        # 0.007) therefore lands the lid at world (0, 0, 0.027): its knuckle
        # edge sits 0.5 mm into the base, a small designed embed that keeps
        # the parts physically connected (declared below with allow_overlap).
        Box(0.10, 0.08, 0.015).translate((0.0, 0.04, 0.007)),
        name="body",
        # A recolor keeps the material -- still plastic, still 1050 kg/m^3 --
        # and only changes what it looks like: an amber lid on a metal base.
        material=Material.ABS_PLASTIC,
        color=(0.62, 0.45, 0.16),
    )

    model.articulation(
        "lid_hinge",
        ArticulationType.REVOLUTE,
        base,
        lid,
        # The hinge line is the lid/base contact edge: rotating around it
        # keeps the parts touching instead of pulling the lid off the box.
        origin=Origin(xyz=(0.0, -0.04, 0.02)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=0.0, upper=1.5708),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.allow_overlap(
        "base",
        "lid",
        reason="hinge knuckle embedded in the base",
        shape_a="body",
        shape_b="body",
    )
    ctx.expect_contact("base", "lid")
    return ctx.report()
