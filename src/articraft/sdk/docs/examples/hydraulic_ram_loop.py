"""A lever raised by a hydraulic ram: one closed loop, posed by one joint.

The ram is four bodies pinned in a ring: base -> arm, base -> barrel,
barrel -> rod, and the rod's eye pinned back onto the arm. A tree cannot carry
that last pin, so it is declared as one more articulation and becomes a loop
closure. The barrel and rod are not posed at all: each carries a ``drive`` that
solves its value from the arm, so posing ``arm_hinge`` moves the whole
mechanism and the ram can never detach.

Every joint here refers to the same two anchor constants. Deriving geometry,
drives, and the closing pin from shared constants is what keeps them agreeing.
"""

from __future__ import annotations

import math

from build123d import Box, Cylinder, Pos, Rot

from articraft.sdk import (
    AimAt,
    ArticulatedObject,
    ArticulationType,
    Material,
    MotionLimits,
    Origin,
    SpanTo,
    TestContext,
    TestReport,
)

# The two pins of the ram, each in the frame of the part that carries it.
ARM_EYE = (0.42, 0.0, -0.02)  # rod eye pin, in the arm's frame
BARREL_PIVOT = (0.28, 0.0, 0.05)  # barrel trunnion pin, in the base's frame
REST = 0.20  # gap the ram spans at stroke = 0, and the barrel's working length


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("ram_lever")

    base = model.part("base")
    base.add(Box(0.5, 0.24, 0.08), name="slab", material=Material.STEEL, color=(0.25, 0.28, 0.32))
    base.add(Pos(X=0.02, Z=0.1) * Box(0.08, 0.1, 0.3), name="post", material=Material.STEEL)

    arm = model.part("arm")
    arm.add(
        Pos(X=0.3) * Box(0.62, 0.07, 0.07),
        name="beam",
        material=Material.STEEL,
        color=(0.85, 0.55, 0.1),
    )

    # Barrel geometry along +X from its pivot; slightly shorter than REST so the
    # tube never pokes past the rod eye at full retraction.
    barrel = model.part("barrel")
    barrel.add(
        Rot(Y=90) * Pos(Z=0.09) * Cylinder(0.035, 0.18),
        name="tube",
        material=Material.STEEL,
        color=(0.85, 0.55, 0.1),
    )

    # The rod's eye sits at x = REST in its own frame: at stroke = 0 the eye
    # lands exactly on the arm's pin, which is what rest_length means below.
    rod = model.part("rod")
    rod.add(
        Rot(Y=90) * Pos(Z=0.10) * Cylinder(0.016, 0.18),
        name="shaft",
        material=Material.STEEL.but(roughness=0.15),
    )
    rod.add(
        Pos(X=REST) * Cylinder(0.03, 0.05, rotation=(90, 0, 0)),
        name="eye",
        material=Material.STEEL,
        color=(0.85, 0.55, 0.1),
    )

    # The tree: the first articulation declared for each child owns its edge.
    model.articulation(
        "arm_hinge",
        ArticulationType.REVOLUTE,
        base,
        arm,
        origin=Origin(xyz=(0.0, 0.0, 0.25)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.3, upper=0.05),
    )
    model.articulation(
        "barrel_pivot",
        ArticulationType.REVOLUTE,
        base,
        barrel,
        origin=Origin(xyz=BARREL_PIVOT),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-1.6, upper=1.6),
        # Not posed: the barrel swivels to stay aimed at the arm's eye pin.
        drive=AimAt("arm", ARM_EYE),
    )
    model.articulation(
        "stroke",
        ArticulationType.PRISMATIC,
        barrel,
        rod,
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=-0.05, upper=0.2),
        # Not posed: the rod is as long as the gap it must span.
        drive=SpanTo("arm", ARM_EYE, rest_length=REST),
    )
    # The loop closure: rod already has a parent, so this second articulation
    # reaching it closes the ring instead of re-parenting it. It exports as a
    # regular USD joint marked physics:excludeFromArticulation, which physics
    # engines enforce as the pin it is.
    model.articulation(
        "rod_eye_pin",
        ArticulationType.REVOLUTE,
        arm,
        rod,
        origin=Origin(xyz=ARM_EYE),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-3.0, upper=3.0),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.allow_overlap(
        "barrel",
        "rod",
        shape_a="tube",
        shape_b="shaft",
        reason="the rod slides inside the barrel bore",
    )

    # Sample the ONE posed joint and prove the loop stays closed everywhere:
    # the rod's eye must sit on the arm's pin at every sampled angle. This is
    # the check that catches a linkage drifting apart.
    poses = ctx.sample_joint("arm_hinge", samples=7)
    eye_path = ctx.track_point("rod", (REST, 0.0, 0.0), poses)
    pin_path = ctx.track_point("arm", ARM_EYE, poses)
    worst = max(math.dist(eye, pin) for eye, pin in zip(eye_path, pin_path, strict=True))
    ctx.check(
        "rod eye rides the arm pin through the sweep",
        worst < 1e-6,
        details=f"worst gap {worst * 1000:.4f} mm",
    )
    ctx.record_metric("ram_worst_eye_gap_m", worst)

    # The parts of the ram still meet: rod inside barrel at every pose.
    ctx.expect_contact_at_poses("barrel", "rod", poses)
    return ctx.report()
