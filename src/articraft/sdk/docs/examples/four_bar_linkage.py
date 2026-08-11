"""A four bar linkage: a fixed length link pinned at both ends.

This is the other half of the loop vocabulary. A hydraulic ram *changes length*,
so its two joints are solved in closed form by drives -- see
``hydraulic_ram_loop.py``. A pitman arm, a pull rod, a drag brace, a hood hinge
link or a scissor arm does not change length: it is one rigid part pinned at
both ends, and no formula gives its angle. Declare the second pin as one more
articulation and it becomes a loop closure. Posing the crank then swings the
whole ring, because the follower angles are solved from the pin.

The mistake this example exists to prevent is reaching for ``AimAt`` here.
``AimAt`` points a part at an anchor; it never makes the part *reach* it. A link
that only aims ends up hanging in space with its far eye nowhere near the pin it
is supposed to hold, and every check still passes.

Layout, all in metres: ground A--D, crank A--B, coupler B--C, rocker D--C.
"""

from __future__ import annotations

import math

from build123d import Box, Cylinder, Pos, Rot

from articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    Material,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
)

GROUND_A = (-0.20, 0.0, 0.10)  # crank pin on the ground link
GROUND_D = (0.25, 0.0, 0.10)  # rocker pin on the ground link
CRANK = 0.20  # A to B
ROCKER = 0.25  # D to C

POINT_B = (GROUND_A[0] + CRANK, 0.0, GROUND_A[2])
POINT_C = (GROUND_D[0], 0.0, GROUND_D[2] + ROCKER)
COUPLER = math.dist(POINT_B, POINT_C)  # B to C: the fixed length link
# Aim the coupler's own +X down its length, so its far end sits at (COUPLER, 0, 0).
COUPLER_TILT = -math.atan2(POINT_C[2] - POINT_B[2], POINT_C[0] - POINT_B[0])

STEEL = Material.STEEL.but(roughness=0.35)


def _link(part, length: float, radius: float = 0.018) -> None:
    """A round bar along +X from the part's own origin, with an eye at each end."""

    part.add(Rot(Y=90) * Pos(Z=length / 2) * Cylinder(radius, length), name="bar", material=STEEL)
    for name, x in (("near_eye", 0.0), ("far_eye", length)):
        part.add(
            Pos(X=x) * Cylinder(radius * 1.8, radius * 1.6, rotation=(90, 0, 0)),
            name=name,
            material=STEEL,
        )


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("four_bar_linkage")

    ground = model.part("ground")
    ground.add(Box(0.62, 0.06, 0.05), name="bed", material=STEEL, color=(0.24, 0.26, 0.30))
    for name, anchor in (("crank_post", GROUND_A), ("rocker_post", GROUND_D)):
        ground.add(
            Pos(X=anchor[0], Z=anchor[2] / 2) * Box(0.05, 0.05, anchor[2]),
            name=name,
            material=STEEL,
            color=(0.24, 0.26, 0.30),
        )

    crank = model.part("crank")
    _link(crank, CRANK, radius=0.022)

    coupler = model.part("coupler")
    _link(coupler, COUPLER)

    rocker = model.part("rocker")
    _link(rocker, ROCKER, radius=0.020)

    # The tree: three joints, each the first to reach its child.
    model.articulation(
        "crank_pin",
        ArticulationType.REVOLUTE,
        ground,
        crank,
        origin=Origin(xyz=GROUND_A),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.55, upper=0.55),
    )
    model.articulation(
        "coupler_pin",
        ArticulationType.REVOLUTE,
        crank,
        coupler,
        # The coupler hangs off the crank's far end, tilted to point at C.
        origin=Origin(xyz=(CRANK, 0.0, 0.0), rpy=(0.0, COUPLER_TILT, 0.0)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-1.5, upper=1.5),
    )
    model.articulation(
        "rocker_pin",
        ArticulationType.REVOLUTE,
        ground,
        rocker,
        # Rotated so the rocker's own +X points up at C.
        origin=Origin(xyz=GROUND_D, rpy=(0.0, -math.pi / 2, 0.0)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-1.5, upper=1.5),
    )
    # The closure: the coupler's far end is pinned to the rocker's far end. The
    # rocker already has a parent, so this second articulation reaching it closes
    # the ring rather than re-parenting it. No drive belongs here: the coupler is
    # a fixed length, and its angle has no closed form.
    model.articulation(
        "closing_pin",
        ArticulationType.REVOLUTE,
        coupler,
        rocker,
        origin=Origin(xyz=(COUPLER, 0.0, 0.0)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-2.0, upper=2.0),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)

    # Pose only the crank. The coupler and rocker are followers: their angles
    # are solved so the closing pin stays shut, which is what makes this a
    # mechanism rather than three loose bars.
    poses = ctx.sample_joint("crank_pin", samples=7)

    # The far end of the coupler and the far end of the rocker are the same
    # physical pin, so they must ride together everywhere in the sweep.
    ctx.expect_distance_at_poses(
        "coupler",
        "rocker",
        poses,
        shape_a="far_eye",
        shape_b="far_eye",
        maximum=0.002,
        name="closing pin stays shut",
    )

    # A frozen mechanism would also pass the check above, so prove the rocker
    # actually swings, and that its far end covers real ground.
    swept = ctx.track_point("rocker", (ROCKER, 0.0, 0.0), poses)
    travel = max(math.dist(swept[0], point) for point in swept)
    ctx.check(
        "the rocker swings through the crank's range",
        travel > 0.05,
        details=f"rocker end travelled {travel * 1000:.0f} mm",
    )
    ctx.record_metric("rocker_end_travel_m", travel)
    return ctx.report()
