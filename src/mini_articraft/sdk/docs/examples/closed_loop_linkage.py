"""A four-bar linkage: the case a tree of joints cannot describe.

Count the pivots. `coupler` is pinned to `left_crank` at one end and to
`right_crank` at the other, and both cranks are pinned to `ground`. That is a
ring of four joints around four bodies, so one joint has to close it.

Author every joint the mechanism physically has, then name the spanning tree in
`model.articulation(...)`. The joint left out is exported with
`excludeFromArticulation`, and a simulator solves it as an ordinary constraint.

Leaving that joint out entirely is the mistake this example exists to prevent:
the parts still export, and then flap loose the moment anything touches them.
"""

from __future__ import annotations

from mini_articraft.sdk import (
    BoxGeometry,
    JointAxis,
    JointDOF,
    JointFrame,
    Material,
    RigidBodyAssembly,
    TestContext,
    TestReport,
)

SPAN = 0.120  # distance between the two ground pivots
RISE = 0.080  # crank length
BAR = 0.010  # bar thickness


def _bar_along_x(length: float) -> BoxGeometry:
    """A bar whose local origin sits on its first pivot."""
    return BoxGeometry((length, BAR, BAR)).translate(length / 2.0, 0.0, 0.0)


def _bar_along_z(length: float) -> BoxGeometry:
    return BoxGeometry((BAR, BAR, length)).translate(0.0, 0.0, length / 2.0)


def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("four_bar_linkage")

    ground = model.rigid_body("ground")
    ground.add(_bar_along_x(SPAN), name="frame", material=Material.STEEL)

    left_crank = model.rigid_body("left_crank")
    left_crank.add(_bar_along_z(RISE), name="arm", material=Material.ALUMINUM)

    coupler = model.rigid_body("coupler")
    coupler.add(_bar_along_x(SPAN), name="arm", material=Material.ALUMINUM)

    right_crank = model.rigid_body("right_crank")
    right_crank.add(_bar_along_z(RISE), name="arm", material=Material.ALUMINUM)

    swing = (JointDOF(JointAxis.ROT_Y, limits=(-0.6, 0.6)),)

    # Around the ring. Each pair of frames coincides at rest, so the rectangle
    # closes exactly in the authored pose.
    model.joint(
        "ground_left",
        body0=ground,
        frame0=JointFrame(),
        body1=left_crank,
        frame1=JointFrame(),
        dofs=swing,
    )
    model.joint(
        "left_coupler",
        body0=left_crank,
        frame0=JointFrame(xyz=(0.0, 0.0, RISE)),
        body1=coupler,
        frame1=JointFrame(),
        dofs=swing,
    )
    model.joint(
        "coupler_right",
        body0=coupler,
        frame0=JointFrame(xyz=(SPAN, 0.0, 0.0)),
        body1=right_crank,
        frame1=JointFrame(xyz=(0.0, 0.0, RISE)),
        dofs=swing,
    )
    # The fourth joint closes the ring. It is real, and it is left out of the
    # articulation below.
    model.joint(
        "ground_right",
        body0=ground,
        frame0=JointFrame(xyz=(SPAN, 0.0, 0.0)),
        body1=right_crank,
        frame1=JointFrame(),
        dofs=swing,
    )

    model.articulation(
        "main",
        root=ground,
        joints=["ground_left", "left_coupler", "coupler_right"],
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    resolved = object_model.resolve()

    ctx.check(
        "linkage_closes_a_loop",
        resolved.has_closed_loops,
        "a four-bar is a ring, not a chain",
    )
    excluded = [item.joint.name for item in resolved.joints if item.exclude_from_articulation]
    ctx.check(
        "one_joint_closes_the_ring",
        excluded == ["ground_right"],
        f"expected ground_right to be the loop closer, found {excluded}",
    )
    return ctx.report()
