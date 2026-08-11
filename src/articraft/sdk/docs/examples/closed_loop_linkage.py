"""A four-bar linkage: the mechanism a tree of joints cannot describe.

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

from articraft.sdk import (
    BoxGeometry,
    JointAxis,
    JointDOF,
    JointFrame,
    Material,
    RigidBodyAssembly,
    TestContext,
    TestReport,
)

SPAN = 0.120  # between the two ground pivots
RISE = 0.080  # crank length
BAR = 0.010  # bar thickness


def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("four_bar_linkage")

    def bar(name: str, length: float, upright: bool = False):
        body = model.rigid_body(name)
        size = (BAR, BAR, length) if upright else (length, BAR, BAR)
        offset = (0.0, 0.0, length / 2.0) if upright else (length / 2.0, 0.0, 0.0)
        body.add(
            BoxGeometry(size).translate(*offset),
            name="arm",
            material=Material.STEEL if name == "ground" else Material.ALUMINUM,
        )
        return body

    ground = bar("ground", SPAN)
    left_crank = bar("left_crank", RISE, upright=True)
    coupler = bar("coupler", SPAN)
    right_crank = bar("right_crank", RISE, upright=True)

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
