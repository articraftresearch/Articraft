from __future__ import annotations

import math

import pytest
from build123d import Box

from articraft.sdk.assembly import JointAxis, JointDOF, JointFrame, RigidBodyAssembly
from articraft.sdk.errors import LoopClosureError
from articraft.sdk.testing import TestContext

FULL = (-math.pi, math.pi)
GROUND_A = (-0.2, 0.0, 0.1)
GROUND_D = (0.25, 0.0, 0.1)
CRANK = 0.2
ROCKER = 0.25
POINT_B = (GROUND_A[0] + CRANK, 0.0, GROUND_A[2])
POINT_C = (GROUND_D[0], 0.0, GROUND_D[2] + ROCKER)
COUPLER = math.dist(POINT_B, POINT_C)
COUPLER_TILT = -math.atan2(POINT_C[2] - POINT_B[2], POINT_C[0] - POINT_B[0])
LINK = 0.2
SLIDE = (-0.4, 0.0)


def four_bar(*, coupler_limits=FULL, crank_limits=FULL) -> RigidBodyAssembly:
    assembly = RigidBodyAssembly("four_bar")
    ground = assembly.rigid_body("ground")
    crank = assembly.rigid_body("crank")
    coupler = assembly.rigid_body("coupler")
    rocker = assembly.rigid_body("rocker")
    for body, length in ((ground, 0.5), (crank, CRANK), (coupler, COUPLER), (rocker, ROCKER)):
        body.add(Box(max(length, 0.05), 0.04, 0.04), name="bar")
    crank_pin = assembly.joint(
        "crank_pin",
        ground.at(JointFrame(xyz=GROUND_A)),
        crank.at(),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=crank_limits),),
    )
    coupler_pin = assembly.joint(
        "coupler_pin",
        crank.at(JointFrame(xyz=(CRANK, 0.0, 0.0), rpy=(0.0, COUPLER_TILT, 0.0))),
        coupler.at(),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=coupler_limits),),
    )
    rocker_pin = assembly.joint(
        "rocker_pin",
        ground.at(JointFrame(xyz=GROUND_D, rpy=(0.0, -math.pi / 2.0, 0.0))),
        rocker.at(),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=FULL),),
    )
    assembly.joint(
        "closing_pin",
        coupler.at(JointFrame(xyz=(COUPLER, 0.0, 0.0))),
        rocker.at(JointFrame(xyz=(ROCKER, 0.0, 0.0))),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=FULL),),
    )
    assembly.articulation("main", root=ground, joints=(crank_pin, coupler_pin, rocker_pin))
    return assembly


def slider_crank(*, slide_limits=SLIDE) -> RigidBodyAssembly:
    """A crank driving a slider through a link, closed back onto the slider."""

    assembly = RigidBodyAssembly("slider_crank")
    ground = assembly.rigid_body("ground")
    crank = assembly.rigid_body("crank")
    link = assembly.rigid_body("link")
    slider = assembly.rigid_body("slider")
    for body, length in ((ground, 0.6), (crank, CRANK), (link, LINK), (slider, 0.08)):
        body.add(Box(max(length, 0.05), 0.04, 0.04), name="bar")
    crank_pin = assembly.joint(
        "crank_pin",
        ground.at(JointFrame(xyz=(-0.25, 0.0, 0.0))),
        crank.at(),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=FULL),),
    )
    link_pin = assembly.joint(
        "link_pin",
        crank.at(JointFrame(xyz=(CRANK, 0.0, 0.0))),
        link.at(),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=FULL),),
    )
    guide = assembly.joint(
        "guide",
        ground.at(JointFrame(xyz=(0.15, 0.0, 0.0))),
        slider.at(),
        dofs=(JointDOF(JointAxis.TRANS_X, limits=slide_limits),),
    )
    assembly.joint(
        "closing_pin",
        link.at(JointFrame(xyz=(LINK, 0.0, 0.0))),
        slider.at(),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=FULL),),
    )
    assembly.articulation("main", root=ground, joints=(crank_pin, link_pin, guide))
    return assembly


def hinge_only() -> RigidBodyAssembly:
    assembly = RigidBodyAssembly("lid")
    box = assembly.rigid_body("box")
    lid = assembly.rigid_body("lid")
    box.add(Box(0.2, 0.2, 0.1), name="shell")
    lid.add(Box(0.2, 0.2, 0.01), name="panel")
    hinge = assembly.joint(
        "hinge",
        box.at(JointFrame(xyz=(0.0, 0.1, 0.05))),
        lid.at(),
        dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 1.9)),),
    )
    assembly.articulation("main", root=box, joints=(hinge,))
    return assembly


def failure_details(model: RigidBodyAssembly, **kwargs: object) -> str:
    context = TestContext(model)
    passed = context.fail_if_loop_limits_contradict(**kwargs)  # type: ignore[arg-type]
    report = context.report()
    if passed:
        return ""
    return "\n".join(failure.details for failure in report.failures)


def test_a_ring_that_agrees_with_its_limits_reports_nothing() -> None:
    assert failure_details(four_bar()) == ""


def test_a_follower_limit_pointing_the_wrong_way_is_named() -> None:
    details = failure_details(four_bar(coupler_limits=(0.0, 1.75)))

    assert "follower='coupler_pin.rotY'" in details
    assert "declared=(0, 1.75)" in details
    assert "needs at least (-1.8" in details


def test_a_follower_limit_narrower_than_the_ring_is_named() -> None:
    details = failure_details(four_bar(coupler_limits=(-1.0, 1.0)))

    assert "follower='coupler_pin.rotY'" in details
    assert "declared=(-1, 1)" in details


@pytest.mark.parametrize("samples", [5, 9, 17, 33])
def test_the_defect_is_found_at_every_sweep_density(samples: int) -> None:
    details = failure_details(four_bar(coupler_limits=(0.0, 1.75)), samples=samples)

    assert "follower='coupler_pin.rotY'" in details


def test_a_driver_range_wider_than_the_linkage_is_named() -> None:
    details = failure_details(slider_crank(slide_limits=(-0.05, 0.05)))

    assert "driver='guide.transX'" in details
    assert "wider than the mechanism" in details


@pytest.mark.parametrize("slack", [0.0, 0.02, 0.05])
def test_slack_on_a_bounded_coordinate_is_not_a_defect(slack: float) -> None:
    assert failure_details(slider_crank(slide_limits=(SLIDE[0] - slack, SLIDE[1] + slack))) == ""


def test_a_full_circle_hinge_makes_no_claim() -> None:
    # The four-bar cannot turn any of its pins the whole way round, and its
    # pins are all authored -pi..pi. That is not a range claim, so nothing is
    # reported -- but the linkage really does refuse those poses.
    resolved = four_bar().resolve()
    with pytest.raises(LoopClosureError):
        resolved.forward_kinematics({"crank_pin.rotY": math.pi})

    assert failure_details(four_bar()) == ""


def test_a_hinge_that_needs_the_other_turn_is_not_a_contradiction() -> None:
    # Driving from the far side of the ring solves the crank to +4.05 rad, the
    # same pose as -2.23 and inside its limits. One turn away is not outside.
    assert failure_details(four_bar(crank_limits=(-2.5, 2.5))) == ""


def test_an_assembly_without_a_loop_is_skipped() -> None:
    context = TestContext(hinge_only())

    assert context.fail_if_loop_limits_contradict() is True
    assert context.report().checks_run == 1


def test_the_sweep_warns_about_coordinates_it_could_not_drive() -> None:
    context = TestContext(slider_crank(slide_limits=(-0.05, 0.05)))
    context.fail_if_loop_limits_contradict(samples=9, max_solves=18)
    warnings = context.report().warnings

    assert any("not swept as drivers" in warning for warning in warnings)


def test_forward_kinematics_still_validates_what_the_solver_returns() -> None:
    # The relaxed solve is private for this reason: it can answer with a
    # coordinate outside its own limits, and the public path must not.
    resolved = four_bar(coupler_limits=(0.0, 1.75)).resolve()
    positions, _ = resolved._kinematics({"crank_pin.rotY": 0.25}, relax_limits=True)

    assert positions["coupler_pin.rotY"] < 0.0
    with pytest.raises(LoopClosureError):
        resolved.forward_kinematics({"crank_pin.rotY": 0.25})
