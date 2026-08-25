from __future__ import annotations

import math
import re

import pytest
from build123d import Box

from articraft.sdk.assembly import JointAxis, JointDOF, JointFrame, RigidBodyAssembly
from articraft.sdk.errors import LoopClosureError
from articraft.sdk.testing import (
    TestContext,
    _articulation_sweep_values,
    _round_out,
    _swept_from_rest,
)

FULL = (-math.pi, math.pi)
GROUND_A = (-0.2, 0.0, 0.1)
GROUND_D = (0.25, 0.0, 0.1)
CRANK = 0.2
ROCKER = 0.25
POINT_B = (GROUND_A[0] + CRANK, 0.0, GROUND_A[2])
POINT_C = (GROUND_D[0], 0.0, GROUND_D[2] + ROCKER)
COUPLER = math.dist(POINT_B, POINT_C)
COUPLER_TILT = -math.atan2(POINT_C[2] - POINT_B[2], POINT_C[0] - POINT_B[0])
LINK = 0.32
SLIDE = (-0.4, 0.0)


def four_bar(*, coupler_limits=FULL, crank_limits=FULL, rocker_limits=FULL) -> RigidBodyAssembly:
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
        dofs=(JointDOF(JointAxis.ROT_Y, limits=rocker_limits),),
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
    """A crank driving a slider through a link, closed back onto the slider.

    The link is longer than the crank, so neither end of the stroke folds the
    pair onto a singular one-parameter family. The stroke stays (l + a) minus
    (l - a) = 2a = 0.4, matching SLIDE.
    """

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
        ground.at(JointFrame(xyz=(-0.25 + CRANK + LINK, 0.0, 0.0))),
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
    needed = re.search(r"needs at least \((-?[\d.]+), (-?[\d.]+)\)", details)
    assert needed is not None
    # The knee folds the way the limits forbid, whatever the sweep density is.
    assert float(needed.group(1)) < 0.0


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


@pytest.mark.parametrize("samples", [5, 7, 9, 13])
def test_a_hinge_with_slack_past_its_own_travel_is_not_a_contradiction(samples: int) -> None:
    # Walked from the rest pose, the far side of the ring arrives with the
    # crank at -2.23 rad, inside its limits -- not at the +4.05 a solve from
    # rest used to report for the same pose. The slack past the crank's own
    # travel stays quiet at every density: the overrun is a fraction of the
    # declared travel, not of however many samples landed on the walls.
    assert failure_details(four_bar(crank_limits=(-2.5, 2.5)), samples=samples) == ""


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


def test_the_sweep_stops_where_the_ring_stops_closing() -> None:
    # Driving the coupler pin, the four-bar's ring stops closing partway round
    # on one side. The walk from the rest pose ends there: the mechanism cannot
    # pass through a pose with no solution, so everything past it is counted
    # out of reach instead of being solved into a lifted copy of the linkage
    # with the crank a whole turn away.
    resolved = four_bar().resolve()
    joint = resolved.get_joint("coupler_pin").joint
    dof = joint.dofs[0]
    sweep = _articulation_sweep_values(joint, 33, dof)
    reached, _, span_reached, out_of_reach = _swept_from_rest(
        resolved, "coupler_pin.rotY", sweep, frozenset({"crank_pin.rotY", "rocker_pin.rotY"})
    )

    assert out_of_reach > 0
    assert span_reached < sweep[-1] - sweep[0]
    assert all(abs(free["crank_pin.rotY"]) < 2.5 for _, free in reached)
    # The kept poses are one unbroken stretch of driver travel around rest.
    index_of = {value: index for index, value in enumerate(sweep)}
    kept = sorted(index_of[value] for value, _ in reached)
    assert kept == list(range(kept[0], kept[-1] + 1))


def double_four_bar(*, second_coupler_limits=FULL) -> RigidBodyAssembly:
    """Two identical four-bar rings driven by one shared crank."""

    assembly = RigidBodyAssembly("double_four_bar")
    ground = assembly.rigid_body("ground")
    crank = assembly.rigid_body("crank")
    for body, length in ((ground, 0.5), (crank, CRANK)):
        body.add(Box(max(length, 0.05), 0.04, 0.04), name="bar")
    joints = [
        assembly.joint(
            "crank_pin",
            ground.at(JointFrame(xyz=GROUND_A)),
            crank.at(),
            dofs=(JointDOF(JointAxis.ROT_Y, limits=FULL),),
        )
    ]
    for tag, coupler_limits in (("a", FULL), ("b", second_coupler_limits)):
        coupler = assembly.rigid_body(f"coupler_{tag}")
        rocker = assembly.rigid_body(f"rocker_{tag}")
        for body, length in ((coupler, COUPLER), (rocker, ROCKER)):
            body.add(Box(max(length, 0.05), 0.04, 0.04), name="bar")
        joints.append(
            assembly.joint(
                f"coupler_pin_{tag}",
                crank.at(JointFrame(xyz=(CRANK, 0.0, 0.0), rpy=(0.0, COUPLER_TILT, 0.0))),
                coupler.at(),
                dofs=(JointDOF(JointAxis.ROT_Y, limits=coupler_limits),),
            )
        )
        joints.append(
            assembly.joint(
                f"rocker_pin_{tag}",
                ground.at(JointFrame(xyz=GROUND_D, rpy=(0.0, -math.pi / 2.0, 0.0))),
                rocker.at(),
                dofs=(JointDOF(JointAxis.ROT_Y, limits=FULL),),
            )
        )
        assembly.joint(
            f"closing_pin_{tag}",
            coupler.at(JointFrame(xyz=(COUPLER, 0.0, 0.0))),
            rocker.at(JointFrame(xyz=(ROCKER, 0.0, 0.0))),
            dofs=(JointDOF(JointAxis.ROT_Y, limits=FULL),),
        )
    assembly.articulation("main", root=ground, joints=tuple(joints))
    return assembly


def test_two_rings_sharing_a_driver_stay_silent_together() -> None:
    # The shared crank sits on both rings' paths, so it is planned once and
    # an honest pair of rings reports nothing.
    assert failure_details(double_four_bar()) == ""


def test_a_defect_on_the_second_ring_is_still_found() -> None:
    details = failure_details(double_four_bar(second_coupler_limits=(0.0, 1.75)))

    assert "follower='coupler_pin_b.rotY'" in details


def test_findings_come_out_in_a_stable_order() -> None:
    details = failure_details(four_bar(coupler_limits=(0.0, 1.75), rocker_limits=(-0.5, 0.5)))

    coupler = details.index("follower='coupler_pin.rotY'")
    rocker = details.index("follower='rocker_pin.rotY'")
    assert coupler < rocker


def test_a_closure_the_tree_does_not_span_is_passed_over() -> None:
    # A closure to a body outside the articulation tree has no driver the
    # sweep could turn, so the check records a pass instead of guessing.
    assembly = RigidBodyAssembly("lidded")
    box = assembly.rigid_body("box")
    lid = assembly.rigid_body("lid")
    latch = assembly.rigid_body("latch")
    box.add(Box(0.2, 0.2, 0.1), name="shell")
    lid.add(Box(0.2, 0.2, 0.01), name="panel")
    latch.add(Box(0.04, 0.04, 0.04), name="block")
    hinge = assembly.joint(
        "hinge",
        box.at(JointFrame(xyz=(0.0, 0.1, 0.05))),
        lid.at(),
        dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 1.9)),),
    )
    assembly.joint(
        "latch_pin",
        lid.at(JointFrame(xyz=(0.1, 0.0, 0.0))),
        latch.at(),
        dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 0.5)),),
    )
    assembly.articulation("main", root=box, joints=(hinge,))
    context = TestContext(assembly)

    assert context.fail_if_loop_limits_contradict() is True


def test_a_branch_answer_is_reported_next_to_the_jam() -> None:
    # Poses the limits leave unsolvable make the case. Where the bounded solve
    # answers from a different assembly branch instead, that is reported with
    # them, whole turns removed before the distance is read.
    details = failure_details(four_bar(coupler_limits=(0.0, 1.75)))

    assert "poses unsolvable" in details
    assert "away in joint coordinates -- a different assembly branch" in details


@pytest.mark.parametrize(
    "value",
    [3.3457, -1.8694, 0.9999999999, 1.0000000001, -0.1, 1e-5, 12345.678, -3.0],
)
def test_needs_bounds_round_outward(value: float) -> None:
    # The printed range must survive being copied back: parsing the four shown
    # digits has to give an interval that still contains the measured value.
    lower = float(f"{_round_out(value, up=False):.4g}")
    upper = float(f"{_round_out(value, up=True):.4g}")

    assert lower <= value <= upper
