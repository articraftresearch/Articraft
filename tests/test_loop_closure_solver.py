"""Loops with no closed form are solved numerically when a pose is placed.

Drives handle a ram, whose geometry is a triangle. A four bar has no such
identity, so its follower joints are solved from the pin instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box

from articraft.sdk import ArticulatedObject, ArticulationType, MotionLimits, Origin
from articraft.sdk._collision import MeshCollisionKernel, _origin_matrix
from articraft.sdk.errors import LoopClosureError
from articraft.sdk.joints import partition_articulations


def _bar(model: ArticulatedObject, name: str, length: float):
    part = model.part(name)
    part.add(Box(max(length, 0.05), 0.04, 0.04), name="bar")
    return part


# A planar four bar, laid out so the closing pin sits at the rocker's far end:
# ground A--D, crank A--B, coupler B--C, rocker D--C.
GROUND_A = (-0.2, 0.0, 0.1)
GROUND_D = (0.25, 0.0, 0.1)
CRANK = 0.2
ROCKER = 0.25
POINT_B = (GROUND_A[0] + CRANK, 0.0, GROUND_A[2])
POINT_C = (GROUND_D[0], 0.0, GROUND_D[2] + ROCKER)
COUPLER = math.dist(POINT_B, POINT_C)
COUPLER_TILT = -math.atan2(POINT_C[2] - POINT_B[2], POINT_C[0] - POINT_B[0])


def _four_bar(*, coupler: float = COUPLER) -> ArticulatedObject:
    model = ArticulatedObject("four_bar")
    ground = _bar(model, "ground", 0.5)
    crank = _bar(model, "crank", CRANK)
    coupler_part = _bar(model, "coupler", coupler)
    rocker_part = _bar(model, "rocker", ROCKER)
    model.articulation(
        "crank_pin",
        ArticulationType.REVOLUTE,
        ground,
        crank,
        origin=Origin(xyz=GROUND_A),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.6, upper=0.6),
    )
    model.articulation(
        "coupler_pin",
        ArticulationType.REVOLUTE,
        crank,
        coupler_part,
        origin=Origin(xyz=(CRANK, 0.0, 0.0), rpy=(0.0, COUPLER_TILT, 0.0)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-1.5, upper=1.5),
    )
    model.articulation(
        "rocker_pin",
        ArticulationType.REVOLUTE,
        ground,
        rocker_part,
        origin=Origin(xyz=GROUND_D, rpy=(0.0, -math.pi / 2, 0.0)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-1.5, upper=1.5),
    )
    model.articulation(
        "closing_pin",
        ArticulationType.REVOLUTE,
        coupler_part,
        rocker_part,
        origin=Origin(xyz=(coupler, 0.0, 0.0)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-2.0, upper=2.0),
    )
    model.validate()
    return model


def _worst_pin_gap(model: ArticulatedObject, poses: list[dict[str, float]]) -> float:
    kernel = MeshCollisionKernel(model, mesh_tolerance=0.002)
    _tree, loops = partition_articulations(model.articulations)
    rest = kernel.world_transforms({})
    worst = 0.0
    for pose in poses:
        transforms = kernel.world_transforms(pose)
        for closure in loops:
            pin_rest = (rest[closure.parent] @ _origin_matrix(closure.origin))[:3, 3]
            local = np.linalg.inv(rest[closure.child]) @ np.array([*pin_rest, 1.0])
            pin = (transforms[closure.parent] @ _origin_matrix(closure.origin))[:3, 3]
            worst = max(worst, float(np.linalg.norm(pin - (transforms[closure.child] @ local)[:3])))
    return worst


def _rocker_tip(kernel: MeshCollisionKernel, pose: dict[str, float]) -> np.ndarray:
    transforms = kernel.world_transforms(pose)
    return (transforms["rocker"] @ np.array([ROCKER, 0.0, 0.0, 1.0]))[:3]


def test_four_bar_stays_closed_across_its_crank() -> None:
    model = _four_bar()
    poses = [{"crank_pin": float(value)} for value in np.linspace(-0.55, 0.55, 15)]
    assert _worst_pin_gap(model, poses) < 1e-6


def test_the_follower_actually_swings() -> None:
    """A loop frozen in place would also score a zero gap."""

    kernel = MeshCollisionKernel(_four_bar(), mesh_tolerance=0.002)
    swept = [_rocker_tip(kernel, {"crank_pin": angle}) for angle in (-0.5, 0.5)]
    assert float(np.linalg.norm(swept[0] - swept[1])) > 0.15


def test_the_coupler_traces_a_curve_no_single_hinge_could() -> None:
    """The point of a four bar: the coupler translates while it rotates.

    Its pinned end rides the crank's circle, but its far end sweeps a coupler
    curve, covering very different distances for equal crank steps. A single
    hinge could only ever draw a constant radius arc.
    """

    kernel = MeshCollisionKernel(_four_bar(), mesh_tolerance=0.002)
    path = [
        (
            kernel.world_transforms({"crank_pin": float(angle)})["coupler"]
            @ np.array([COUPLER, 0.0, 0.0, 1.0])
        )[:3]
        for angle in np.linspace(-0.5, 0.5, 5)
    ]
    spans = [float(np.linalg.norm(path[index + 1] - path[index])) for index in range(4)]
    assert min(spans) > 0.005
    assert max(spans) / min(spans) > 2.0


def test_whatever_the_caller_poses_leads() -> None:
    """Pose the rocker instead and the crank becomes the follower."""

    model = _four_bar()
    assert _worst_pin_gap(model, [{"rocker_pin": -0.2}]) < 1e-6
    kernel = MeshCollisionKernel(model, mesh_tolerance=0.002)
    driven_by_rocker = kernel.world_transforms({"rocker_pin": -0.2})["crank"]
    assert not np.allclose(driven_by_rocker, kernel.world_transforms({})["crank"])


def test_solutions_are_deterministic_and_history_free() -> None:
    model = _four_bar()
    fresh = MeshCollisionKernel(model, mesh_tolerance=0.002)
    used = MeshCollisionKernel(model, mesh_tolerance=0.002)
    used.world_transforms({"crank_pin": -0.55})
    used.world_transforms({"crank_pin": 0.55})
    assert np.allclose(
        fresh.world_transforms({"crank_pin": 0.1})["rocker"],
        used.world_transforms({"crank_pin": 0.1})["rocker"],
    )


def test_unreachable_pose_is_an_error_not_a_broken_placement() -> None:
    """Swung past where the linkage can follow, no assembly exists: say so.

    The bound is real: with this geometry the rocker pin can be carried until
    the crank and coupler are stretched into a line, and no further.
    """

    kernel = MeshCollisionKernel(_four_bar(), mesh_tolerance=0.002)
    with pytest.raises(LoopClosureError, match="cannot reach this pose"):
        kernel.world_transforms({"crank_pin": 2.6})


def test_the_reachable_bound_matches_the_geometry() -> None:
    """Where the solver gives up should be where the linkage actually binds."""

    kernel = MeshCollisionKernel(_four_bar(), mesh_tolerance=0.002)
    span = COUPLER + CRANK
    for angle in (0.1, -0.1, -0.3):
        offset = np.array([ROCKER * math.sin(angle), 0.0, ROCKER * math.cos(angle)])
        reach = float(np.linalg.norm(np.array(GROUND_D) + offset - np.array(GROUND_A)))
        assert reach <= span  # analytically reachable
        kernel.world_transforms({"rocker_pin": angle})  # so the solver must place it

    offset = np.array([ROCKER * math.sin(0.25), 0.0, ROCKER * math.cos(0.25)])
    reach = float(np.linalg.norm(np.array(GROUND_D) + offset - np.array(GROUND_A)))
    assert reach > span  # analytically out of reach
    with pytest.raises(LoopClosureError):
        kernel.world_transforms({"rocker_pin": 0.25})


def test_models_without_loops_are_untouched() -> None:
    model = ArticulatedObject("plain")
    base = _bar(model, "base", 0.3)
    arm = _bar(model, "arm", 0.3)
    model.articulation(
        "hinge",
        ArticulationType.REVOLUTE,
        base,
        arm,
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-1.0, upper=1.0),
    )
    kernel = MeshCollisionKernel(model, mesh_tolerance=0.002)
    candidates, targets = kernel._loop_solution_plan()
    assert candidates == [] and targets == []
    assert np.allclose(
        kernel.world_transforms({"hinge": 0.4})["arm"],
        kernel._place({"hinge": 0.4})["arm"],
    )
