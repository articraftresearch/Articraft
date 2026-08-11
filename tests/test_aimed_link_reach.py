"""An aimed link must be pinned, not merely pointed.

``AimAt`` turns a part to face an anchor. That is right for the barrel of a ram,
whose rod extends the rest of the way, and wrong for a fixed length link -- a
pitman arm, a pull rod -- which needs a loop closure. Reaching for the first
where the second was needed leaves the link hanging in space, and every other
check passes.
"""

from __future__ import annotations

import math

import numpy as np
from build123d import Box, Cylinder, Pos, Rot

from articraft.sdk import (
    AimAt,
    ArticulatedObject,
    ArticulationType,
    MotionLimits,
    Origin,
    Part,
    SpanTo,
    TestContext,
)

ANCHOR = (0.30, 0.0, 0.0)  # on the rocking beam, in its own frame


def _beam_and_crank() -> tuple[ArticulatedObject, Part, Part]:
    model = ArticulatedObject("linkage")
    base = model.part("base")
    base.add(Box(0.4, 0.1, 0.05), name="bed")
    beam = model.part("beam")
    beam.add(Pos(X=0.15) * Box(0.34, 0.05, 0.05), name="beam")
    crank = model.part("crank")
    crank.add(Pos(X=0.06) * Box(0.14, 0.04, 0.04), name="web")
    model.articulation(
        "beam_rocker",
        ArticulationType.REVOLUTE,
        base,
        beam,
        origin=Origin(xyz=(-0.1, 0.0, 0.25)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.35, upper=0.35),
    )
    model.articulation(
        "crank_rotation",
        ArticulationType.REVOLUTE,
        base,
        crank,
        origin=Origin(xyz=(0.12, 0.0, 0.05)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.6, upper=0.6),
    )
    return model, crank, beam


def _link_only_aimed() -> ArticulatedObject:
    """The mistake: a fixed length rod that points at the beam and never touches."""

    model, crank, _beam = _beam_and_crank()
    rod = model.part("rod")
    rod.add(Rot(Y=90) * Pos(Z=0.11) * Cylinder(0.012, 0.22), name="rod")
    model.articulation(
        "rod_pin",
        ArticulationType.REVOLUTE,
        crank,
        rod,
        origin=Origin(xyz=(0.12, 0.0, 0.0)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-1.4, upper=1.4),
        drive=AimAt("beam", ANCHOR),
    )
    model.validate()
    return model


def _proper_ram() -> ArticulatedObject:
    """The correct use: the barrel aims, and the rod it drives reaches."""

    model, crank, _beam = _beam_and_crank()
    barrel = model.part("barrel")
    barrel.add(Rot(Y=90) * Pos(Z=0.06) * Cylinder(0.02, 0.12), name="tube")
    rod = model.part("rod")
    rod.add(Rot(Y=90) * Pos(Z=0.06) * Cylinder(0.01, 0.12), name="shaft")
    rod.add(Pos(X=0.16) * Cylinder(0.02, 0.03, rotation=(90, 0, 0)), name="eye")
    kernel_model = model
    barrel_joint = model.articulation(
        "barrel_pivot",
        ArticulationType.REVOLUTE,
        crank,
        barrel,
        origin=Origin(xyz=(0.12, 0.0, 0.0)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-1.6, upper=1.6),
        drive=AimAt("beam", ANCHOR),
    )
    assert barrel_joint.drive is not None
    model.articulation(
        "ram_stroke",
        ArticulationType.PRISMATIC,
        barrel,
        rod,
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=-0.3, upper=0.4),
        drive=SpanTo("beam", ANCHOR, rest_length=0.16),
    )
    kernel_model.validate()
    return model


def _report(model: ArticulatedObject):
    ctx = TestContext(model)
    ctx.fail_if_aimed_link_never_reaches()
    return ctx.report()


def test_a_link_that_only_points_is_caught() -> None:
    report = _report(_link_only_aimed())
    assert not report.passed
    details = report.failures[0].details
    assert "never attaches" in details
    assert "loop closure" in details  # the message says what to do instead


def test_a_ram_whose_rod_reaches_passes() -> None:
    assert _report(_proper_ram()).passed


def test_the_gap_really_does_wander_on_the_broken_one() -> None:
    """Pin the physics the check is reading, not just its verdict."""

    from articraft.sdk._collision import MeshCollisionKernel

    model = _link_only_aimed()
    kernel = MeshCollisionKernel(model, mesh_tolerance=0.002)
    gaps = []
    for angle in np.linspace(-0.35, 0.35, 5):
        transforms = kernel.world_transforms({"beam_rocker": float(angle)})
        beam = transforms["beam"]
        anchor = beam[:3, :3] @ np.asarray(ANCHOR) + beam[:3, 3]
        vertices = kernel.part_world_vertices("rod", {"beam_rocker": float(angle)})
        gaps.append(float(np.linalg.norm(vertices - anchor, axis=1).min()))
    assert max(gaps) - min(gaps) > 0.02


def test_models_with_no_aimed_joints_pass_trivially() -> None:
    model, _crank, _beam = _beam_and_crank()
    model.validate()
    assert _report(model).passed


def test_the_shipped_examples_are_clean() -> None:
    """Both doc examples must model their linkage the way they teach it."""

    import runpy

    from articraft import package_dir

    for name in ("hydraulic_ram_loop.py", "four_bar_linkage.py"):
        values = runpy.run_path(str(package_dir / "sdk" / "docs" / "examples" / name))
        assert _report(values["object_model"]).passed, name


def test_four_bar_example_holds_its_pin_and_swings() -> None:
    import runpy

    from articraft import package_dir

    values = runpy.run_path(str(package_dir / "sdk" / "docs" / "examples" / "four_bar_linkage.py"))
    report = values["run_tests"]()
    assert report.passed, report.failures
    travel = next(metric for metric in report.metrics if metric.name == "rocker_end_travel_m")
    assert travel.value > 0.05
    assert math.isfinite(travel.value)
