"""Give each part a mass by saying what it is made of.

A part is one rigid body, so a simulator needs to know how heavy it is, where that
weight sits, and how it resists rotation. You name the material; the SDK measures the
volume, center of mass, and inertia from the geometry the part already contains.

Three ways to declare it, in precedence order (give exactly one):

  1. ``mass=`` sets kilograms directly and ignores volume.
  2. ``density=`` multiplies your kg/m^3 by the measured volume.
  3. ``material=`` looks up a density from ``MaterialDensity`` and does the same.

``center_of_mass``, ``diagonal_inertia``, and ``principal_axes`` are measured unless
you pass them explicitly.
"""

from __future__ import annotations

import math

from mini_articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    CylinderGeometry,
    MassProperties,
    MaterialDensity,
    MotionLimits,
    Origin,
    RoundedBoxGeometry,
    TestContext,
    TestReport,
)
from mini_articraft.sdk.mesh import boolean_difference, weld

RADIUS = 0.055
WALL = 0.004
HEIGHT = 0.10
HINGE_Y = -RADIUS  # the pivot sits on the rear rim


def build_object_model() -> ArticulatedObject:
    """A round tin: a steel base, a hollow steel body, and a hardwood lid."""

    model = ArticulatedObject("weighted_tin")

    # Steel base disc. Mass is the material's density times the measured volume.
    base = model.part("base", mass_properties=MassProperties(material=MaterialDensity.STEEL))
    base.add(
        CylinderGeometry(RADIUS + 0.006, 0.006, radial_segments=64).translate(0.0, 0.0, 0.003),
        name="base_disc",
        color=(0.55, 0.56, 0.60),
    )

    # Hollow steel body: the cavity is cut away, so the measured volume is the
    # wall rather than a solid cylinder, and the mass follows automatically.
    body_part = model.part("body", mass_properties=MassProperties(material=MaterialDensity.STEEL))
    shell = boolean_difference(
        CylinderGeometry(RADIUS, HEIGHT, radial_segments=64).translate(0.0, 0.0, HEIGHT / 2),
        CylinderGeometry(RADIUS - WALL, HEIGHT, radial_segments=64).translate(
            0.0, 0.0, HEIGHT / 2 + WALL
        ),
    )
    # The hinge lug is welded into the wall, so the part stays one molded piece.
    lug = RoundedBoxGeometry((0.020, 0.010, 0.012), 0.004).translate(0.0, HINGE_Y, HEIGHT - 0.004)
    body_part.add(
        weld(shell, lug, radius=0.004, tolerance=0.0012),
        name="body_shell",
        color=(0.72, 0.73, 0.76),
    )

    # A hardwood lid, authored in the hinge frame: the disc reaches forward from
    # the pivot so it covers the mouth, and the barrel sits on the pivot itself.
    lid = model.part("lid", mass_properties=MassProperties(material=MaterialDensity.HARDWOOD))
    disc = CylinderGeometry(RADIUS, 0.008, radial_segments=64).translate(0.0, RADIUS, 0.004)
    barrel = CylinderGeometry(0.006, 0.024, radial_segments=32).rotate_y(math.pi / 2)
    lid.add(
        weld(disc, barrel, radius=0.004, tolerance=0.0012),
        name="lid_disc",
        color=(0.62, 0.45, 0.24),
    )

    model.articulation(
        "body_to_base",
        ArticulationType.FIXED,
        base,
        body_part,
        origin=Origin(xyz=(0.0, 0.0, 0.006)),
    )
    model.articulation(
        "lid_hinge",
        ArticulationType.REVOLUTE,
        body_part,
        lid,
        origin=Origin(xyz=(0.0, HINGE_Y, HEIGHT)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(effort=4.0, velocity=2.0, lower=0.0, upper=1.9),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    # With physics enabled this runs on every compile; asserting it here keeps the
    # example honest either way.
    ctx.fail_if_parts_have_no_mass()
    _, closed_top = ctx.shape_world_bounds("lid", "lid_disc")
    with ctx.pose({"lid_hinge": 1.6}):
        _, open_top = ctx.shape_world_bounds("lid", "lid_disc")
        ctx.check(
            "lid_opens_upward",
            open_top[2] > closed_top[2] + 0.03,
            "At an open pose the lid should stand well above its closed height.",
        )
    return ctx.report()
