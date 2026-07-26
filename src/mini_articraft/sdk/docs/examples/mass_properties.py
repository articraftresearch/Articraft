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

from mini_articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    BoxGeometry,
    CylinderGeometry,
    MassProperties,
    MaterialDensity,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
)
from mini_articraft.sdk.mesh import boolean_difference


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("weighted_bin")

    # A steel base plate: mass comes from the material's density times the
    # measured volume of the geometry below.
    base = model.part("base", mass=MassProperties(material=MaterialDensity.STEEL))
    base.add(BoxGeometry((0.24, 0.18, 0.012)), name="base_plate", color=(0.55, 0.56, 0.60))

    # A hollow plastic bin. The cavity is cut out, so the measured volume is the
    # wall, not a solid block -- the mass follows automatically.
    shell = boolean_difference(
        BoxGeometry((0.22, 0.16, 0.14)).translate(0.0, 0.0, 0.076),
        BoxGeometry((0.20, 0.14, 0.13)).translate(0.0, 0.0, 0.082),
    )
    body = model.part("body", mass=MassProperties(material=MaterialDensity.ABS_PLASTIC))
    body.add(shell, name="bin_shell", color=(0.85, 0.86, 0.88))

    # A lid whose real weight is known: an explicit mass wins over any volume,
    # which is useful when the geometry stands in for something denser.
    lid = model.part("lid", mass=MassProperties(mass=0.12))
    lid.add(
        CylinderGeometry(0.02, 0.006).rotate_y(1.5708),
        name="hinge_barrel",
        color=(0.20, 0.21, 0.24),
    )
    lid.add(
        BoxGeometry((0.22, 0.16, 0.008)).translate(0.0, 0.08, 0.0),
        name="lid_panel",
        color=(0.20, 0.21, 0.24),
    )

    # The bin is bolted to the plate: a FIXED articulation keeps one root part.
    model.articulation(
        "body_to_base",
        ArticulationType.FIXED,
        base,
        body,
        origin=Origin(xyz=(0.0, 0.0, 0.006)),
    )
    model.articulation(
        "lid_hinge",
        ArticulationType.REVOLUTE,
        body,
        lid,
        origin=Origin(xyz=(0.0, -0.08, 0.146)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(effort=6.0, velocity=2.0, lower=0.0, upper=1.9),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    # With physics enabled this check is part of every compile; running it here
    # keeps the example honest either way.
    ctx.fail_if_parts_have_no_mass()
    return ctx.report()
