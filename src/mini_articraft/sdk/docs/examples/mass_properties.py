"""Say what each shape is made of, and the physics follows.

A part is one rigid body, so a simulator needs to know how heavy it is, where that
weight sits, and how it resists rotation. You name the material on each shape; the
SDK measures volume, center of mass, and inertia from the geometry.

Material lives on the shape, not the part, so one rigid body can be made of more
than one thing -- this tin is a steel body with a hardwood lid, and it weighs and
balances accordingly. The same material also decides how each surface behaves on
contact and how it looks, so naming it once is usually all a shape needs.

Reach for ``MassProperties`` on the part only to override measurement: a density
the library does not cover, or a known weight for geometry that stands in for
something else.
"""

from __future__ import annotations

import math

from mini_articraft.sdk import (
    CylinderGeometry,
    JointAxis,
    JointDOF,
    JointFrame,
    Material,
    RigidBodyAssembly,
    RoundedBoxGeometry,
    TestContext,
    TestReport,
)
from mini_articraft.sdk.mesh import boolean_difference, weld

RADIUS = 0.055
WALL = 0.004
HEIGHT = 0.10
HINGE_Y = -RADIUS  # the pivot sits on the rear rim


def build_object_model() -> RigidBodyAssembly:
    """A round tin: a steel base, a hollow steel body, and a hardwood lid."""

    model = RigidBodyAssembly("weighted_tin")

    # Steel base disc. Mass is the material's density times the measured volume.
    base = model.rigid_body("base")
    base.add(
        CylinderGeometry(RADIUS + 0.006, 0.006, radial_segments=64).translate(0.0, 0.0, 0.003),
        name="base_disc",
        material=Material.STEEL,
    )

    # Hollow steel body: the cavity is cut away, so the measured volume is the
    # wall rather than a solid cylinder, and the mass follows automatically.
    body_part = model.rigid_body("body")
    # Weld the hinge lug onto the solid cylinder first, then cut the cavity. Welding
    # into an already-hollow shell makes the bead negotiate the thin wall and the
    # curved inner surface at once, which is where slivers and degenerate faces come
    # from. Build up on clean geometry, subtract last.
    lug = RoundedBoxGeometry((0.020, 0.010, 0.012), 0.004).translate(
        0.0, HINGE_Y - 0.003, HEIGHT - 0.004
    )
    molded = weld(
        CylinderGeometry(RADIUS, HEIGHT, radial_segments=64).translate(0.0, 0.0, HEIGHT / 2),
        lug,
        radius=0.004,
        tolerance=0.0012,
    )
    shell = boolean_difference(
        molded,
        CylinderGeometry(RADIUS - WALL, HEIGHT, radial_segments=64).translate(
            0.0, 0.0, HEIGHT / 2 + WALL
        ),
    )
    body_part.add(shell, name="body_shell", material=Material.STEEL)

    # A hardwood lid, authored in the hinge frame: the disc reaches forward from
    # the pivot so it covers the mouth, and the barrel sits on the pivot itself.
    lid = model.rigid_body("lid")
    disc = CylinderGeometry(RADIUS, 0.008, radial_segments=64).translate(0.0, RADIUS, 0.004)
    barrel = CylinderGeometry(0.006, 0.024, radial_segments=32).rotate_y(math.pi / 2)
    lid.add(
        weld(disc, barrel, radius=0.004, tolerance=0.0012),
        name="lid_disc",
        material=Material.HARDWOOD,
    )

    # No free axes is a fixed joint: the body is welded to its base.
    model.joint(
        "body_to_base",
        body0=base,
        frame0=JointFrame(xyz=(0.0, 0.0, 0.006)),
        body1=body_part,
        frame1=JointFrame(),
    )
    model.joint(
        "lid_hinge",
        body0=body_part,
        frame0=JointFrame(xyz=(0.0, HINGE_Y, HEIGHT)),
        body1=lid,
        frame1=JointFrame(),
        dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 1.9)),),
    )
    model.articulation("main", root=base, joints=["body_to_base", "lid_hinge"])
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
