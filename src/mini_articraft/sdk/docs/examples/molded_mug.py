"""Attach a protrusion by molding it into the body -- no mounting pads.

A handle, spout, leg, or arm joins a body as one continuous molded piece. Shape the
protrusion's own end so it overlaps the body, then blend them. This mug shows the
geometry pattern with a handle:

  1. Sweep the handle as a loop whose two ends sit a few mm INSIDE the body wall.
  2. `weld(body, handle)` builds a smooth transition across the overlap.
  3. Add the single welded result to the part.

Do not bridge the handle to the body with a separate mounting pad, boss, or bracket.
The handle's own end reaches into the body and the weld generates the final surface.
"""

from __future__ import annotations

from mini_articraft.sdk import (
    LatheGeometry,
    RigidBodyAssembly,
    TestContext,
    TestReport,
)
from mini_articraft.sdk.mesh import (
    rounded_rect_profile,
    sweep_profile_along_spline,
    weld,
)


def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("ceramic_mug")
    mug = model.rigid_body("mug")

    # Ceramic body: a lathe silhouette (radius, z) as one closed solid.
    body = LatheGeometry(
        [
            (0.000, 0.000),
            (0.038, 0.000),
            (0.042, 0.010),
            (0.041, 0.090),
            (0.044, 0.100),
            (0.040, 0.104),
            (0.000, 0.104),
        ],
        segments=96,
        closed=True,
    )

    # Handle: a C-loop swept tube on the +x side. Both ends land a few mm INSIDE the
    # body wall (x < the body radius there), so the handle's own ends overlap the body.
    handle = sweep_profile_along_spline(
        (
            (0.034, 0.0, 0.090),  # upper end, embedded in the wall
            (0.072, 0.0, 0.082),
            (0.084, 0.0, 0.055),
            (0.074, 0.0, 0.026),
            (0.034, 0.0, 0.020),  # lower end, embedded in the wall
        ),
        profile=rounded_rect_profile(0.012, 0.008, 0.003, corner_segments=8),
        samples_per_segment=16,
        cap_profile=True,
        up_hint=(0.0, 1.0, 0.0),
    )

    # Fuse into one molded ceramic piece. No mounting pad bridges the gap.
    molded = weld(body, handle, radius=0.006, tolerance=0.002, profile="round")
    mug.add(molded, name="body_with_molded_handle", color=(0.92, 0.91, 0.88))
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    _, hi = ctx.shape_world_bounds("mug", "body_with_molded_handle")
    ctx.check(
        "molded_handle_reaches_out_from_body",
        hi[0] > 0.06,
        "The welded handle should extend outward from the mug wall as one piece.",
    )
    return ctx.report()
