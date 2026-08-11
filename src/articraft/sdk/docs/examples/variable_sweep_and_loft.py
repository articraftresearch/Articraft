"""Build a soft vessel form with a shaped handle.

The body uses smooth loft interpolation between authored profiles. The handle
uses sweep sections to change its width, thickness, rotation, and profile along
one curved path.
"""

from __future__ import annotations

from articraft.sdk import (
    RigidBodyAssembly,
    TestContext,
    TestReport,
)
from articraft.sdk.mesh import (
    LoftSection,
    SectionLoftSpec,
    SweepSection,
    rounded_rect_profile,
    section_loft,
    superellipse_profile,
    sweep_profile_along_spline,
)


def vessel_section(width: float, depth: float, z: float, exponent: float) -> LoftSection:
    return LoftSection(
        tuple((x, y, z) for x, y in superellipse_profile(width, depth, exponent, segments=48))
    )


def build_object_model() -> RigidBodyAssembly:
    body_geometry = section_loft(
        SectionLoftSpec(
            sections=(
                vessel_section(0.064, 0.052, 0.000, 3.2),
                vessel_section(0.092, 0.070, 0.018, 3.0),
                vessel_section(0.105, 0.078, 0.072, 2.5),
                vessel_section(0.088, 0.066, 0.126, 2.8),
                vessel_section(0.076, 0.058, 0.148, 3.4),
            ),
            interpolation="catmull_rom",
            samples_per_span=5,
        )
    )

    handle_geometry = sweep_profile_along_spline(
        (
            (0.041, 0.0, 0.118),
            (0.075, 0.0, 0.111),
            (0.088, 0.0, 0.078),
            (0.079, 0.0, 0.039),
            (0.044, 0.0, 0.030),
        ),
        profile=rounded_rect_profile(0.012, 0.008, 0.003, corner_segments=5),
        samples_per_segment=12,
        up_hint=(0.0, 1.0, 0.0),
        sections=(
            SweepSection(0.0, scale=0.72),
            SweepSection(
                0.5,
                profile=superellipse_profile(0.015, 0.009, 2.2, segments=28),
                scale=(1.12, 0.90),
                rotation=0.18,
                offset=(0.001, 0.0),
            ),
            SweepSection(1.0, scale=0.72),
        ),
        section_interpolation="catmull_rom",
        section_tension=0.15,
    )

    model = RigidBodyAssembly("variable_sweep_and_loft")
    body = model.rigid_body("body")
    body.add(body_geometry, name="smooth_body", color=(0.27, 0.55, 0.62))
    body.add(handle_geometry, name="shaped_handle", color=(0.18, 0.38, 0.43))
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    body_low, body_high = ctx.shape_world_bounds("body", "smooth_body")
    handle_low, handle_high = ctx.shape_world_bounds("body", "shaped_handle")
    ctx.check(
        "smooth_body_has_expected_height",
        body_low[2] <= 0.001 and body_high[2] >= 0.147,
        "The smooth loft should span all authored body sections.",
    )
    ctx.check(
        "handle_reaches_beyond_body",
        handle_high[0] > body_high[0] and handle_low[2] < 0.04,
        "The shaped sweep should form a full handle outside the body.",
    )
    return ctx.report()
