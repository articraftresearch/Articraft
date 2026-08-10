from __future__ import annotations

from articraft.sdk import (
    ArticulatedObject,
    TestContext,
    TestReport,
)
from articraft.sdk.mesh import (
    LoftSection,
    SectionLoftSpec,
    section_loft,
    tube_from_spline_points,
)


def rectangle_section(width: float, depth: float, z: float) -> LoftSection:
    x = width * 0.5
    y = depth * 0.5
    return LoftSection(((-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)))


def build_object_model() -> ArticulatedObject:
    housing = section_loft(
        SectionLoftSpec(
            sections=(
                rectangle_section(0.16, 0.10, 0.00),
                rectangle_section(0.20, 0.13, 0.15),
                rectangle_section(0.14, 0.09, 0.30),
            )
        )
    )
    wire = tube_from_spline_points(
        ((0.09, 0.0, 0.03), (0.10, 0.0, 0.10), (0.10, 0.0, 0.20), (0.08, 0.0, 0.27)),
        radius=0.01,
        samples_per_segment=8,
    )

    model = ArticulatedObject("loft_with_wire")
    body = model.part("body")
    body.add(housing, name="housing", color=(0.35, 0.38, 0.42))
    body.add(wire, name="routed_wire", color=(0.06, 0.06, 0.07))
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    return TestContext(object_model).report()
