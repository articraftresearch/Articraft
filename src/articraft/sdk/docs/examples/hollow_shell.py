from __future__ import annotations

from articraft.sdk import (
    ArticulatedObject,
    BoxGeometry,
    TestContext,
    TestReport,
)
from articraft.sdk.mesh import boolean_difference


def build_object_model() -> ArticulatedObject:
    outer = BoxGeometry((0.30, 0.24, 0.20))
    cavity = BoxGeometry((0.26, 0.20, 0.18)).translate(0.0, 0.0, 0.03)
    shell = boolean_difference(outer, cavity)

    model = ArticulatedObject("hollow_shell")
    body = model.part("body")
    body.add(shell, name="housing", color=(0.18, 0.42, 0.68))
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    return TestContext(object_model).report()
