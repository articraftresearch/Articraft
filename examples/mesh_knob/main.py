"""Canonical procedural-mesh example: a lathe-turned knob, single part.

One MeshGeometry, built from a closed (radius, z) silhouette, with an
authored custom check that the result is watertight -- the property that
makes it safe for booleans, export, and printing.

Compile it:  python -m articraft.compiler.worker <run_dir>
"""

from __future__ import annotations

from articraft.sdk import (
    LatheGeometry,
    RigidBodyAssembly,
    TestContext,
    TestReport,
)
from articraft.sdk.mesh import boolean_union

PROFILE = [
    (0.000, 0.000),
    (0.016, 0.000),
    (0.020, 0.006),
    (0.020, 0.014),
    (0.016, 0.020),
    (0.000, 0.020),
]

RIDGE_PROFILE = [
    (0.000, 0.006),
    (0.022, 0.006),
    (0.024, 0.008),
    (0.022, 0.010),
    (0.000, 0.010),
]


def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("knob")
    knob = model.rigid_body("knob")

    body = LatheGeometry(PROFILE)
    ridge = LatheGeometry(RIDGE_PROFILE)
    knob.add(boolean_union(body, ridge), name="body", color=(0.16, 0.34, 0.55, 1.0))
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    body = object_model.get_rigid_body("knob").get_shape("body")
    ctx.check("body_is_watertight", body.is_watertight, "the union must stay watertight")
    return ctx.report()
