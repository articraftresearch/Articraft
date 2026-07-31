"""Public SDK for authoring and testing articulated objects.

One canonical import path per category:

- Object and articulation modeling, geometry classes, physical testing, and
  errors live here at the package root.
- Advanced mesh authoring and repair recipes (booleans, welds, snapping,
  profile/wire sampling, sweep helpers, section lofts, shell partitioning,
  refinement) live under ``mini_articraft.sdk.mesh``.
- USDZ publication lives under ``mini_articraft.sdk.export`` so importing the
  root SDK does not eagerly load OpenUSD.
"""

from __future__ import annotations

from mini_articraft.sdk._mesh.core import (
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    DomeGeometry,
    ExtrudeGeometry,
    ExtrudeWithHolesGeometry,
    LatheGeometry,
    LoftGeometry,
    MeshGeometry,
    RoundedBoxGeometry,
    SphereGeometry,
    SuperellipsoidGeometry,
    TorusGeometry,
)
from mini_articraft.sdk._mesh.health import (
    MeshHealthFinding,
    MeshHealthIssue,
    MeshHealthReport,
    analyze_mesh_health,
)
from mini_articraft.sdk._mesh.sweeps import (
    ArcPipeGeometry,
    PipeGeometry,
    SweepGeometry,
    WirePolylineGeometry,
)
from mini_articraft.sdk.errors import SDKError, ValidationError
from mini_articraft.sdk.joints import (
    Articulation,
    ArticulationType,
    MotionLimits,
    Origin,
)
from mini_articraft.sdk.mass import MassProperties
from mini_articraft.sdk.materials import Material
from mini_articraft.sdk.object import ArticulatedObject, Part
from mini_articraft.sdk.testing import (
    AllowedMeshIssues,
    AllowedOverlap,
    DistanceFinding,
    FailureKind,
    GeometryMetrics,
    PoseSample,
    TestArtifact,
    TestContext,
    TestFailure,
    TestMetric,
    TestReport,
)
from mini_articraft.sdk.visual import (
    LineOverlay,
    MeridionalSectionView,
    ModelView,
    MotionStripView,
    PointOverlay,
    SectionView,
    render_view,
)

__all__ = [
    "AllowedMeshIssues",
    "AllowedOverlap",
    "ArcPipeGeometry",
    "ArticulatedObject",
    "Articulation",
    "ArticulationType",
    "BoxGeometry",
    "CapsuleGeometry",
    "ConeGeometry",
    "CylinderGeometry",
    "DistanceFinding",
    "DomeGeometry",
    "ExtrudeGeometry",
    "ExtrudeWithHolesGeometry",
    "FailureKind",
    "GeometryMetrics",
    "LatheGeometry",
    "LineOverlay",
    "LoftGeometry",
    "MassProperties",
    "Material",
    "MeridionalSectionView",
    "MeshGeometry",
    "MeshHealthFinding",
    "MeshHealthIssue",
    "MeshHealthReport",
    "ModelView",
    "MotionLimits",
    "MotionStripView",
    "Origin",
    "Part",
    "PipeGeometry",
    "PointOverlay",
    "PoseSample",
    "RoundedBoxGeometry",
    "SDKError",
    "SectionView",
    "SphereGeometry",
    "SuperellipsoidGeometry",
    "SweepGeometry",
    "TestArtifact",
    "TestContext",
    "TestFailure",
    "TestMetric",
    "TestReport",
    "TorusGeometry",
    "ValidationError",
    "WirePolylineGeometry",
    "analyze_mesh_health",
    "render_view",
]
