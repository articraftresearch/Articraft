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

from mini_articraft.sdk._mesh_core import (
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
from mini_articraft.sdk._mesh_health import (
    MeshHealthFinding,
    MeshHealthIssue,
    MeshHealthReport,
    analyze_mesh_health,
)
from mini_articraft.sdk._mesh_sweeps import (
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
from mini_articraft.sdk.materials import Material, SurfaceKind
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
    ImagePoint,
    LineOverlay,
    MeridionalSectionView,
    ModelView,
    MotionStripView,
    PointOverlay,
    ProjectedPoint,
    Reticle,
    SectionView,
    SurfaceHit,
    ViewProbe,
    annotate_image,
    probe_view,
    project_model_points,
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
    "ImagePoint",
    "LatheGeometry",
    "LineOverlay",
    "LoftGeometry",
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
    "ProjectedPoint",
    "Reticle",
    "RoundedBoxGeometry",
    "SDKError",
    "SectionView",
    "SphereGeometry",
    "SuperellipsoidGeometry",
    "SurfaceHit",
    "SurfaceKind",
    "SweepGeometry",
    "TestArtifact",
    "TestContext",
    "TestFailure",
    "TestMetric",
    "TestReport",
    "TorusGeometry",
    "ValidationError",
    "ViewProbe",
    "WirePolylineGeometry",
    "analyze_mesh_health",
    "annotate_image",
    "probe_view",
    "project_model_points",
    "render_view",
]
