"""Public SDK for authoring and testing articulated objects.

Everything here is imported from the package root, and mesh recipes also from
``mini_articraft.sdk.mesh``. The modules below are how the source is organised,
not extra import paths.

Six threads run through this package:

- **structure** -- what the object is and how it moves. ``object`` holds
  ``Part`` and ``ArticulatedObject``; ``joints`` holds ``Articulation`` and
  its motion limits.
- **geometry** -- how shapes are made. ``mesh`` is the public face; ``_mesh``
  is the engine behind it. Geometry is undifferentiated: a boolean or a weld
  does not know whether the result is heavy or shiny.
- **physics and appearance** -- what a shape is made of. Both live on one
  ``Material`` in ``materials``, deliberately: density, friction and
  restitution alongside colour, metallic and roughness. ``mass`` measures a
  part from those densities and lets you override the result; ``ambientcg``
  fetches texture sets.
- **verify** -- checking the result. ``testing`` provides ``TestContext``,
  which the authored ``run_tests()`` uses.
- **inspect** -- seeing the result. ``visual`` renders views for the author to
  look at.
- **publish** -- writing the result out. ``export`` turns a model into a
  validated USDZ package plus its manifest. It is imported explicitly rather
  than re-exported here, which keeps OpenUSD out of a plain
  ``mini_articraft.sdk`` import.
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
