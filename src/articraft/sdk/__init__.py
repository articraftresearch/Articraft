"""Public SDK for authoring and testing articulated objects.

Everything here is imported from the package root, and mesh recipes also from
``articraft.sdk.mesh``. The modules below are how the source is organised,
not extra import paths.

Six threads run through this package:

- **structure** -- what the object is and how it moves. ``bodies`` holds
  ``RigidBody``; ``assembly`` holds ``RigidBodyAssembly``, its ``Joint``s and
  the ``Articulation``s that name the trees a simulator solves.
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
  ``articraft.sdk`` import.
"""

from __future__ import annotations

from articraft.sdk._mesh.core import (
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
from articraft.sdk._mesh.health import (
    MeshHealthFinding,
    MeshHealthIssue,
    MeshHealthReport,
    analyze_mesh_health,
)
from articraft.sdk._mesh.sweeps import (
    ArcPipeGeometry,
    PipeGeometry,
    SweepGeometry,
    WirePolylineGeometry,
)
from articraft.sdk.assembly import (
    WORLD,
    Articulation,
    Joint,
    JointAxis,
    JointDOF,
    JointFrame,
    PhysicsState,
    RigidBodyAssembly,
)
from articraft.sdk.bodies import RigidBody
from articraft.sdk.errors import SDKError, ValidationError
from articraft.sdk.mass import MassProperties
from articraft.sdk.materials import Material
from articraft.sdk.physics import EARTH_GRAVITY, BodyState, PhysicsScene
from articraft.sdk.testing import (
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
from articraft.sdk.visual import (
    ImagePoint,
    LineOverlay,
    MeridionalSectionView,
    ModelView,
    MotionStripView,
    PointOverlay,
    Reticle,
    SectionView,
    annotate_image,
    render_view,
)

__all__ = [
    "EARTH_GRAVITY",
    "WORLD",
    "AllowedMeshIssues",
    "AllowedOverlap",
    "ArcPipeGeometry",
    "Articulation",
    "BodyState",
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
    "Joint",
    "JointAxis",
    "JointDOF",
    "JointFrame",
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
    "MotionStripView",
    "PhysicsScene",
    "PhysicsState",
    "PipeGeometry",
    "PointOverlay",
    "PoseSample",
    "Reticle",
    "RigidBody",
    "RigidBodyAssembly",
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
    "annotate_image",
    "render_view",
]
