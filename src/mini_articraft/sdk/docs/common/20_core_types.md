# Shared units and types

This page lists the rules shared by the object, mesh, articulation, testing, and
export APIs. Read the linked owner page for signatures and examples.

## Units and coordinates

Mini Articraft geometry uses meters. Build123d dimensions, mesh vertices,
`Origin.xyz`, prismatic positions, prismatic limits, bounds, distances, and
tolerances all use meters.

The SDK uses a right handed XYZ coordinate system. USD stages use Z as the up
axis and declare one meter per unit.

The SDK uses radians for these values:

- `Origin.rpy`
- Revolute and continuous positions
- Revolute limits
- `MeshGeometry.rotate(...)` and its axis helpers

Build123d `Rot(...)` and build123d rotation methods use degrees because that is
the build123d convention. Convert angles when a value moves between the two
APIs.

## Geometry values

`Part.add(...)` accepts a build123d `Shape` or `MeshGeometry`. Both use the
part's local coordinates. Apply a build123d `Pos`, `Rot`, or `Location` before
adding a build123d shape. Mini Articraft does not add another shape transform.

Every shape has a unique name within its part. A shape can have a `Material`, or
an RGB/RGBA color as shorthand for a matte material. The SDK has no separate
visual or asset type.

Read [articulated objects and parts](30_assembly.md) for named shape
authoring and validation. Read [mesh geometry and solid
builders](../mesh/00_mesh_geometry.md) for mesh editing, conversion, and
procedural builders.

## Articulation values

`Origin`, `MotionLimits`, `ArticulationType`, and `Articulation` describe the
connection between two rigid parts. An articulation origin places a child part
relative to its parent. It does not transform one named shape.

Read [articulations](35_joints.md) for the fixed, revolute, continuous, and
prismatic rules.

## Testing values

`TestContext` records authored geometry checks. `TestReport` contains their
blocking failures, warnings, and justified allowances. `DistanceFinding` is
the inspection result from `distance_between(...)`. `AllowedOverlap` records
one justified named shape pair.

Read [testing geometry and assemblies](40_testing.md) for exact checks, world
bounds, distance inspection, and compiler owned checks.

## Errors and export

The public SDK error types are `SDKError` and `ValidationError`. Read
[errors](10_errors.md) for when each kind of failure is raised.

Read [USDZ export](50_usdz_export.md) for the part and shape hierarchy, materials,
joint targets, stage units, and output numbering.

## Public type ownership

Use these pages for the complete public surface:

- `RigidBodyAssembly` and `RigidBody` are documented in [articulated objects and
  parts](30_assembly.md).
- `Origin`, `MotionLimits`, `ArticulationType`, and `Articulation` are
  documented in [articulations](35_joints.md).
- `MeshGeometry` and mesh builders are documented in [mesh geometry and solid
  builders](../mesh/00_mesh_geometry.md) and the other pages in the mesh
  directory.
- `TestContext`, `TestReport`, `TestFailure`, `DistanceFinding`, and
  `AllowedOverlap` are documented in [testing geometry and
  assemblies](40_testing.md).
- `SDKError` and `ValidationError` are documented in [errors](10_errors.md).
