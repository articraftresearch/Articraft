# Mesh authoring SDK

The mini-articraft SDK is a small Python library for articulated 3D objects. It uses build123d
solids and procedural triangle meshes. You can use both geometry types in one object.

The public package is `mini_articraft.sdk`. All lengths use meters. Articulation rotations and
mesh rotations use radians. Build123d rotations use degrees.

## Create an object

An `ArticulatedObject` contains rigid parts. Each part contains one or more named shapes. Put
shapes in the same part if they always move together.

```python
from build123d import Box

from mini_articraft.sdk import ArticulatedObject, TestContext
from mini_articraft.sdk.export import export_object


model = ArticulatedObject("box")
model.part("body").add(Box(0.1, 0.1, 0.1), name="shell")
model.validate()

report = TestContext(model).report()
assert report.passed

result = export_object(model, "output")
print(result.usdz)
```

Each shape must have a name. The name must be unique in its part. A shape can also have an RGB
or RGBA color.

## Select a geometry type

Use build123d for exact solids, cuts, and fillets. Also use it for work that depends on faces or
edges. Add the completed build123d shape directly to a part.

Use `MeshGeometry` to edit vertices or make a procedural surface. Also use it for mesh booleans
and mesh repair. The SDK has builders for common solids and curved forms. It also has profiles,
sweeps, section lofts, shells, booleans, welds, and smooth operations.

```python
from mini_articraft.sdk import ArticulatedObject, RoundedBoxGeometry


model = ArticulatedObject("housing")
housing = RoundedBoxGeometry((0.12, 0.075, 0.028), radius=0.006)
model.part("body").add(housing, name="housing", color=(0.25, 0.30, 0.36))
```

`MeshGeometry` has a list of vertices and a list of triangle faces. Its transforms change the
mesh. They return the same object, so you can use a sequence of transforms. Use `copy()` first if
you must keep the source mesh.

Build123d and mesh geometry use the same local coordinates in a part. Use
`build123d_to_mesh()` only if a build123d shape must enter a mesh operation.

## Add motion

An articulation connects one parent part to one child part. The SDK has fixed, revolute,
continuous, and prismatic articulations.

```python
from build123d import Box

from mini_articraft.sdk import ArticulationType, MotionLimits, Origin


lid = model.part("lid")
lid.add(Box(0.1, 0.1, 0.01), name="panel")

model.articulation(
    "body_to_lid",
    ArticulationType.REVOLUTE,
    "body",
    "lid",
    origin=Origin(xyz=(0.0, 0.05, 0.05)),
    axis=(1.0, 0.0, 0.0),
    motion_limits=MotionLimits(lower=0.0, upper=1.8),
)
```

Make child geometry in the local frame of the child part. The articulation origin puts that
frame in the parent frame. A model must have one root part. Each other part must have one path to
the root part.

## Check and export the object

`model.validate()` checks names, geometry values, articulation rules, and the part tree.
`TestContext` checks physical relations such as distance, overlap, support, and motion.

The compiler also does these tests:

- It finds isolated parts.
- It finds disconnected geometry.
- It finds scale problems.
- It finds unwanted overlap.
- It finds joints that separate during motion.

USDZ export is in `mini_articraft.sdk.export`. Thus, a normal SDK import does not load OpenUSD.
Each part becomes one rigid body. Each named shape keeps its mesh and color.

## Reference

- Start with the [SDK quickstart](../src/mini_articraft/sdk/docs/common/00_quickstart.md).
- Read the [object and part API](../src/mini_articraft/sdk/docs/common/30_articulated_object.md).
- Read the [articulation API](../src/mini_articraft/sdk/docs/common/35_joints.md).
- Read the [test API](../src/mini_articraft/sdk/docs/common/40_testing.md).
- Read the [USDZ export API](../src/mini_articraft/sdk/docs/common/50_usdz_export.md).
- Read the [mesh API](../src/mini_articraft/sdk/docs/mesh/00_mesh_geometry.md).
- Use the [complete SDK examples](../src/mini_articraft/sdk/docs/examples).

The mesh reference has separate pages for profiles, sweeps, and section lofts. It also has pages
for booleans, shells, welds, and mesh refinement. The package includes a small set of
[build123d documents](../src/mini_articraft/sdk/docs/build123d).
