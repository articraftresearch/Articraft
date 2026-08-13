# Assemblies and rigid bodies

Use `RigidBodyAssembly` as the complete physical object. It contains rigid bodies, physical
joints, and optional articulation trees.

```python
from articraft.sdk import Material, RigidBodyAssembly

model = RigidBodyAssembly("small_table")
body = model.rigid_body("body")
body.add(Box(0.8, 0.5, 0.04), name="top", material=Material.HARDWOOD)
```

Use meters for all geometry and linear physics values.

## `RigidBodyAssembly`

Create an assembly with a name and an optional physics scene:

```python
RigidBodyAssembly(name: str, *, scene: PhysicsScene = PhysicsScene())
```

Assembly, body, joint, and articulation names must be Python style identifiers. Use letters,
digits, and underscores, but do not start with a digit.

The exporter uses these names in USD paths, degree of freedom identifiers, and the manifest.
Shape names can be any nonempty string.

Create bodies through `rigid_body(...)`:

```python
model.rigid_body(
    name: str,
    *,
    mass_properties: MassProperties | None = None,
    body_state: BodyState | None = None,
) -> RigidBody
```

`MassProperties` can replace mass values from geometry and materials. `BodyState` sets the
initial flags and velocity of the body.

Read [simulation settings](38_simulation_settings.md) for `PhysicsScene` and `BodyState`.

The assembly exposes `rigid_bodies`, `joints`, and `articulations` for inspection. Use its
creation methods so duplicate names fail immediately.

## `RigidBody` and `body.add(...)`

Put all geometry that moves together in one `RigidBody`. Create another body only when the
geometry needs independent motion.

Add a named shape to a body:

```python
body.add(
    shape: build123d.Shape | MeshGeometry,
    *,
    name: str,
    material: Material | None = None,
    coating: Material | None = None,
    color: Sequence[float] | None = None,
) -> build123d.Shape | MeshGeometry
```

Shape names must be unique within a body. Build123d shapes and meshes both use body local
coordinates.

Apply a build123d or mesh transform before you call `add(...)`. The body does not store another
transform for each shape.

`material` accepts a `Material` that supplies density, contact values, and appearance. `coating`
changes the surface without changing the core density. `color` applies a display tint.

```python
from build123d import Box, Pos

housing = model.rigid_body("housing")
housing.add(Box(0.30, 0.22, 0.08), name="shell", material=Material.ALUMINUM)
housing.add(
    Pos(X=0.12) * Box(0.08, 0.02, 0.02),
    name="handle",
    material=Material.ABS_PLASTIC,
)
```

Shapes in one body can overlap. Overlap a protrusion with its supporting surface to keep the
body connected.

Use `body.shape(name)` to get a shape. Use `model.get_rigid_body(body_or_name)` to get a body.

## Geometry frames

Bind a local frame to a body with `body.at(...)`:

```python
body.at(source=None) -> BodyFrame
```

`source` accepts these values:

- A point with three numbers.
- A `JointFrame`.
- A build123d `Location`, `Plane`, or `Axis`.
- A build123d `Face`, `Edge`, `Vertex`, or `Shape`.
- A `MeshGeometry`.

The feature direction becomes the local Z axis of the frame. A flat face uses its center and
normal.

A rotational face uses its symmetry axis. A straight edge uses its midpoint and tangent.

A circle, arc, or ellipse uses the normal through its center. A complete shape or mesh uses
the center of its bounds.

Build123d locations keep their complete transform. The conversion also keeps the build123d
degree convention.

Use build123d topology queries instead of copied coordinates:

```python
from build123d import Axis

top = housing_shape.faces().sort_by(Axis.Z)[-1]
top_frame = housing_body.at(top)
hinge_axis = door_body.at(door_shape.edges().filter_by(Axis.X)[0])
```

Use `WORLD.at(...)` to create a world endpoint. Use a raw `JointFrame` when exact numbers are
the clearest input.

Convert an existing frame to other body coordinates with `frame_in(...)`:

```python
model.frame_in(source: BodyFrame, body: RigidBody | WORLD = WORLD) -> BodyFrame
```

Use this method to derive the second endpoint of a closure or fixed mount. It avoids duplicate
transform calculations.

## Resolve the assembly

Resolve the authored graph before a consumer uses it:

```python
resolved = model.resolve()
```

`resolve()` returns an immutable `ResolvedRigidBodyAssembly`. Export, testing, rendering,
collision checks, and motion use this same result.

The result contains these values:

- Every rigid body and physical joint.
- Each validated articulation tree.
- The derived `exclude_from_articulation` value for each joint.
- The validated reference `PhysicsState`.
- The assembly `PhysicsScene`.

Do not set `exclude_from_articulation` yourself. Resolution derives it from the articulation
trees.

## Physics states

`PhysicsState` stores one world transform for each body. Body poses are authoritative. Degree
of freedom positions are optional checked metadata.

```python
state = model.physics_state(
    {
        base: JointFrame(xyz=(0.0, 0.0, 0.0)),
        lid: lid_world_matrix,
    }
)
```

Use `resolved.forward_kinematics({...})` for an articulation tree. The loop solver finds
unspecified positions that keep closed constraints together.

Use a complete `PhysicsState` when one tree cannot define the complete pose. Also use it for
poses from a physics engine.

## Validate the assembly

`model.validate()` calls `resolve()` and returns `None` when validation passes.

Validation applies these rules:

- The assembly has at least one body.
- Each body has a unique name and one valid named shape.
- Joint and articulation names are unique.
- Joint frames are finite and endpoints are valid.
- The physical joint graph is connected.
- Each articulation is a connected tree without a cycle.
- A body or selected joint belongs to no more than one articulation.
- The reference state satisfies all constraints and limits.

An assembly can have no articulation. It can also have multiple separate articulations.

Use a complete `PhysicsState` when articulation trees cannot define one complete pose.

## USD layout

The exporter writes each body as a sibling USD rigid body. Named shapes remain below their
body:

```text
/World/<assembly>/rigid_bodies/<body>/shapes/<shape>
```

Joints are under `/World/<assembly>/joints`. This layout describes physical ownership, not an
articulation parent and child hierarchy.
