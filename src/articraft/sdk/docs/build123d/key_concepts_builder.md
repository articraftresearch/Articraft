# Key Concepts: Builder Mode

Source:

- https://build123d.readthedocs.io/en/latest/_sources/key_concepts_builder.rst.txt

Use this page when writing `BuildLine`, `BuildSketch`, or `BuildPart` code.
Builder mode is usually the easiest build123d style for generated object code
because it keeps construction steps explicit.

## Builder And Algebra APIs

build123d has two primary modeling styles:

- Builder mode: context managers such as `BuildPart`, `BuildSketch`, and
  `BuildLine`.
- Algebra mode: shape expressions using operators such as `+`, `-`, `&`, and
  placement with `*`.

Both styles can be mixed in one model, but algebra expressions should not be used
inside an active builder context.

## Builder Paradigm

A builder maintains a running result. Each object or operation created inside the
builder updates that result according to its `mode`.

Common modes:

- `Mode.ADD`: add to the running result.
- `Mode.SUBTRACT`: subtract from the running result.
- `Mode.INTERSECT`: keep only the overlap with the running result.
- `Mode.REPLACE`: replace the running result.
- `Mode.PRIVATE`: create the object without combining it with the running result.

Final builder outputs:

- `BuildLine(...).line`
- `BuildSketch(...).sketch`
- `BuildPart(...).part`

Example:

```python
from build123d import *

with BuildPart() as example_part:
    with BuildSketch():
        Rectangle(20, 20)
    extrude(amount=10)

    with BuildSketch(Plane(example_part.faces().sort_by(Axis.Z).last)):
        Circle(5)
    extrude(amount=-5, mode=Mode.SUBTRACT)

result_part = example_part.part
```

In articraft, pass `builder.part` to `model.rigid_body(...)`, not the builder
object itself.

## Placement Must Happen Before Creation

Builder objects add themselves to the active builder when they are created. Do
not create an object and then move the returned temporary object:

```python
with BuildPart() as invalid:
    Cylinder(1, 2).moved(Location((1, 2, 3)))
```

The move happens after the cylinder has already been added to the builder. Use a
location context instead:

```python
with BuildPart() as valid:
    with Locations((1, 2, 3)):
        Cylinder(1, 2)
```

## Builders Are Tools, Not Shapes

Builders create normal build123d shapes, but the builder object is not itself the
shape.

```python
with BuildPart() as my_part:
    Box(10, 10, 10)

part_shape = my_part.part
```

Likewise:

```python
with BuildSketch() as my_sketch:
    Circle(2)

sketch_shape = my_sketch.sketch
```

```python
with BuildLine() as my_line:
    Line((0, 0), (1, 0))

line_shape = my_line.line
```

## Implicit Builder Scope

Objects and operations apply to the builder that is active in their Python scope.
You normally do not pass the builder instance into object constructors.

```python
with BuildPart() as part_builder:
    Box(10, 10, 10)
    with BuildSketch() as sketch_builder:
        Circle(2)
```

Here `Box` applies to `part_builder`, while `Circle` applies to
`sketch_builder`.

## Workplanes

Builders accept workplanes. The default is usually `Plane.XY`. Workplanes let
you sketch or build in a local coordinate system.

```python
with BuildPart(Plane.XY) as example:
    Box(10, 10, 4)

    with BuildSketch(example.faces().sort_by(Axis.Z)[-1]):
        Circle(2)
    extrude(amount=2)

    with BuildSketch(Plane.XZ):
        Rectangle(4, 2)
```

Workplanes can come from standard planes or from selected faces. Multiple
workplanes can be supplied when the same operation should repeat on several
planes.

## Location Contexts

Location contexts place objects before those objects are created.

Common contexts:

- `Locations`: explicit positions and orientations.
- `GridLocations`: rectangular grids.
- `PolarLocations`: circular patterns.
- `HexLocations`: hexagonal grids.

Example:

```python
with BuildPart():
    with Locations((0, 10), (0, -10)):
        Box(1, 1, 1)

        with GridLocations(x_spacing=5, y_spacing=5, x_count=2, y_count=2):
            Sphere(1)

        Cylinder(1, 1)
```

The `Box` and `Cylinder` are created at the two `Locations`. The `Sphere` is
created at the nested grid locations relative to each outer location.

## Operation Inputs

Operations often take a shape or an iterable of shapes. Builder selectors return
`ShapeList`, which can be passed directly to operations.

```python
with BuildPart() as pipes:
    Box(10, 10, 10, rotation=(10, 20, 30))
    fillet(pipes.edges(Select.LAST), radius=0.2)
```

Here the fillet is applied to edges from the last operation.

## Rotation

Sketch rotations are single angles in degrees. Part rotations are three-angle
rotations, either as `Rotation(x, y, z)` or as a tuple.

```python
with BuildPart():
    Box(10, 10, 10, rotation=(10, 20, 30))
```

For precise placement with position and orientation, pass `Location` objects to
`Locations`.

## Pending Objects

When a nested builder exits, it can push its result into the parent builder as a
pending object. For example, a `BuildSketch` inside a `BuildPart` creates pending
faces that `extrude()` can consume.

```python
with BuildPart() as pillow_block:
    with BuildSketch() as plan:
        Rectangle(width, height)
        fillet(plan.vertices(), radius=f_rad)
    extrude(amount=thickness)
```

Most code should not touch pending objects directly, but they are available as
`builder.pending_edges` and `builder.pending_faces` when debugging.

