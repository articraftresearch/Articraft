# Articulations

An articulation places one child part relative to one parent part and defines
their allowed motion.

The public articulation API uses these types:

```python
from articraft.sdk import (
    Articulation,
    ArticulationType,
    MotionLimits,
    Origin,
)
```

## `Origin`

```python
Origin(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
)
```

`xyz` places the articulation frame in the parent part frame. It uses meters.
`rpy` rotates that frame with roll, pitch, and yaw in radians.

The rotation order is:

```text
Rz(yaw) @ Ry(pitch) @ Rx(roll)
```

Both tuples must contain three finite numeric values. `Origin` is immutable.

## Frame and direction rules

Articraft evaluates a child transform with this order:

```text
world(child) = world(parent) @ origin @ motion(q)
```

This gives the following rules:

1. `origin` is relative to the parent part.
2. `axis` is expressed in the rotated articulation frame, not in world space.
3. At `q = 0`, the child part frame is the articulation frame.
4. Positive revolute and continuous motion follows the right hand rule around
   `axis`.
5. Positive prismatic motion moves the child along positive `axis`.

The SDK normalizes a moving articulation axis before applying motion. Provide a
clear unit direction anyway. A moving articulation rejects a zero axis.

## `ArticulationType`

There are four articulation types.

| Type | Motion value | Required limits |
| --- | --- | --- |
| `FIXED` | No motion | No `MotionLimits` |
| `REVOLUTE` | Rotation in radians | Finite `lower` and `upper` |
| `CONTINUOUS` | Unbounded rotation in radians | `MotionLimits` without position bounds |
| `PRISMATIC` | Translation in meters | Finite `lower` and `upper` |

The exact lowercase strings `"fixed"`, `"revolute"`, `"continuous"`, and
`"prismatic"` are also accepted. Prefer `ArticulationType` values.

## `MotionLimits`

```python
MotionLimits(
    effort: float = 1.0,
    velocity: float = 1.0,
    lower: float | None = None,
    upper: float | None = None,
)
```

`effort` and `velocity` must be positive finite values. `lower` and `upper`
must be finite when present. `lower` may equal `upper`, but it cannot be greater
than `upper`.

Revolute bounds use radians. Prismatic bounds use meters. Continuous
articulations leave both bounds as `None`.

Rotational velocity uses radians per second. Prismatic velocity uses meters per
second. Effort is stored as numeric metadata for another system to interpret.

The SDK stores effort and velocity and writes them to export metadata. Mini
Articraft does not simulate force, torque, or velocity. Position bounds are not
an automatic clamp for authored test poses. Supply pose values that make sense
for the design.

`MotionLimits` is immutable.

## `Articulation`

```python
Articulation(
    name: str,
    articulation_type: ArticulationType | str,
    parent: str,
    child: str,
    origin: Origin = Origin(),
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    motion_limits: MotionLimits | None = None,
)
```

The constructor validates the name, type, frame, axis, and type specific motion
limits. It also rejects an articulation whose parent and child names are the
same.

Prefer `model.articulation(...)` for authoring. It also resolves the parent and
child against the current model and checks articulation name uniqueness before
appending the articulation.

## `model.articulation(...)`

```python
model.articulation(
    name: str,
    articulation_type: ArticulationType | str,
    parent: str | Part,
    child: str | Part,
    *,
    origin: Origin | None = None,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    motion_limits: MotionLimits | None = None,
) -> Articulation
```

`origin=None` becomes `Origin()`. The method accepts part names or the `Part`
objects returned by `model.part(...)`. Both parts must already exist.

The returned `Articulation` is the same object stored in
`model.articulations`.

## Fixed articulation

A fixed articulation places a child without adding motion. Do not pass
`motion_limits`.

```python
model.articulation(
    "base_to_post",
    ArticulationType.FIXED,
    base,
    post,
    origin=Origin(xyz=(0.0, 0.0, 0.08)),
)
```

The axis is not used for fixed motion. It still must contain three finite
numeric values because every `Articulation` has an axis field.

## Revolute articulation

A revolute articulation rotates the child around a bounded axis. Both position
bounds are required and use radians.

```python
model.articulation(
    "post_to_arm",
    ArticulationType.REVOLUTE,
    post,
    arm,
    origin=Origin(xyz=(0.0, 0.0, 0.22)),
    axis=(0.0, 1.0, 0.0),
    motion_limits=MotionLimits(
        effort=5.0,
        velocity=2.0,
        lower=-0.8,
        upper=0.8,
    ),
)
```

If positive motion turns the child the wrong way, reverse the axis. Keep lower
and upper in their numeric order.

## Continuous articulation

A continuous articulation rotates without lower or upper position bounds. It
still requires `MotionLimits` for positive effort and velocity values.

```python
model.articulation(
    "housing_to_rotor",
    ArticulationType.CONTINUOUS,
    housing,
    rotor,
    axis=(0.0, 0.0, 1.0),
    motion_limits=MotionLimits(effort=2.0, velocity=10.0),
)
```

Passing `lower` or `upper` to a continuous articulation raises
`ValidationError`.

## Prismatic articulation

A prismatic articulation moves the child along an axis. Both position bounds
are required and use meters.

```python
model.articulation(
    "cabinet_to_drawer",
    ArticulationType.PRISMATIC,
    cabinet,
    drawer,
    origin=Origin(xyz=(0.0, 0.0, 0.12)),
    axis=(1.0, 0.0, 0.0),
    motion_limits=MotionLimits(
        effort=40.0,
        velocity=0.25,
        lower=0.0,
        upper=0.28,
    ),
)
```

Model a sliding member with enough hidden length to remain inside its guide at
maximum travel. Put the articulation origin at the seating plane or guide
entry. Set the upper bound to the usable travel after the required insertion
length is kept.

## Rotated articulation frames

`Origin.rpy` rotates the meaning of `axis` because the axis is local to the
articulation frame.

```python
model.articulation(
    "base_to_tilted_arm",
    ArticulationType.REVOLUTE,
    base,
    arm,
    origin=Origin(
        xyz=(0.0, 0.0, 0.16),
        rpy=(0.0, 0.0, 0.5),
    ),
    axis=(0.0, 1.0, 0.0),
    motion_limits=MotionLimits(lower=-0.6, upper=0.9),
)
```

The `0.5` yaw value is in radians. Build123d `Rot` is unrelated and uses
degrees.

## The articulation tree

Full model validation requires one rooted tree.

- Each child part may have only one parent articulation.
- Parent and child must be different parts.
- Exactly one part must have no parent articulation.
- Every other part must be reachable from that root.
- Cycles and detached branches are invalid.

A single part model needs no articulation and is its own root.

The tree rule describes rigid ownership and motion. It does not prove that
parts touch. Compile time geometry checks report physically isolated parts and
unintended overlaps separately.

## Lookup and inspection

```python
hinge = model.get_articulation("post_to_arm")
```

The returned object exposes these fields:

- `name`
- `articulation_type`
- `parent` and `child`
- `origin`
- `axis`
- `motion_limits`

Call `articulation.validate()` to validate one articulation again. Call
`model.validate()` to check its part references and its place in the complete
tree.
