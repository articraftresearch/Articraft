# Joints and articulations

Motion is two steps. A **joint** connects two rigid bodies and says which
freedoms they have. An **articulation** names the tree of joints the simulator
solves; anything left out of it closes a loop.

```python
from articraft.sdk import WORLD, JointAxis, JointDOF, JointFrame
```

## `JointFrame`

```python
JointFrame(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
)
```

Where the joint sits **in one body's own coordinates**, in meters and radians.
Every joint has two: one per body. They coincide at rest, so placing them is how
you say "these two points are pinned together."

`rpy` is roll, pitch, then yaw, applied in that order. Rotating a frame is also
how you get an axis the six named ones do not cover: to hinge about the
diagonal `(1, 1, 0)`, yaw the frame 45 degrees and use its own X.

## `JointDOF`

```python
JointDOF(axis: JointAxis, limits: tuple[float, float] | None = None)
```

One freedom the joint allows. `JointAxis` has six members — `TRANS_X`,
`TRANS_Y`, `TRANS_Z`, `ROT_X`, `ROT_Y`, `ROT_Z` — and any axis you do not list
is locked.

`limits` is `(lower, upper)`, in meters for a linear axis and radians for a
rotational one. **A range must contain zero**, because zero is the pose you
authored the object in. Leave `limits` out for a free axis, such as a wheel that
spins without end.

## `model.joint(...)`

```python
model.joint(
    name: str,
    *,
    body0: RigidBody | str | WORLD,
    frame0: JointFrame,
    body1: RigidBody | str | WORLD,
    frame1: JointFrame,
    dofs: tuple[JointDOF, ...] = (),
) -> Joint
```

The joint kind falls out of `dofs` rather than being named:

| `dofs` | what it is |
| --- | --- |
| `()` | fixed — the bodies are welded |
| one rotational axis | a hinge |
| one linear axis | a slide |
| three rotational axes | a ball joint |
| any other combination | a general joint, exported as USD D6 |

```python
lid_hinge = model.joint(
    "lid_hinge",
    body0=base,
    frame0=JointFrame(xyz=(0.0, -0.04, 0.02)),
    body1=lid,
    frame1=JointFrame(),
    dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 1.57)),),
)
```

`body0` and `body1` are symmetric — neither is the parent. One of them may be
`WORLD`, which pins a body to the ground rather than to another body.

## `model.articulation(...)`

```python
model.articulation(
    name: str,
    *,
    root: RigidBody | Joint | str,
    joints: Iterable[Joint | str] | None = None,
) -> Articulation
```

Names the spanning tree the simulator solves in reduced coordinates. `root` is
the body the tree hangs from, or a `WORLD` joint that anchors it.

`joints` may be omitted when the assembly is one unambiguous tree; list them
when there is a ring, because then you are choosing which joint is left out.

## Closed loops

Count the pivots before writing joints. **A body pinned in two places takes two
joints, and that makes the mechanism a ring rather than a chain.** Linkages,
four-bars, parallel grippers, scissor mechanisms and folding braces all are.

Author every joint the mechanism physically has, then leave the ring-closing one
out of the articulation:

```python
model.joint("ground_left", body0=ground, frame0=JointFrame(), body1=left, frame1=JointFrame(), dofs=swing)
model.joint("left_coupler", body0=left, frame0=JointFrame(xyz=(0.0, 0.0, RISE)), body1=coupler, frame1=JointFrame(), dofs=swing)
model.joint("coupler_right", body0=coupler, frame0=JointFrame(xyz=(SPAN, 0.0, 0.0)), body1=right, frame1=JointFrame(xyz=(0.0, 0.0, RISE)), dofs=swing)
model.joint("ground_right", body0=ground, frame0=JointFrame(xyz=(SPAN, 0.0, 0.0)), body1=right, frame1=JointFrame(), dofs=swing)

model.articulation("main", root=ground, joints=["ground_left", "left_coupler", "coupler_right"])
```

`ground_right` still exports, as a regular USD joint marked
`physics:excludeFromArticulation`, which a physics engine enforces as the pin it
is. `resolved.has_closed_loops` reports whether a ring was found.

Authoring a ring as a chain is a modelling error: the parts export and then flap
loose the moment the object is simulated. See
`docs/sdk/examples/closed_loop_linkage.py`.

**A loop-closing joint cannot be posed.** Its value is decided by the rest of
the mechanism, so `ctx.pose({...})` rejects it — pose the tree joints instead.
Posing a closed loop from joint values is not supported at all; supply body
poses or let a physics engine solve it.

## `PhysicsState`

```python
PhysicsState(body_poses, *, dof_positions=None)
```

Where every body actually is: a 4x4 world transform per body, plus the joint
values those poses imply. Body poses are the authority, so a state that
disagrees with its own joints is rejected rather than quietly reinterpreted.

`resolved.forward_kinematics({"lid_hinge": 0.5})` builds one from joint values,
for a tree. A closed loop cannot be posed that way and needs a full state.

## Frames must meet

The two frames of a joint coincide at rest. If they do not, validation says so
against the axis that disagrees:

```
physics state violates locked axis 'transX' on joint 'crank_coupler': value=-0.3
```

That means the bodies are 0.3 m apart along X where the joint says they are
pinned. Move a frame, or the geometry, until they meet.

## Units

Meters and radians throughout. Rotational limits are radians here and exported
to USD as degrees; the exporter converts. Linear limits are meters both ways.
