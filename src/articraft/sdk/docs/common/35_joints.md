# Joints and articulations

A `Joint` is a physical constraint between two endpoints. An `Articulation` selects a joint
tree for the motion solver.

This separation lets one physical assembly contain closed loops.

```python
from articraft.sdk import WORLD, BodyFrame, JointAxis, JointDOF, JointFrame
```

## `JointFrame`

Use `JointFrame` for a local position and orientation:

```python
JointFrame(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
)
```

Each joint has one local frame at each endpoint. The frames coincide in the authored zero
state.

`xyz` uses meters. `rpy` uses extrinsic XYZ roll, pitch, and yaw in radians.

Usually, create a `BodyFrame` with `body.at(...)` or `WORLD.at(...)`. This binds the local
`JointFrame` to its endpoint.

The method accepts points and build123d features. The natural feature axis becomes the local Z
axis of the frame.

Use `ROT_Z` or `TRANS_Z` to move along that selected feature. Rotate the frame when the motion
axis does not align with body axes.

## `JointDOF` and `JointAxis`

Use `JointDOF` to make one joint axis free or limited:

```python
JointDOF(axis: JointAxis, limits: tuple[float, float] | None = None)
```

`JointAxis` contains these six axes:

- `TRANS_X`, `TRANS_Y`, and `TRANS_Z`.
- `ROT_X`, `ROT_Y`, and `ROT_Z`.

All unlisted axes are locked. Translation limits use meters. Rotation limits use radians.

Each limit range must include zero. Omit `limits` when the axis has no bounds.

## `model.joint(...)`

Create a physical joint with two bound endpoints:

```python
model.joint(
    name: str,
    endpoint0: BodyFrame,
    endpoint1: BodyFrame,
    *,
    dofs: Iterable[JointDOF] = (),
) -> Joint
```

The endpoints are equal peers. Neither endpoint is the parent. One endpoint can use
`WORLD.at(...)`.

A joint cannot connect `WORLD` to itself.

| Degrees of freedom | Physical joint |
| --- | --- |
| None | Fixed |
| One rotation axis | Revolute |
| One translation axis | Prismatic |
| Another combination | Generic D6 |

```python
door_hinge = model.joint(
    "door_hinge",
    base.at((0.0, -0.04, 0.02)),
    door.at((0.0, -0.04, 0.0)),
    dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 1.57)),),
)
```

## Removable bodies

A separate body does not always need retained hardware. A teapot lid can sit on its mating
rim and lift away freely. Use a generic D6 relation with all six axes free when the body must
remain in the assembly graph:

```python
free_motion = tuple(JointDOF(axis) for axis in JointAxis)
removable_lid = model.joint(
    "removable_lid",
    pot.at((0.0, 0.0, 0.12)),
    lid.at((0.0, 0.0, 0.0)),
    dofs=free_motion,
)
```

This relation describes the possible poses. It does not imply a physical hinge, latch, or
other retention hardware. Model the seated rim and clearance in the geometry.

## `model.articulation(...)`

Select the tree that a motion solver uses:

```python
model.articulation(
    name: str,
    *,
    root: RigidBody | Joint | str,
    joints: Iterable[Joint | str] | None = None,
) -> Articulation
```

Use a body as the root of a floating articulation. Use a world joint as the root of a fixed
articulation.

Do not include another world joint in a floating articulation. Use that joint as the root when
you must anchor the assembly.

The selected joints must form one connected tree without a cycle. Specify the joint list for
cycles or multiple articulations.

The SDK can infer the list only for one unambiguous tree. A body or joint can belong to only
one articulation.

## Closed loops

A four bar mechanism has four physical joints. Its articulation tree has three joints.

Author all four constraints and select three:

```python
swing = (JointDOF(JointAxis.ROT_Y, limits=(-1.5, 1.5)),)

ground_crank = model.joint(
    "ground_crank",
    ground.at(),
    crank.at(),
    dofs=swing,
)
crank_coupler = model.joint(
    "crank_coupler",
    crank.at((CRANK_LENGTH, 0.0, 0.0)),
    coupler.at(),
    dofs=swing,
)
ground_rocker = model.joint(
    "ground_rocker",
    ground.at((GROUND_SPAN, 0.0, 0.0)),
    rocker.at(),
    dofs=swing,
)

model.articulation(
    "main",
    root=ground,
    joints=(ground_crank, crank_coupler, ground_rocker),
)

coupler_tip = coupler.at((COUPLER_LENGTH, 0.0, 0.0))
model.joint(
    "closing_pin",
    coupler_tip,
    model.frame_in(coupler_tip, rocker),
    dofs=swing,
)
```

`closing_pin` remains a physical USD constraint. Resolution sets
`exclude_from_articulation=True` for this joint.

Do not pose a closing joint directly. Supply a tree degree of freedom to
`forward_kinematics(...)` or `TestContext.pose(...)`.

The loop solver finds the other positions that close the mechanism. It raises
`LoopClosureError` when the pose is not reachable.

Use `TestContext.expect_coaxial(...)` when the closing pin uses a build123d edge or axis. This
check confirms that the closure stays on the geometry feature.

```python
state = model.resolve().forward_kinematics({"ground_crank.rotY": 0.4})
```

A degree of freedom identifier contains the joint name and axis value. For example,
`ground_crank.rotY` identifies the rotation above.

`PhysicsState.dof_positions` includes derived values for closing joints. The solver uses the
authored D6 constraints, but it does not simulate dynamics.

Closed loop simulation stability depends on the physics engine, time step, and solver
settings.

## Authoritative states

Use `PhysicsState` when you already have the world pose of every body:

```python
PhysicsState(body_poses, *, dof_positions=None)
```

`body_poses` maps every body name to a 4 by 4 world transform. These transforms are
authoritative.

Resolution checks the transform against locked axes and limits. It then derives degree of
freedom metadata.

Use a complete state for poses from a simulator. Also use it when one articulation tree cannot
define the whole assembly.

```python
state = model.physics_state(body_world_transforms)
with TestContext(model).state(state):
    ...
```

An assembly does not need an articulation. Without one, export writes each joint as a regular
constraint.
