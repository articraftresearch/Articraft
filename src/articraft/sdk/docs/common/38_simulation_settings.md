# Simulation settings

These describe the world an object is simulated in and how each part starts moving. They change no
geometry. Articraft's own viewer and `simulate_usdz` ignore them entirely: local runs always
use Earth gravity with every part free and at rest. They are written into the USDZ for a downstream
simulator to read.

```python
from articraft.sdk import BodyState, PhysicsScene
```

`PhysicsScene` belongs to the model, because there is one world. `BodyState` belongs to a part,
because each part is one rigid body. See [articulated objects and
parts](30_articulated_object.md) for the structure these attach to.

## `PhysicsScene`

```python
PhysicsScene(direction: Vec3 = (0.0, 0.0, -1.0), magnitude: float = 9.81)
```

One scene belongs to the whole model. The default is Earth gravity down the stage Z up axis, so
most models never pass it.

```python
from articraft.sdk import ArticulatedObject, PhysicsScene


moon = ArticulatedObject("rover", scene=PhysicsScene(magnitude=1.62))
```

`direction` is a world-space direction and is stored normalized, so its length is ignored. It must
be nonzero. `magnitude` is in m/s^2 and must not be negative; `0.0` is a valid free-fall world. The
default is available as `EARTH_GRAVITY` when a scene needs to say so explicitly.

## `BodyState`

```python
BodyState(
    enabled: bool = True,
    kinematic: bool = False,
    linear_velocity: Vec3 = (0.0, 0.0, 0.0),
    angular_velocity: Vec3 = (0.0, 0.0, 0.0),
    starts_asleep: bool = False,
)
```

Each part is one rigid body, so each part carries one of these. The default is a free body at rest.

```python
from articraft.sdk import BodyState


base = model.part("base", body_state=BodyState(kinematic=True))
flywheel = model.part("flywheel", body_state=BodyState(angular_velocity=(0.0, 0.0, 12.0)))
```

- `enabled=False` leaves the part in the scene as a static collider. Other bodies still hit it, but
  no forces act on it.
- `kinematic=True` means the part is moved by animation rather than by forces. It pushes other
  bodies; nothing pushes it. A disabled part cannot also be kinematic.
- `linear_velocity` is in meters per second and `angular_velocity` is in radians per second, both
  in world coordinates, at the first simulation step. USD stores angular velocity in degrees per
  second; the exporter converts.
- `starts_asleep=True` puts the part in the simulator's sleeping set until something touches it.
