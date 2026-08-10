"""Simulation settings: the world the object falls in, and how each part starts.

Neither of these changes the object's shape. They are the numbers a simulator
needs before the first step: which way is down and how hard, and whether a part
is a free rigid body, a scripted one, or asleep.

``PhysicsScene`` belongs to the object because there is one world. ``BodyState``
belongs to a part because each part is one rigid body.

Mini Articraft's own viewer and ``simulate_usdz`` ignore both -- they always run
Earth gravity with every part free and at rest. These values are written for the
downstream simulator that reads the USDZ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk._values import Vec3, _as_vec3, _finite

EARTH_GRAVITY = 9.81


@dataclass(frozen=True, slots=True)
class PhysicsScene:
    """The world an object is simulated in.

    ``direction`` is the way gravity pulls, in world coordinates, and is stored
    normalized. The default is Earth gravity down the stage's Z up axis.
    ``magnitude`` is in m/s^2; zero is a valid free-fall world.
    """

    direction: Vec3 = (0.0, 0.0, -1.0)
    magnitude: float = EARTH_GRAVITY

    def __post_init__(self) -> None:
        direction = _as_vec3(self.direction, field_name="gravity direction")
        length = math.hypot(*direction)
        if length == 0.0:
            raise ValidationError("gravity direction must be non-zero")
        magnitude = _finite(self.magnitude, field_name="gravity magnitude")
        if magnitude < 0.0:
            raise ValidationError("gravity magnitude must not be negative")
        # Normalized on the way in so the authored value, the manifest, and the
        # USD attribute all say the same thing.
        object.__setattr__(self, "direction", tuple(component / length for component in direction))
        object.__setattr__(self, "magnitude", magnitude)


@dataclass(frozen=True, slots=True)
class BodyState:
    """How one part starts, and whether the simulator moves it at all.

    - ``enabled`` off leaves the part in the scene as a static collider: it is
      still there to hit, but no forces act on it.
    - ``kinematic`` on means the part is moved by animation rather than by
      forces. It pushes other bodies; nothing pushes it.
    - ``linear_velocity`` (m/s) and ``angular_velocity`` (rad/s) are the
      velocities at the first step, in world coordinates.
    - ``starts_asleep`` puts the part in the simulator's sleeping set until
      something touches it.
    """

    enabled: bool = True
    kinematic: bool = False
    linear_velocity: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)
    starts_asleep: bool = False

    def __post_init__(self) -> None:
        for field_name in ("enabled", "kinematic", "starts_asleep"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValidationError(f"body state {field_name} must be a bool")
        if not self.enabled and self.kinematic:
            raise ValidationError(
                "a disabled body cannot be kinematic: disabling already stops the "
                "simulator from moving the part"
            )
        object.__setattr__(
            self,
            "linear_velocity",
            _as_vec3(self.linear_velocity, field_name="body state linear_velocity"),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _as_vec3(self.angular_velocity, field_name="body state angular_velocity"),
        )
