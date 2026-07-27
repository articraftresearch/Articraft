"""What a shape is for: drawing, colliding, or both.

A shape carries two jobs that usually coincide and sometimes must not. The visible
surface of a kettle is also what it bumps into, so the default role is both. But a
detailed handle is expensive to collide against, and a simplified stand-in for it
should never be drawn, so the two jobs can be split across shapes.
"""

from __future__ import annotations

from enum import StrEnum

from mini_articraft.sdk.errors import ValidationError


class ShapeRole(StrEnum):
    """Whether a shape is drawn, collided against, or both."""

    VISUAL_AND_COLLISION = "visual_and_collision"
    VISUAL = "visual"
    COLLISION = "collision"

    @property
    def is_visual(self) -> bool:
        """Whether the shape is rendered."""
        return self is not ShapeRole.COLLISION

    @property
    def is_collider(self) -> bool:
        """Whether the shape takes part in collision."""
        return self is not ShapeRole.VISUAL


class CollisionApproximation(StrEnum):
    """How a simulator is allowed to simplify a mesh collider.

    The values are the standard ``UsdPhysics`` approximation tokens. Engines
    generally cannot simulate a dynamic body against a raw triangle mesh, so
    ``NONE`` is only appropriate for geometry that never moves.
    """

    CONVEX_DECOMPOSITION = "convexDecomposition"
    CONVEX_HULL = "convexHull"
    BOUNDING_CUBE = "boundingCube"
    BOUNDING_SPHERE = "boundingSphere"
    MESH_SIMPLIFICATION = "meshSimplification"
    NONE = "none"


DEFAULT_COLLISION_APPROXIMATION = CollisionApproximation.CONVEX_DECOMPOSITION


def _as_shape_role(value: object, *, context: str) -> ShapeRole:
    if isinstance(value, ShapeRole):
        return value
    raise ValidationError(f"{context} role must be a ShapeRole")


def _as_approximation(value: object, *, context: str) -> CollisionApproximation:
    if isinstance(value, CollisionApproximation):
        return value
    raise ValidationError(f"{context} collision_approximation must be a CollisionApproximation")


__all__ = [
    "DEFAULT_COLLISION_APPROXIMATION",
    "CollisionApproximation",
    "ShapeRole",
]
