from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import trimesh
from build123d import Axis, Location, Plane, Vector
from build123d.topology import Edge, Face, Shape, Vertex

from articraft.sdk._mesh.core import MeshGeometry
from articraft.sdk._values import _as_vec3 as _vec3
from articraft.sdk.errors import ValidationError

if TYPE_CHECKING:
    from articraft.sdk.bodies import RigidBody

Vec3: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class JointFrame:
    """A USD joint frame local to one rigid-body endpoint, in metres and radians."""

    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz", _vec3(self.xyz, field_name="joint frame xyz"))
        object.__setattr__(self, "rpy", _vec3(self.rpy, field_name="joint frame rpy"))


FrameSource: TypeAlias = (
    JointFrame | Location | Plane | Axis | Vector | Shape | MeshGeometry | Sequence[float] | None
)


@dataclass(frozen=True, slots=True)
class _WorldEndpoint:
    def at(self, source: FrameSource = None) -> BodyFrame:
        """Bind a frame to the USD world endpoint."""

        return BodyFrame(self, _as_joint_frame(source))

    def __repr__(self) -> str:
        return "WORLD"


WORLD = _WorldEndpoint()
"""The USD world endpoint. A joint may connect one rigid body to ``WORLD``."""


@dataclass(frozen=True, slots=True)
class BodyFrame:
    """A joint frame bound to the rigid body whose coordinates it uses."""

    body: RigidBody | _WorldEndpoint
    frame: JointFrame = field(default_factory=JointFrame)

    def __post_init__(self) -> None:
        from articraft.sdk.bodies import RigidBody

        if self.body is not WORLD and not isinstance(self.body, RigidBody):
            raise ValidationError("body frame must belong to a RigidBody or WORLD")
        if not isinstance(self.frame, JointFrame):
            raise ValidationError("body frame must contain a JointFrame")

    @property
    def xyz(self) -> Vec3:
        return self.frame.xyz

    @property
    def rpy(self) -> Vec3:
        return self.frame.rpy


def _as_joint_frame(source: FrameSource) -> JointFrame:
    if source is None:
        return JointFrame()
    if isinstance(source, JointFrame):
        return source
    if isinstance(source, Plane | Axis):
        return _location_frame(source.location)
    if isinstance(source, Location):
        return _location_frame(source)
    if isinstance(source, Face):
        return _location_frame(_face_location(source))
    if isinstance(source, Edge):
        return _location_frame(_edge_location(source))
    if isinstance(source, Vertex):
        return JointFrame(xyz=_vec3(source.center(), field_name="vertex center"))
    if isinstance(source, Shape):
        return JointFrame(
            xyz=_vec3(source.bounding_box().center(), field_name="shape bounds center")
        )
    if isinstance(source, MeshGeometry):
        source.validate()
        if not source.vertices:
            raise ValidationError("mesh frame source must contain vertices")
        vertices = np.asarray(source.vertices, dtype=np.float64)
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        return JointFrame(xyz=_vec3(center, field_name="mesh bounds center"))
    return JointFrame(xyz=_vec3(source, field_name="body frame point"))


def _face_location(face: Face) -> Location:
    """The frame a face means when it anchors a joint.

    A rotational surface -- a bore, a pin, a cone seat -- means its axis of
    symmetry, not the outward normal at some surface point: a hinge anchored
    to a cylindrical face should spin about the cylinder, so local Z becomes
    the rotation axis, positioned where the face sits along it. A flat face
    keeps its center and normal.
    """

    rotation_axis = face.axis_of_rotation
    if rotation_axis is None:
        return Plane(origin=face.center(), z_dir=face.normal_at()).location
    direction = Vector(rotation_axis.direction).normalized()
    offset = Vector(face.center()) - Vector(rotation_axis.position)
    origin = Vector(rotation_axis.position) + direction * offset.dot(direction)
    return Axis(origin, direction).location


def _edge_location(edge: Edge) -> Location:
    """The frame an edge means when it anchors a joint.

    A circle or an arc -- a hole rim, a fillet edge -- means the axis through
    its center, normal to its plane; the on-curve tangent it would otherwise
    yield is never the hinge a rim is selected for. A straight or freeform
    edge keeps its midpoint and tangent.
    """

    if str(edge.geom_type).rsplit(".", 1)[-1] in {"CIRCLE", "ELLIPSE"}:
        return Axis(edge.arc_center, edge.normal()).location
    return Axis(edge.center(), edge.tangent_at()).location


def _location_frame(location: Location) -> JointFrame:
    transform = location.wrapped.Transformation()
    matrix = np.identity(4, dtype=np.float64)
    for row in range(3):
        for column in range(4):
            matrix[row, column] = transform.Value(row + 1, column + 1)
    orientation = trimesh.transformations.euler_from_matrix(matrix, axes="sxyz")
    return JointFrame(
        xyz=(float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])),
        rpy=(
            float(orientation[0]),
            float(orientation[1]),
            float(orientation[2]),
        ),
    )


__all__ = ["WORLD", "BodyFrame", "FrameSource", "JointFrame"]
