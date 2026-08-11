from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, cast

from build123d import Axis
from build123d.topology import Shape

from articraft.sdk._mesh.core import MeshGeometry
from articraft.sdk._values import Vec3, _as_name, _as_vec3, _finite
from articraft.sdk.errors import ValidationError
from articraft.sdk.mass import MassProperties
from articraft.sdk.materials import Color, Material, _as_color, _as_material
from articraft.sdk.physics import BodyState

Geometry: TypeAlias = Shape | MeshGeometry

__all__ = ["Geometry", "RigidBody", "RigidBodyRef"]


@dataclass(frozen=True, slots=True)
class _ShapeData:
    name: str
    geometry: Geometry
    material: Material | None
    coating: Material | None
    tint: Color | None

    @property
    def surface_material(self) -> Material | None:
        return self.coating if self.coating is not None else self.material

    @property
    def display_material(self) -> Material | None:
        surface = self.surface_material
        if surface is None:
            if self.tint is None:
                return None
            return Material(name="color", density=1.0, base_color=self.tint)
        return surface if self.tint is None else surface.but(color=self.tint)

    @property
    def color(self) -> Color | None:
        display = self.display_material
        return None if display is None else display.base_color


@dataclass(eq=False)
class RigidBody:
    """One USD rigid body and its geometry, authored in the body's local frame."""

    name: str
    _shapes: dict[str, _ShapeData] = field(default_factory=dict, init=False, repr=False)
    mass_properties: MassProperties | None = field(default=None, kw_only=True)
    body_state: BodyState = field(default=BodyState(), kw_only=True)

    def __post_init__(self) -> None:
        self.name = _as_name(self.name, field_name="rigid body name")
        self._validate_physics()

    def _validate_physics(self) -> None:
        if self.mass_properties is not None and not isinstance(
            self.mass_properties, MassProperties
        ):
            raise ValidationError(f"rigid body {self.name!r} mass must be MassProperties")
        if not isinstance(self.body_state, BodyState):
            raise ValidationError(f"rigid body {self.name!r} body_state must be BodyState")

    def add(
        self,
        shape: Geometry,
        *,
        name: str,
        material: Material | None = None,
        coating: Material | None = None,
        color: Sequence[float] | None = None,
    ) -> Geometry:
        """Add named geometry in this rigid body's local frame."""

        shape_name = _as_name(name, field_name=f"shape name on rigid body {self.name!r}")
        if shape_name in self._shapes:
            raise ValidationError(
                f"duplicate shape name {shape_name!r} on rigid body {self.name!r}"
            )
        _validate_geometry(shape, context=f"rigid body {self.name!r} shape {shape_name!r}")
        self._shapes[shape_name] = _ShapeData(
            name=shape_name,
            geometry=shape,
            material=(
                None
                if material is None
                else _as_material(
                    material,
                    field_name=f"rigid body {self.name!r} shape {shape_name!r}",
                )
            ),
            coating=(
                None
                if coating is None
                else _as_material(
                    coating,
                    field_name=f"rigid body {self.name!r} shape {shape_name!r} coating",
                )
            ),
            tint=(
                None
                if color is None
                else _as_color(
                    color,
                    field_name=f"rigid body {self.name!r} shape {shape_name!r} color",
                )
            ),
        )
        return shape

    def anchor(
        self,
        shape: str,
        *,
        x: float = 0.5,
        y: float = 0.5,
        z: float = 0.5,
        offset: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> Vec3:
        """A point on one of this body's shapes, in the body's own frame.

        Each of ``x``, ``y`` and ``z`` runs 0 to 1 across the shape, so 0.5 is
        the middle. **An extreme value probes the geometry itself**: ``z=1``
        means the surface at the top, at whatever the other two coordinates say,
        not the corner of a bounding box. On a domed lid ``anchor("dome", z=1)``
        is the crown; a bounding box would have put it in the air above the rim.

        Naming an edge or corner sets two or three coordinates to extremes. That
        is exact on a boxy shape, and approximate on a curved one, where the
        combination of separate surface extents need not lie on the surface.
        """

        geometry = self.get_shape(shape)
        low, high = _shape_bounds(geometry)
        fractions = (
            _finite(x, field_name="anchor x"),
            _finite(y, field_name="anchor y"),
            _finite(z, field_name="anchor z"),
        )
        point = [lo + (hi - lo) * f for lo, hi, f in zip(low, high, fractions, strict=True)]
        for axis, fraction in enumerate(fractions):
            if fraction not in (0.0, 1.0):
                continue
            surface = _surface_along(geometry, point, axis, outward=fraction == 1.0)
            if surface is not None:
                point[axis] = surface
        shift = _as_vec3(offset, field_name="anchor offset")
        return cast(Vec3, tuple(value + delta for value, delta in zip(point, shift, strict=True)))

    def get_shape(self, name: str) -> Geometry:
        shape_name = _as_name(name, field_name="shape name")
        entry = self._shapes.get(shape_name)
        if entry is None:
            raise ValidationError(f"unknown shape {shape_name!r} on rigid body {self.name!r}")
        return entry.geometry

    def _iter_shapes(self) -> Iterator[_ShapeData]:
        return iter(self._shapes.values())

    def validate(self) -> None:
        self.name = _as_name(self.name, field_name="rigid body name")
        self._validate_physics()
        if not self._shapes:
            raise ValidationError(f"rigid body {self.name!r} must contain at least one shape")
        for name, entry in self._shapes.items():
            if name != entry.name:
                raise ValidationError(f"rigid body {self.name!r} has an invalid shape name")
            _validate_geometry(entry.geometry, context=f"rigid body {self.name!r} shape {name!r}")
            if entry.material is not None:
                _as_material(
                    entry.material,
                    field_name=f"rigid body {self.name!r} shape {name!r} material",
                )
            if entry.coating is not None:
                _as_material(
                    entry.coating,
                    field_name=f"rigid body {self.name!r} shape {name!r} coating",
                )


RigidBodyRef: TypeAlias = str | RigidBody


def _validate_geometry(shape: object, *, context: str) -> None:
    if isinstance(shape, Shape):
        is_null = shape.is_null
        if is_null() if callable(is_null) else is_null:
            raise ValidationError(f"{context} must be non-empty")
        is_valid = shape.is_valid
        if not (is_valid() if callable(is_valid) else is_valid):
            raise ValidationError(f"{context} must be a valid build123d Shape")
        return
    if isinstance(shape, MeshGeometry):
        try:
            shape.validate()
        except (ValidationError, TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(f"{context} is not valid mesh geometry: {exc}") from exc
        if not shape.vertices or not shape.faces:
            raise ValidationError(f"{context} must be non-empty")
        return
    raise ValidationError(f"{context} must be a build123d Shape or MeshGeometry")


def _shape_bounds(shape: Geometry) -> tuple[Vec3, Vec3]:
    """The axis aligned extent of either kind of geometry."""

    if isinstance(shape, MeshGeometry):
        low, high = shape.bounds
        return (
            cast(Vec3, tuple(float(v) for v in low)),
            cast(Vec3, tuple(float(v) for v in high)),
        )
    box = shape.bounding_box()
    return (
        cast(Vec3, tuple(float(v) for v in box.min)),
        cast(Vec3, tuple(float(v) for v in box.max)),
    )


def _surface_along(
    geometry: Geometry, point: Sequence[float], axis: int, *, outward: bool
) -> float | None:
    """Where the geometry's surface sits along one axis, through ``point``.

    Returns ``None`` when the ray misses, which leaves the caller with the
    bounding box value it already had.
    """

    if isinstance(geometry, MeshGeometry):
        return _mesh_surface_along(geometry, point, axis, outward=outward)
    direction = [0.0, 0.0, 0.0]
    direction[axis] = 1.0
    origin = list(point)
    try:
        # The build123d stubs omit this; it exists at runtime.
        hits = geometry.find_intersection_points(  # pyright: ignore[reportAttributeAccessIssue]
            Axis(tuple(origin), tuple(direction))
        )
    except Exception:
        return None
    values = [float(hit[0].to_tuple()[axis]) for hit in hits]
    if not values:
        return None
    return max(values) if outward else min(values)


def _mesh_surface_along(
    mesh: MeshGeometry, point: Sequence[float], axis: int, *, outward: bool
) -> float | None:
    """The same query for a mesh, read off the vertices near that line.

    A triangle cast would be exact; the vertices in a thin column around the
    line are close enough to place a joint and cost nothing to compute.
    """

    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        return None
    others = [index for index in range(3) if index != axis]
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    span = float(max(high[index] - low[index] for index in others)) or 1.0
    near = np.ones(len(vertices), dtype=bool)
    for index in others:
        near &= np.abs(vertices[:, index] - point[index]) <= span * 0.1
    column = vertices[near]
    if not len(column):
        return None
    return float(column[:, axis].max() if outward else column[:, axis].min())
