from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from build123d.topology import Shape

from articraft.sdk._mesh.core import MeshGeometry
from articraft.sdk.errors import ValidationError
from articraft.sdk.joints import _as_name
from articraft.sdk.mass import MassProperties
from articraft.sdk.materials import Color, Material, _as_color, _as_material

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

    def __post_init__(self) -> None:
        self.name = _as_name(self.name, field_name="rigid body name")
        if self.mass_properties is not None and not isinstance(
            self.mass_properties, MassProperties
        ):
            raise ValidationError(f"rigid body {self.name!r} mass must be MassProperties")

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
        if self.mass_properties is not None and not isinstance(
            self.mass_properties, MassProperties
        ):
            raise ValidationError(f"rigid body {self.name!r} mass must be MassProperties")
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
