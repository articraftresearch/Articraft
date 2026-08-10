from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from build123d.topology import Shape

from articraft.sdk._mesh.core import MeshGeometry
from articraft.sdk.errors import ValidationError
from articraft.sdk.joints import (
    Articulation,
    ArticulationType,
    MotionLimits,
    Origin,
    Vec3,
    _as_name,
    _coerce_part_name,
)
from articraft.sdk.mass import MassProperties
from articraft.sdk.materials import Color, Material, _as_color, _as_material
from articraft.sdk.physics import BodyState, PhysicsScene

Geometry: TypeAlias = Shape | MeshGeometry

__all__ = [
    "ArticulatedObject",
    "BodyState",
    "Color",
    "Geometry",
    "Material",
    "Part",
    "PartRef",
    "PhysicsScene",
]


@dataclass(frozen=True, slots=True)
class _ShapeData:
    name: str
    geometry: Geometry
    material: Material | None
    coating: Material | None
    tint: Color | None

    @property
    def surface_material(self) -> Material | None:
        """What the outside of this shape is: its coating, else its own material.

        Friction and looks are surface properties, so both come from here.
        Density does not: a chrome-plated knob slides like chrome and weighs like
        plastic.
        """
        return self.coating if self.coating is not None else self.material

    @property
    def display_material(self) -> Material | None:
        """The surface as it should be drawn, with any one-off tint applied."""
        surface = self.surface_material
        if surface is None:
            if self.tint is None:
                return None
            return Material(name="color", density=1.0, base_color=self.tint)
        return surface if self.tint is None else surface.but(color=self.tint)

    @property
    def color(self) -> Color | None:
        """Base color, for display-color fallbacks."""
        display = self.display_material
        return None if display is None else display.base_color


@dataclass
class Part:
    name: str
    _shapes: dict[str, _ShapeData] = field(default_factory=dict, init=False, repr=False)
    mass_properties: MassProperties | None = field(default=None, kw_only=True)
    body_state: BodyState = field(default=BodyState(), kw_only=True)

    def __post_init__(self) -> None:
        self.name = _as_name(self.name, field_name="part name")
        self._validate_physics()

    def add(
        self,
        shape: Geometry,
        *,
        name: str,
        material: Material | None = None,
        coating: Material | None = None,
        color: Sequence[float] | None = None,
    ) -> Geometry:
        """Add named geometry in this part's local frame.

        ``material`` says what the shape is made of. It decides the shape's mass,
        how it behaves on contact, and how it looks, so it is usually the only
        thing you need.

        ``coating`` covers the shape in a different material: a rubber grip on a
        steel bar is heavy like steel and grippy like rubber. Friction and looks
        follow the coating; mass stays with the material underneath.

        ``color`` tints the surface of this one shape and changes no physics.
        For anything more, derive a material with ``Material.but(...)`` and give
        it a name to reuse across shapes and parts.
        """

        shape_name = _as_name(name, field_name=f"shape name on part {self.name!r}")
        if shape_name in self._shapes:
            raise ValidationError(f"duplicate shape name {shape_name!r} on part {self.name!r}")
        _validate_geometry(shape, context=f"part {self.name!r} shape {shape_name!r}")
        self._shapes[shape_name] = _ShapeData(
            name=shape_name,
            geometry=shape,
            material=(
                None
                if material is None
                else _as_material(material, field_name=f"part {self.name!r} shape {shape_name!r}")
            ),
            coating=(
                None
                if coating is None
                else _as_material(
                    coating, field_name=f"part {self.name!r} shape {shape_name!r} coating"
                )
            ),
            tint=(
                None
                if color is None
                else _as_color(color, field_name=f"part {self.name!r} shape {shape_name!r} color")
            ),
        )
        return shape

    def get_shape(self, name: str) -> Geometry:
        shape_name = _as_name(name, field_name="shape name")
        entry = self._shapes.get(shape_name)
        if entry is None:
            raise ValidationError(f"unknown shape {shape_name!r} on part {self.name!r}")
        return entry.geometry

    def _iter_shapes(self) -> Iterator[_ShapeData]:
        return iter(self._shapes.values())

    def _validate_physics(self) -> None:
        if self.mass_properties is not None and not isinstance(
            self.mass_properties, MassProperties
        ):
            raise ValidationError(f"part {self.name!r} mass must be MassProperties")
        if not isinstance(self.body_state, BodyState):
            raise ValidationError(f"part {self.name!r} body_state must be BodyState")

    def validate(self) -> None:
        self.name = _as_name(self.name, field_name="part name")
        self._validate_physics()
        if not self._shapes:
            raise ValidationError(f"part {self.name!r} must contain at least one shape")
        for name, entry in self._shapes.items():
            if name != entry.name:
                raise ValidationError(f"part {self.name!r} contains an invalid shape name")
            _validate_geometry(entry.geometry, context=f"part {self.name!r} shape {name!r}")
            if entry.material is not None:
                _as_material(
                    entry.material, field_name=f"part {self.name!r} shape {name!r} material"
                )
            if entry.coating is not None:
                _as_material(entry.coating, field_name=f"part {self.name!r} shape {name!r} coating")


PartRef: TypeAlias = str | Part


@dataclass
class ArticulatedObject:
    name: str
    scene: PhysicsScene = field(default=PhysicsScene(), kw_only=True)
    parts: list[Part] = field(default_factory=list, init=False)
    articulations: list[Articulation] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.name = _as_name(self.name, field_name="object name")
        if not isinstance(self.scene, PhysicsScene):
            raise ValidationError(f"object {self.name!r} scene must be a PhysicsScene")

    @property
    def meters_per_unit(self) -> float:
        return 1.0

    def part(
        self,
        name: str,
        *,
        mass_properties: MassProperties | None = None,
        body_state: BodyState | None = None,
    ) -> Part:
        part = Part(
            name=name,
            mass_properties=mass_properties,
            body_state=BodyState() if body_state is None else body_state,
        )
        if any(existing.name == part.name for existing in self.parts):
            raise ValidationError(f"duplicate part name: {part.name!r}")
        self.parts.append(part)
        return part

    def articulation(
        self,
        name: str,
        articulation_type: ArticulationType | str,
        parent: PartRef,
        child: PartRef,
        *,
        origin: Origin | None = None,
        axis: Vec3 = (0.0, 0.0, 1.0),
        motion_limits: MotionLimits | None = None,
    ) -> Articulation:
        parent_name = _coerce_part_name(parent, field_name="parent")
        child_name = _coerce_part_name(child, field_name="child")
        self.get_part(parent_name)
        self.get_part(child_name)
        articulation = Articulation(
            name=name,
            articulation_type=articulation_type,
            parent=parent_name,
            child=child_name,
            origin=Origin() if origin is None else origin,
            axis=axis,
            motion_limits=motion_limits,
        )
        if any(existing.name == articulation.name for existing in self.articulations):
            raise ValidationError(f"duplicate articulation name: {articulation.name!r}")
        self.articulations.append(articulation)
        return articulation

    def get_part(self, part: PartRef) -> Part:
        name = _coerce_part_name(part, field_name="part")
        for existing in self.parts:
            if existing.name == name:
                return existing
        raise ValidationError(f"unknown part: {name!r}")

    def get_articulation(self, name: str | Articulation) -> Articulation:
        key = (
            name.name
            if isinstance(name, Articulation)
            else _as_name(name, field_name="articulation name")
        )
        for articulation in self.articulations:
            if articulation.name == key:
                return articulation
        raise ValidationError(f"unknown articulation: {key!r}")

    def validate(self) -> None:
        self.name = _as_name(self.name, field_name="object name")
        if not isinstance(self.scene, PhysicsScene):
            raise ValidationError(f"object {self.name!r} scene must be a PhysicsScene")
        if not self.parts:
            raise ValidationError("object must contain at least one part")

        if any(not isinstance(part, Part) for part in self.parts):
            raise ValidationError("object parts must be Part instances")
        for part in self.parts:
            part.validate()
        part_names = [part.name for part in self.parts]
        if len(set(part_names)) != len(part_names):
            raise ValidationError("part names must be unique")

        if any(not isinstance(articulation, Articulation) for articulation in self.articulations):
            raise ValidationError("object articulations must be Articulation instances")
        for articulation in self.articulations:
            articulation.validate()
        articulation_names = [articulation.name for articulation in self.articulations]
        if len(set(articulation_names)) != len(articulation_names):
            raise ValidationError("articulation names must be unique")

        part_name_set = set(part_names)
        child_to_articulation: dict[str, Articulation] = {}
        children: dict[str, list[str]] = {name: [] for name in part_name_set}
        for articulation in self.articulations:
            if articulation.parent not in part_name_set:
                raise ValidationError(
                    f"articulation {articulation.name!r} references missing parent part "
                    f"{articulation.parent!r}"
                )
            if articulation.child not in part_name_set:
                raise ValidationError(
                    f"articulation {articulation.name!r} references missing child part "
                    f"{articulation.child!r}"
                )
            previous = child_to_articulation.get(articulation.child)
            if previous is not None:
                raise ValidationError(
                    f"part {articulation.child!r} has multiple parent articulations: "
                    f"{previous.name!r} and {articulation.name!r}"
                )
            child_to_articulation[articulation.child] = articulation
            children[articulation.parent].append(articulation.child)

        roots = sorted(part_name_set - set(child_to_articulation))
        if not roots:
            raise ValidationError("object has no root part")
        if len(roots) > 1:
            raise ValidationError(f"object must have exactly one root part, found {roots}")

        visited: set[str] = set()
        stack = roots[:]
        while stack:
            part_name = stack.pop()
            if part_name in visited:
                continue
            visited.add(part_name)
            stack.extend(children[part_name])
        if visited != part_name_set:
            raise ValidationError(
                f"object contains unreachable parts: {sorted(part_name_set - visited)}"
            )


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
