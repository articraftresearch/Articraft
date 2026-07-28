"""What a shape is made of, and what it looks like.

Two orthogonal ideas, deliberately kept apart:

- :class:`Material` is the **substance**: steel, hardwood, rubber. It decides how
  heavy the shape is, how it behaves on contact, and -- unless you say otherwise --
  how it looks.
- :class:`Appearance` is the **surface**: base color, metallic, roughness. It is a
  description of light, not of matter, and it exists so a shape can look like
  something it is not. A chrome-plated knob is plastic that looks like metal; a
  painted steel panel is steel that looks like paint.

Most shapes only need the first. Naming a material gives a plausible appearance
for free, so the common case is a single declaration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from mini_articraft.sdk.errors import ValidationError

Color: TypeAlias = tuple[float, float, float, float]
Rgb: TypeAlias = tuple[float, float, float]


class Material(StrEnum):
    """What a shape is made of.

    One material answers every physical question about a shape: its density
    (and so its mass), its friction and restitution on contact, and the
    appearance it takes unless overridden.
    """

    STEEL = "steel"
    ALUMINUM = "aluminum"
    ABS_PLASTIC = "abs_plastic"
    GLASS = "glass"
    HARDWOOD = "hardwood"
    RUBBER = "rubber"

    @property
    def density(self) -> float:
        """Density in kg/m^3."""
        return _PROPERTIES[self].density

    @property
    def static_friction(self) -> float:
        """Coefficient resisting the start of sliding."""
        return _PROPERTIES[self].static_friction

    @property
    def dynamic_friction(self) -> float:
        """Coefficient resisting sliding already underway."""
        return _PROPERTIES[self].dynamic_friction

    @property
    def restitution(self) -> float:
        """How much of an impact's speed survives the bounce, in [0, 1]."""
        return _PROPERTIES[self].restitution

    @property
    def appearance(self) -> Appearance:
        """The look this material takes when nothing overrides it."""
        return _PROPERTIES[self].appearance


@dataclass(frozen=True, slots=True)
class Appearance:
    """A metallic/roughness surface description in linear-ish authoring space.

    - ``base_color`` is ``(r, g, b, a)`` in ``[0, 1]``; the alpha channel is the
      surface opacity (``a < 1`` makes the shape translucent).
    - ``metallic`` in ``[0, 1]``: ``0`` is a dielectric (plastic, wood, rubber),
      ``1`` is a raw metal.
    - ``roughness`` in ``[0, 1]``: ``0`` is a mirror-smooth surface, ``1`` is
      fully diffuse.
    - ``emissive`` is an optional ``(r, g, b)`` glow color, unlit by the scene.
    """

    base_color: Color = (0.8, 0.8, 0.8, 1.0)
    metallic: float = 0.0
    roughness: float = 0.6
    emissive: Rgb | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "base_color", _as_color(self.base_color, field_name="appearance base_color")
        )
        object.__setattr__(
            self, "metallic", _as_unit(self.metallic, field_name="appearance metallic")
        )
        object.__setattr__(
            self, "roughness", _as_unit(self.roughness, field_name="appearance roughness")
        )
        if self.emissive is not None:
            object.__setattr__(
                self, "emissive", _as_rgb(self.emissive, field_name="appearance emissive")
            )

    @property
    def opacity(self) -> float:
        return self.base_color[3]

    def recolored(self, color: Sequence[float]) -> Appearance:
        """This appearance in a different color, keeping how the surface responds."""
        return Appearance(
            base_color=_as_color(color, field_name="color"),
            metallic=self.metallic,
            roughness=self.roughness,
            emissive=self.emissive,
        )

    # Presets for surfaces that do not follow from a material -- a painted panel,
    # a plated knob, a glowing indicator.

    @classmethod
    def metal(cls, color: Sequence[float] = (0.82, 0.82, 0.85, 1.0), *, roughness: float = 0.35):
        return cls(
            base_color=_as_color(color, field_name="metal color"), metallic=1.0, roughness=roughness
        )

    @classmethod
    def plastic(cls, color: Sequence[float], *, roughness: float = 0.45) -> Appearance:
        return cls(
            base_color=_as_color(color, field_name="plastic color"),
            metallic=0.0,
            roughness=roughness,
        )

    @classmethod
    def rubber(cls, color: Sequence[float], *, roughness: float = 0.9) -> Appearance:
        return cls(
            base_color=_as_color(color, field_name="rubber color"),
            metallic=0.0,
            roughness=roughness,
        )

    @classmethod
    def matte(cls, color: Sequence[float], *, roughness: float = 0.8) -> Appearance:
        return cls(
            base_color=_as_color(color, field_name="matte color"), metallic=0.0, roughness=roughness
        )

    @classmethod
    def glass(
        cls, color: Sequence[float] = (0.9, 0.93, 0.96, 0.25), *, roughness: float = 0.05
    ) -> Appearance:
        return cls(
            base_color=_as_color(color, field_name="glass color"), metallic=0.0, roughness=roughness
        )


def _as_material(value: object, *, field_name: str) -> Material:
    if not isinstance(value, Material):
        raise ValidationError(
            f"{field_name} must be a Material ({', '.join(m.value for m in Material)})"
        )
    return value


def _as_appearance(value: object, *, field_name: str) -> Appearance:
    if not isinstance(value, Appearance):
        raise ValidationError(f"{field_name} must be an Appearance")
    return value


def _as_color(value: Sequence[float], *, field_name: str) -> Color:
    if isinstance(value, (str, bytes)):
        raise ValidationError(f"{field_name} must have 3 or 4 numeric values")
    try:
        raw = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must have 3 or 4 numeric values") from exc
    if len(raw) == 3:
        raw = (*raw, 1.0)
    if len(raw) != 4:
        raise ValidationError(f"{field_name} must have 3 or 4 numeric values")
    if any(not math.isfinite(component) for component in raw):
        raise ValidationError(f"{field_name} values must be finite")
    if any(component < 0.0 or component > 1.0 for component in raw):
        raise ValidationError(f"{field_name} values must be between 0.0 and 1.0")
    return raw


def _as_rgb(value: Sequence[float], *, field_name: str) -> Rgb:
    if isinstance(value, (str, bytes)):
        raise ValidationError(f"{field_name} must have 3 numeric values")
    try:
        raw = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must have 3 numeric values") from exc
    if len(raw) != 3:
        raise ValidationError(f"{field_name} must have 3 numeric values")
    if any(not math.isfinite(component) for component in raw):
        raise ValidationError(f"{field_name} values must be finite")
    if any(component < 0.0 or component > 1.0 for component in raw):
        raise ValidationError(f"{field_name} values must be between 0.0 and 1.0")
    return raw


def _as_unit(value: float, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must be a number between 0.0 and 1.0") from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValidationError(f"{field_name} must be between 0.0 and 1.0")
    return number


@dataclass(frozen=True, slots=True)
class _MaterialProperties:
    density: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    appearance: Appearance


# Densities are standard engineering values. Friction and restitution are
# per-material approximations: a simulator combines the two materials actually in
# contact, so these are inputs to that combination, not pair coefficients.
_PROPERTIES: dict[Material, _MaterialProperties] = {
    Material.STEEL: _MaterialProperties(
        density=7850.0,
        static_friction=0.42,
        dynamic_friction=0.36,
        restitution=0.55,
        appearance=Appearance(base_color=(0.72, 0.73, 0.76, 1.0), metallic=1.0, roughness=0.35),
    ),
    Material.ALUMINUM: _MaterialProperties(
        density=2700.0,
        static_friction=0.45,
        dynamic_friction=0.38,
        restitution=0.40,
        appearance=Appearance(base_color=(0.85, 0.86, 0.88, 1.0), metallic=1.0, roughness=0.28),
    ),
    Material.ABS_PLASTIC: _MaterialProperties(
        density=1050.0,
        static_friction=0.40,
        dynamic_friction=0.32,
        restitution=0.45,
        appearance=Appearance(base_color=(0.80, 0.80, 0.82, 1.0), metallic=0.0, roughness=0.45),
    ),
    Material.GLASS: _MaterialProperties(
        density=2500.0,
        static_friction=0.40,
        dynamic_friction=0.35,
        restitution=0.60,
        appearance=Appearance(base_color=(0.90, 0.93, 0.96, 0.25), metallic=0.0, roughness=0.05),
    ),
    Material.HARDWOOD: _MaterialProperties(
        density=700.0,
        static_friction=0.50,
        dynamic_friction=0.40,
        restitution=0.35,
        appearance=Appearance(base_color=(0.62, 0.45, 0.24, 1.0), metallic=0.0, roughness=0.75),
    ),
    Material.RUBBER: _MaterialProperties(
        density=1200.0,
        static_friction=0.95,
        dynamic_friction=0.85,
        restitution=0.75,
        appearance=Appearance(base_color=(0.12, 0.12, 0.13, 1.0), metallic=0.0, roughness=0.90),
    ),
}
