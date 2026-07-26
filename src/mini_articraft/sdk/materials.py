"""Physically based materials for shapes.

A :class:`Material` describes how a shape's surface responds to light using the
metallic/roughness workflow shared by glTF and USD's ``UsdPreviewSurface``. It
is authored once, stored on a shape, exported into the USDZ package as a bound
``UsdPreviewSurface`` shader, and surfaced to the viewer so metal, plastic,
rubber, and glass read differently instead of looking like flat display colors.

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from mini_articraft.sdk.errors import ValidationError

Color: TypeAlias = tuple[float, float, float, float]
Rgb: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Material:
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
            self, "base_color", _as_color(self.base_color, field_name="material base_color")
        )
        object.__setattr__(
            self, "metallic", _as_unit(self.metallic, field_name="material metallic")
        )
        object.__setattr__(
            self, "roughness", _as_unit(self.roughness, field_name="material roughness")
        )
        if self.emissive is not None:
            object.__setattr__(
                self, "emissive", _as_rgb(self.emissive, field_name="material emissive")
            )

    @property
    def opacity(self) -> float:
        return self.base_color[3]

    # Convenience presets so authored code (and the agent) can name a material
    # by what it is instead of tuning metallic/roughness by hand.

    @classmethod
    def metal(
        cls,
        color: Sequence[float] = (0.82, 0.82, 0.85, 1.0),
        *,
        roughness: float = 0.35,
    ) -> Material:
        return cls(
            base_color=_as_color(color, field_name="metal color"),
            metallic=1.0,
            roughness=roughness,
        )

    @classmethod
    def plastic(
        cls,
        color: Sequence[float],
        *,
        roughness: float = 0.45,
    ) -> Material:
        return cls(
            base_color=_as_color(color, field_name="plastic color"),
            metallic=0.0,
            roughness=roughness,
        )

    @classmethod
    def rubber(
        cls,
        color: Sequence[float],
        *,
        roughness: float = 0.9,
    ) -> Material:
        return cls(
            base_color=_as_color(color, field_name="rubber color"),
            metallic=0.0,
            roughness=roughness,
        )

    @classmethod
    def matte(cls, color: Sequence[float], *, roughness: float = 0.8) -> Material:
        return cls(
            base_color=_as_color(color, field_name="matte color"), metallic=0.0, roughness=roughness
        )

    @classmethod
    def glass(
        cls, color: Sequence[float] = (0.9, 0.93, 0.96, 0.25), *, roughness: float = 0.05
    ) -> Material:
        return cls(
            base_color=_as_color(color, field_name="glass color"), metallic=0.0, roughness=roughness
        )


def _as_material(value: object, *, field_name: str) -> Material:
    if not isinstance(value, Material):
        raise ValidationError(f"{field_name} must be a Material")
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
