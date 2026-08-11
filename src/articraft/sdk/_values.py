"""Shared value types, and the validation every SDK value goes through.

These outlived the joint module that used to own them: a vector is a vector
whether it describes a joint frame, a centre of mass, or a velocity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias, cast

from articraft.sdk.errors import ValidationError

Vec3: TypeAlias = tuple[float, float, float]


def _as_vec3(value: Sequence[float], *, field_name: str) -> Vec3:
    if isinstance(value, (str, bytes)):
        raise ValidationError(f"{field_name} must have 3 numeric values")
    try:
        values = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must have 3 numeric values") from exc
    if len(values) != 3:
        raise ValidationError(f"{field_name} must have 3 numeric values")
    if any(not math.isfinite(component) for component in values):
        raise ValidationError(f"{field_name} values must be finite")
    return values


def _as_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    name = value.strip()
    if not name:
        raise ValidationError(f"{field_name} must be non-empty")
    return name


def _optional_finite(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field_name=field_name)


def _positive_finite(value: object, *, field_name: str) -> float:
    result = _finite(value, field_name=field_name)
    if result <= 0.0:
        raise ValidationError(f"{field_name} must be positive")
    return result


def _finite(value: object, *, field_name: str) -> float:
    try:
        result = float(cast(str, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{field_name} must be finite")
    return result
