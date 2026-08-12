"""Shared value types, and the validation every SDK value goes through.

These outlived the joint module that used to own them: a vector is a vector
whether it describes a joint frame, a centre of mass, or a velocity.

``field_name`` is positional-or-keyword on purpose. Callers here grew up in
separate modules with separate habits, and both spellings are in use.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypeAlias, cast

from articraft.sdk.errors import ValidationError

Vec3: TypeAlias = tuple[float, float, float]


def _as_vec3(value: Iterable[float], field_name: str) -> Vec3:
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


def _as_name(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    name = value.strip()
    if not name:
        raise ValidationError(f"{field_name} must be non-empty")
    return name


def _as_identifier(value: object, field_name: str) -> str:
    """A name that survives every namespace it flows into, verbatim.

    Assembly, rigid body, joint, and articulation names become USD prim
    names, manifest keys, MJCF names, and viewer keys, and joint names gain
    a ``.axis`` suffix to form DOF ids. Allowing anything looser means the
    exported prim silently diverges from the authored name, or a dot makes
    one joint's name collide with another's DOF id.
    """

    name = _as_name(value, field_name)
    if not name.isidentifier() or not name.isascii():
        raise ValidationError(
            f"{field_name} must be an identifier: letters, digits, and underscores, "
            f"not starting with a digit; got {name!r}"
        )
    return name


def _finite(value: object, field_name: str) -> float:
    """Every number the SDK accepts passes through here.

    ``OverflowError`` matters as much as ``ValueError``: ``float()`` raises it
    for an int too large to convert, and letting it escape turns a modelling
    mistake into a crash instead of feedback the author can act on. Copies of
    this check that omitted it did exactly that.
    """

    try:
        result = float(cast(str, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{field_name} must be finite")
    return result


def _positive(value: object, field_name: str) -> float:
    result = _finite(value, field_name)
    if result <= 0.0:
        raise ValidationError(f"{field_name} must be positive")
    return result


def _non_negative(value: object, field_name: str) -> float:
    result = _finite(value, field_name)
    if result < 0.0:
        raise ValidationError(f"{field_name} must be non-negative")
    return result
