"""Every SDK module validates numbers through one implementation.

These used to be copied per module, and the copies drifted: four of them
omitted ``OverflowError`` from the ``except`` clause, so a value too large to
convert escaped as a crash instead of the ValidationError the author needed.
"""

from __future__ import annotations

import pytest

from articraft.sdk import _values, assembly, frames, mass, materials, testing
from articraft.sdk.errors import ValidationError

TOO_LARGE = 10**400  # an int float() rejects with OverflowError, not ValueError


def test_every_module_shares_one_implementation() -> None:
    assert testing._finite is _values._finite
    assert assembly._finite is _values._finite
    assert testing._vec3 is _values._as_vec3
    assert frames._vec3 is _values._as_vec3
    assert mass._positive is _values._positive
    assert materials._positive is _values._positive
    assert assembly._positive is _values._positive
    assert testing._non_negative is _values._non_negative


@pytest.mark.parametrize(
    ("validate", "value"),
    [
        (_values._finite, TOO_LARGE),
        (_values._positive, TOO_LARGE),
        (_values._non_negative, TOO_LARGE),
        (_values._as_vec3, (TOO_LARGE, 0.0, 0.0)),
    ],
)
def test_a_value_too_large_to_convert_is_a_validation_error(validate, value) -> None:
    """Not an OverflowError: the author needs feedback, not a stack trace."""

    with pytest.raises(ValidationError):
        validate(value, "field")


def test_field_name_may_be_positional_or_keyword() -> None:
    """Both spellings are in use across the modules that share these."""

    assert _values._finite(1.5, "field") == 1.5
    assert _values._finite(1.5, field_name="field") == 1.5
