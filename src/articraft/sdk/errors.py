from __future__ import annotations


class MiniArticraftError(Exception):
    """Base error for articraft."""


class SDKError(MiniArticraftError):
    """Base error for the articraft SDK."""


class ValidationError(SDKError):
    """Raised when an articulated object definition is invalid."""


class LoopClosureError(ValidationError):
    """Raised when a pose cannot keep a closed loop assembled.

    The mechanism was driven past what its linkage allows, so no placement of
    the follower joints keeps the loop's pin closed. The pose is unreachable,
    not merely awkward: tighten the driving joint's motion limits, or fix the
    link geometry that decides the range.
    """
