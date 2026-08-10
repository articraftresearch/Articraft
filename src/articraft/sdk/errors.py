from __future__ import annotations


class MiniArticraftError(Exception):
    """Base error for articraft."""


class SDKError(MiniArticraftError):
    """Base error for the articraft SDK."""


class ValidationError(SDKError):
    """Raised when an articulated object definition is invalid."""
