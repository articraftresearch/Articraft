from __future__ import annotations

from articraft.sdk.errors import MiniArticraftError, SDKError, ValidationError


class ModelError(MiniArticraftError):
    """Raised when the model response cannot be used."""


__all__ = ["MiniArticraftError", "ModelError", "SDKError", "ValidationError"]
