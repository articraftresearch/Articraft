"""Environment hygiene for processes that run model-authored code."""

from __future__ import annotations

import os


def child_environment() -> dict[str, str]:
    """Copy the host environment without API credentials.

    This keeps model-authored code from receiving provider credentials by
    default. It is hygiene, not isolation: child processes still run with the
    user's OS identity and inherit the remaining environment.
    """
    return {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
