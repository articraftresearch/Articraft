"""Environment hygiene for processes that run model-authored code."""

from __future__ import annotations

import os

#: Substrings that mark a variable as a credential. Matched against the
#: whole name rather than a suffix: the host environment carries
#: ``MODAL_TOKEN_ID``, ``MODAL_TOKEN_SECRET`` and ``ROOMS_MODAL_*``, none of
#: which end in ``_API_KEY`` and all of which are live credentials.
_SECRET_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
)


def child_environment() -> dict[str, str]:
    """Copy the host environment without credentials.

    This keeps model-authored code from receiving credentials by default.
    It is hygiene, not isolation: child processes still run with the user's
    OS identity and inherit the remaining environment.
    """
    return {key: value for key, value in os.environ.items() if not _is_secret(key, value)}


def _is_secret(key: str, value: str) -> bool:
    name = key.upper()
    if any(marker in name for marker in _SECRET_MARKERS):
        return True
    # A connection string carries its password inline -- DATABASE_URL is the
    # one that matters here, and it is a credential however it is named.
    return "://" in value and "@" in value.split("://", 1)[1].split("/", 1)[0]
