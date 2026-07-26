"""Environment hygiene for processes that run model-authored code."""

from __future__ import annotations

import contextlib
import os


def child_environment() -> dict[str, str]:
    """Copy the host environment without API credentials.

    This keeps model-authored code from receiving provider credentials by
    default. It is hygiene, not isolation: child processes still run with the
    user's OS identity and inherit the remaining environment.
    """
    environment = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
    # The compile worker is a separate process, so settings that gate its checks have
    # to travel as environment. Resolving from Settings keeps .env and the env var
    # equivalent instead of only the latter working.
    from mini_articraft.settings import get_settings

    # Settings needs an API key to load; a child that cannot resolve them keeps
    # whatever the parent environment already said.
    with contextlib.suppress(Exception):
        environment["MINI_ARTICRAFT_PHYSICS"] = "1" if get_settings().physics_enabled else "0"
    return environment
