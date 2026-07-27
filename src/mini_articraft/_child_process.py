"""Environment hygiene for processes that run model-authored code."""

from __future__ import annotations

import contextlib
import os


def child_environment(*, physics_enabled: bool | None = None) -> dict[str, str]:
    """Copy the host environment without API credentials.

    This keeps model-authored code from receiving provider credentials by
    default. It is hygiene, not isolation: child processes still run with the
    user's OS identity and inherit the remaining environment.
    """
    environment = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
    # The compile worker is a separate process, so settings that gate its checks have
    # to travel as environment. The caller passes the value it resolved (the --physics
    # flag wins); falling back to Settings keeps .env working on its own.
    if physics_enabled is None:
        from mini_articraft.settings import get_settings

        # Settings needs an API key to load; a child that cannot resolve them keeps
        # whatever the parent environment already said.
        with contextlib.suppress(Exception):
            physics_enabled = get_settings().physics_enabled
    if physics_enabled is not None:
        environment["MINI_ARTICRAFT_PHYSICS"] = "1" if physics_enabled else "0"
    return environment
