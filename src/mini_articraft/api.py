"""Public Python API for generating articulated objects.

Use :func:`generate` from synchronous code and :func:`generate_async` from
asyncio applications. Both functions run the same async agent core and return
a typed :class:`GenerationResult`::

    import mini_articraft

    result = mini_articraft.generate("a desk fan", on_event=print)
    print(result.status, result.artifact)
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, get_args

from mini_articraft.agent import Agent, events
from mini_articraft.environments import LocalEnvironment
from mini_articraft.models import create_model
from mini_articraft.models.anthropic import SUPPORTED_MODELS as ANTHROPIC_MODELS
from mini_articraft.models.anthropic import anthropic_api_key_value
from mini_articraft.models.anthropic import (
    context_window_tokens_for as anthropic_context_window_tokens_for,
)
from mini_articraft.models.gemini import SUPPORTED_MODELS as GEMINI_MODELS
from mini_articraft.models.gemini import (
    context_window_tokens_for as gemini_context_window_tokens_for,
)
from mini_articraft.settings import Settings, get_settings

Provider = Literal["openai", "gemini", "anthropic"]
GenerationStatus = Literal["success", "error"]
Event = events.Event
EventHandler = Callable[[Event], None]
_PROVIDERS: tuple[str, ...] = get_args(Provider)


@dataclass(slots=True)
class GenerationResult:
    """The completed run and its generated artifact, if successful.

    ``run_dir`` is the run directory. ``artifact`` is the generated file
    beneath that directory, or ``None`` when ``status`` is ``"error"``.
    """

    status: GenerationStatus
    run_dir: Path
    artifact: Path | None
    run_id: str = ""
    message: str = ""
    error: str = ""
    attempts: int = 0
    cost: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    compile_report: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the agent produced an artifact."""
        return self.status == "success"


def generate(
    prompt: str,
    *,
    provider: Provider | None = None,
    model: str | None = None,
    image: Path | str | None = None,
    output_dir: Path | str | None = None,
    on_event: EventHandler | None = None,
) -> GenerationResult:
    """Generate an object and block until the run finishes.

    ``on_event`` is called synchronously as the agent reports progress. Asyncio
    applications must use :func:`generate_async` instead. A completed agent
    failure is returned as a result with ``status == "error"``; invalid input
    and failures before a run completes raise exceptions.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "generate() cannot run inside an active event loop; await generate_async() instead"
        )

    return asyncio.run(
        generate_async(
            prompt,
            provider=provider,
            model=model,
            image=image,
            output_dir=output_dir,
            on_event=on_event,
        )
    )


async def generate_async(
    prompt: str,
    *,
    provider: Provider | None = None,
    model: str | None = None,
    image: Path | str | None = None,
    output_dir: Path | str | None = None,
    on_event: EventHandler | None = None,
) -> GenerationResult:
    """Generate an object on the current event loop.

    The coroutine supports normal asyncio cancellation and timeout handling.
    Cancellation takes effect at the next await point, so a synchronous compile
    already in progress may finish first. ``on_event`` runs on the current event
    loop and must not block it.
    """
    settings, image_path = _resolve_request(
        prompt,
        provider=provider,
        model=model,
        image=image,
        output_dir=output_dir,
    )
    payload = await _run_generation(
        settings,
        prompt,
        image_path=image_path,
        on_event=on_event,
    )
    return _result_from_payload(payload)


def _resolve_request(
    prompt: str,
    *,
    provider: Provider | None,
    model: str | None,
    image: Path | str | None,
    output_dir: Path | str | None,
) -> tuple[Settings, Path | None]:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    settings = _resolved_settings(
        get_settings(),
        provider=provider,
        model=model,
        output_dir=Path(output_dir) if output_dir is not None else None,
    )
    if missing := _missing_provider_settings(settings):
        raise ValueError(f"missing required environment variables: {', '.join(missing)}")

    image_path = Path(image) if image is not None else None
    if image_path is not None and not image_path.is_file():
        raise FileNotFoundError(f"reference image not found: {image_path}")
    return settings, image_path


def _resolved_settings(
    base: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
    output_dir: Path | None = None,
    effort: str | None = None,
    compile_timeout: float | None = None,
    physics: bool = False,
) -> Settings:
    """Apply validated CLI or API overrides to ``base``."""
    if provider is not None and provider not in _PROVIDERS:
        raise ValueError(
            f"unsupported provider: {provider}. Supported providers: {', '.join(_PROVIDERS)}"
        )

    selected_provider = provider or base.provider
    updates: dict[str, Any] = {
        key: value
        for key, value in (
            ("provider", provider),
            ("output_dir", output_dir),
            ("openai_reasoning_effort", effort),
            ("compile_timeout_seconds", compile_timeout),
            # The CLI flag only turns the lane on; leaving it off preserves
            # the environment or .env setting.
            ("physics_enabled", True if physics else None),
        )
        if value is not None
    }
    if model is not None:
        model_key = {
            "anthropic": "anthropic_model",
            "gemini": "gemini_model",
            "openai": "openai_model",
        }[selected_provider]
        updates[model_key] = model

    values = base.model_dump()
    values.update(updates)
    settings = Settings.model_validate(values)

    if (
        settings.provider == "anthropic"
        and anthropic_context_window_tokens_for(settings.anthropic_model) is None
    ):
        raise ValueError(
            "unsupported Anthropic model: "
            f"{settings.anthropic_model}. Supported models: {', '.join(ANTHROPIC_MODELS)}"
        )

    if (
        settings.provider == "gemini"
        and gemini_context_window_tokens_for(settings.gemini_model) is None
    ):
        raise ValueError(
            "unsupported Gemini model: "
            f"{settings.gemini_model}. Supported models: {', '.join(GEMINI_MODELS)}"
        )

    return settings


def _missing_provider_settings(settings: Settings) -> list[str]:
    if settings.provider == "anthropic":
        return [] if anthropic_api_key_value(settings) else ["ANTHROPIC_API_KEY"]
    if settings.provider == "gemini":
        return [] if (settings.gemini_api_key or "").strip() else ["GEMINI_API_KEY"]
    return [] if (settings.openai_api_key or "").strip() else ["OPENAI_API_KEY"]


async def _run_generation(
    settings: Settings,
    prompt: str,
    *,
    image_path: Path | None = None,
    on_event: EventHandler | None = None,
) -> dict[str, Any]:
    """Run one agent generation against fully resolved settings."""
    model_client = create_model(settings)
    try:
        env = LocalEnvironment(
            output_dir=settings.output_dir,
            timeout_seconds=settings.compile_timeout_seconds,
            physics_enabled=settings.physics_enabled,
        )
        agent_kwargs: dict[str, Any] = {"max_turns": settings.max_turns}
        if on_event is not None:
            agent_kwargs["on_event"] = on_event
        return await Agent(model_client, env, **agent_kwargs).run(prompt, image_path=image_path)
    finally:
        # Agent.run closes the model too; close() is idempotent, and this
        # finally covers failures before the agent loop starts. Teardown must
        # not replace the generation outcome with a close error.
        with contextlib.suppress(Exception):
            await model_client.close()


def _result_from_payload(payload: dict[str, Any]) -> GenerationResult:
    status = str(payload.get("status") or "")
    if status not in get_args(GenerationStatus):
        raise ValueError(f"unexpected generation status: {status or '<empty>'}")

    run_dir = Path(str(payload.get("run") or ""))
    if not str(run_dir) or str(run_dir) == ".":
        raise ValueError("generation result is missing its run directory")

    result = str(payload.get("result") or "")
    artifact = Path(result) if result else None
    if artifact is not None and not artifact.is_absolute():
        artifact = run_dir / artifact
    if status == "success" and artifact is None:
        raise ValueError("successful generation result is missing its artifact")

    raw_usage = payload.get("token_usage")
    token_usage = (
        {str(key): int(value) for key, value in raw_usage.items()}
        if isinstance(raw_usage, dict)
        else {}
    )
    raw_report = payload.get("compile_report")
    compile_report = dict(raw_report) if isinstance(raw_report, dict) else None

    return GenerationResult(
        status=cast(GenerationStatus, status),
        run_dir=run_dir,
        artifact=artifact,
        run_id=str(payload.get("run_id") or run_dir.name),
        message=str(payload.get("message") or ""),
        error=str(payload.get("error") or ""),
        attempts=int(payload.get("attempts") or 0),
        cost=float(payload.get("cost") or 0.0),
        token_usage=token_usage,
        compile_report=compile_report,
    )


__all__ = [
    "Event",
    "EventHandler",
    "GenerationResult",
    "GenerationStatus",
    "Provider",
    "generate",
    "generate_async",
]
