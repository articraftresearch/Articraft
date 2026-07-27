"""Public Python facade for generating articulated objects.

A small immutable spec around the same plumbing the command line interface
uses: resolve settings, create a model adapter and a local environment, and
run the agent loop.

>>> import mini_articraft
>>> gen = mini_articraft.Generation("a desk fan", provider="anthropic")
>>> run = gen.with_image("reference.png").start()
>>> for event in run.watch():
...     print(event)
>>> result = run.wait()
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import KW_ONLY, dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, get_args

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
_PROVIDERS: tuple[str, ...] = get_args(Provider)


class RunCancelledError(RuntimeError):
    """The generation was cancelled before it finished."""


def resolved_settings(
    base: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
    output_dir: Path | None = None,
    effort: str | None = None,
    compile_timeout: float | None = None,
    physics: bool = False,
) -> Settings:
    """Apply overrides to ``base`` and route ``model`` to the selected provider.

    Raises ``ValueError`` for an unknown provider or an unsupported Anthropic
    or Gemini model.
    """
    if provider is not None and provider not in _PROVIDERS:
        raise ValueError(
            f"unsupported provider: {provider}. Supported providers: {', '.join(_PROVIDERS)}"
        )
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
    settings = base.model_copy(update=updates)

    if model is not None:
        model_key = {
            "anthropic": "anthropic_model",
            "gemini": "gemini_model",
            "openai": "openai_model",
        }[settings.provider]
        settings = settings.model_copy(update={model_key: model})

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


def missing_provider_settings(settings: Settings) -> list[str]:
    if settings.provider == "anthropic":
        return [] if anthropic_api_key_value(settings) else ["ANTHROPIC_API_KEY"]
    if settings.provider == "gemini":
        return [] if (settings.gemini_api_key or "").strip() else ["GEMINI_API_KEY"]
    return [] if settings.openai_api_key else ["OPENAI_API_KEY"]


async def run_generation(
    settings: Settings,
    prompt: str,
    *,
    image_path: Path | None = None,
    on_event: Callable[[events.Event], None] | None = None,
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
        # finally covers failures before the agent loop starts.
        await model_client.close()


@dataclass(frozen=True)
class Generation:
    """An immutable generation spec: pick a provider, model, and image, then run.

    ``with_*`` methods return a new spec, so a configured spec is a safe
    template and one spec can start any number of runs. ``run()`` blocks and
    returns the result dict. ``start()`` returns a :class:`Run` handle for
    watching events while the generation works. Inside a running event loop,
    use ``await generation.run_async()`` instead.

    Settings not covered by a field come from the environment, for example
    ``MINI_ARTICRAFT_MAX_TURNS``. Run ids have second resolution, so two
    runs of the same prompt started in the same second collide.
    """

    prompt: str
    _: KW_ONLY
    provider: str | None = None
    model: str | None = None
    image: Path | str | None = None
    output_dir: Path | str | None = None

    def with_provider(self, provider: str) -> Generation:
        """A copy using this model provider: ``openai``, ``gemini``, or ``anthropic``."""
        return replace(self, provider=provider)

    def with_model(self, model: str) -> Generation:
        """A copy using this model for the selected provider."""
        return replace(self, model=model)

    def with_image(self, path: Path | str) -> Generation:
        """A copy using this local reference image."""
        return replace(self, image=path)

    def with_output_dir(self, path: Path | str) -> Generation:
        """A copy writing runs under this directory instead of ``runs/``."""
        return replace(self, output_dir=path)

    def run(self) -> dict[str, Any]:
        """Run the generation to completion and return the result dict.

        The dict matches the run record: ``status``, ``run`` (the run
        directory), ``result`` (the USDZ path relative to ``run``),
        ``message``, ``error``, ``cost``, and ``token_usage``.
        """
        return self.start().wait()

    def start(self) -> Run:
        """Validate the spec and start a generation in a background thread."""
        settings, image_path = self._resolved()
        return Run(settings, self.prompt, image_path)

    async def run_async(
        self,
        *,
        on_event: Callable[[events.Event], None] | None = None,
    ) -> dict[str, Any]:
        """Run the generation on the current event loop."""
        settings, image_path = self._resolved()
        return await run_generation(
            settings,
            self.prompt,
            image_path=image_path,
            on_event=on_event,
        )

    def _resolved(self) -> tuple[Settings, Path | None]:
        settings = resolved_settings(
            get_settings(),
            provider=self.provider,
            model=self.model,
            output_dir=Path(self.output_dir) if self.output_dir is not None else None,
        )
        if missing := missing_provider_settings(settings):
            raise ValueError(f"missing required environment variables: {', '.join(missing)}")
        image_path = Path(self.image) if self.image is not None else None
        if image_path is not None and not image_path.is_file():
            raise FileNotFoundError(f"reference image not found: {image_path}")
        return settings, image_path


class Run:
    """A running generation: watch its events and wait for its result.

    A run is also a context manager: leaving the ``with`` block waits for the
    run to finish, and an exception inside the block cancels it first.
    """

    def __init__(self, settings: Settings, prompt: str, image_path: Path | None):
        self._events: queue.SimpleQueue[events.Event | None] = queue.SimpleQueue()
        self._result: dict[str, Any] | None = None
        self._error: BaseException | None = None
        self._drained = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[dict[str, Any]] | None = None
        self._cancel_requested = False
        self._thread = threading.Thread(
            target=self._work,
            args=(settings, prompt, image_path),
            daemon=True,
        )
        self._thread.start()

    def __enter__(self) -> Run:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.cancel()
        self._thread.join()

    def _work(self, settings: Settings, prompt: str, image_path: Path | None) -> None:
        try:
            self._result = asyncio.run(self._main(settings, prompt, image_path))
        except asyncio.CancelledError:
            self._error = RunCancelledError("generation cancelled")
        except BaseException as exc:  # re-raised by wait()
            self._error = exc
        finally:
            self._events.put(None)

    async def _main(
        self, settings: Settings, prompt: str, image_path: Path | None
    ) -> dict[str, Any]:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        if self._cancel_requested:
            raise asyncio.CancelledError
        return await run_generation(
            settings, prompt, image_path=image_path, on_event=self._events.put
        )

    def cancel(self) -> None:
        """Ask the run to stop; ``wait()`` then raises :class:`RunCancelledError`.

        Cancellation lands at the next await point, so a compile already in
        progress finishes or times out first. Safe to call more than once or
        after the run has finished.
        """
        self._cancel_requested = True
        loop, task = self._loop, self._task
        if loop is None or task is None:
            return
        # The loop closes when the run finishes; a late cancel is a no-op.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(task.cancel)

    @property
    def done(self) -> bool:
        """Whether the generation has finished."""
        return not self._thread.is_alive()

    def watch(self) -> Iterator[events.Event]:
        """Yield run events until the generation finishes.

        Events are buffered, so a watcher that starts late still sees every
        event. Use one watcher per run: each event goes to one consumer.
        """
        while not self._drained:
            event = self._events.get()
            if event is None:
                self._drained = True
                break
            yield event

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the generation finishes and return the result dict.

        Raises ``TimeoutError`` when ``timeout`` seconds pass first, and
        re-raises the generation's exception when it failed to run.
        """
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError(f"generation still running after {timeout:g}s")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


__all__ = [
    "Generation",
    "Provider",
    "Run",
    "RunCancelledError",
    "missing_provider_settings",
    "resolved_settings",
    "run_generation",
]
