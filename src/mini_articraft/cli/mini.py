from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import typer
from pydantic import ValidationError

from mini_articraft.agent import Agent, events
from mini_articraft.cli.tui import print_settings_error, replay_run, run_live
from mini_articraft.environments import LocalEnvironment
from mini_articraft.models import create_model
from mini_articraft.models.anthropic import SUPPORTED_MODELS as ANTHROPIC_MODELS
from mini_articraft.models.anthropic import anthropic_api_key_value
from mini_articraft.models.anthropic import (
    context_window_tokens_for as anthropic_context_window_tokens_for,
)
from mini_articraft.models.gemini import (
    context_window_tokens_for as gemini_context_window_tokens_for,
)
from mini_articraft.settings import DEFAULT_OUTPUT_DIR, Settings, get_settings
from mini_articraft.viewer import serve_viewer

app = typer.Typer(help="Generate articulated objects with mini-articraft.", add_completion=False)
COMMANDS = {"generate", "replay", "view"}


@app.command()
def generate(
    prompt: str,
    image: Path | None = typer.Option(
        None,
        "--image",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Local reference image for reconstruction.",
    ),
    provider: Literal["openai", "gemini", "anthropic"] | None = typer.Option(
        None,
        "--provider",
        case_sensitive=False,
        help="Model provider to use: openai, gemini, or anthropic.",
    ),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use."),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Run output directory."),
    reasoning_effort: str | None = typer.Option(
        None,
        "--reasoning-effort",
        help="OpenAI reasoning effort.",
    ),
    compile_timeout: float | None = typer.Option(
        None,
        "--compile-timeout",
        min=1.0,
        help="Maximum compile time in seconds.",
    ),
    tui: bool | None = typer.Option(
        None,
        "--tui/--no-tui",
        help="Show the live run UI (default: on when attached to a terminal).",
    ),
) -> None:
    """Generate an object from a prompt."""
    settings = _settings(provider, model, output_dir, reasoning_effort, compile_timeout)
    use_tui = tui if tui is not None else sys.stdout.isatty()
    try:
        if use_tui:
            _generate_with_tui(settings, prompt, image)
        else:
            _print_result(asyncio.run(_generate(settings, prompt, image_path=image)))
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


@app.command()
def replay(
    run: str = typer.Argument(
        ..., help="Run id under the output directory, or a path to a run directory."
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Run output directory."),
    delay: float = typer.Option(0.0, "--delay", help="Pause between events in seconds (TTY only)."),
) -> None:
    """Re-render a recorded run from its conversation log."""
    run_dir = _resolve_run_dir(run, output_dir)
    conversation = run_dir / "conversation.jsonl"
    if not conversation.is_file():
        typer.echo(f"no conversation log at {conversation}", err=True)
        raise typer.Exit(1)

    replay_run(run_dir, delay=delay)


@app.command()
def view(
    run: str = typer.Argument(
        ..., help="Run id under the output directory, or a path to a run directory."
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Run output directory."),
) -> None:
    """Open the articulated USDZ outputs for a run."""
    try:
        serve_viewer(_resolve_run_dir(run, output_dir))
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


async def _generate(
    settings: Settings,
    prompt: str,
    *,
    image_path: Path | None = None,
    on_event: Callable[[events.Event], None] | None = None,
) -> dict[str, Any]:
    model_client = create_model(settings)
    try:
        env = LocalEnvironment(
            output_dir=settings.output_dir,
            timeout_seconds=settings.compile_timeout_seconds,
        )
        agent_kwargs: dict[str, Any] = {"max_turns": settings.max_turns}
        if on_event is not None:
            agent_kwargs["on_event"] = on_event
        return await Agent(model_client, env, **agent_kwargs).run(prompt, image_path=image_path)
    finally:
        # Agent.run closes the model too; close() is idempotent, and this
        # finally covers failures before the agent loop starts.
        await model_client.close()


def _generate_with_tui(settings: Settings, prompt: str, image_path: Path | None) -> None:
    try:
        result = run_live(
            lambda on_event: _generate(
                settings,
                prompt,
                image_path=image_path,
                on_event=on_event,
            )
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        raise typer.Exit(130) from None

    if str(result.get("status")) != "success":
        raise typer.Exit(1)


def _resolve_run_dir(run: str, output_dir: Path | None) -> Path:
    candidate = Path(run)
    return candidate if candidate.is_dir() else (output_dir or _default_output_dir()) / run


def _default_output_dir() -> Path:
    try:
        return get_settings().output_dir
    except ValidationError:
        return DEFAULT_OUTPUT_DIR


def _settings(
    provider: Literal["openai", "gemini", "anthropic"] | None,
    model: str | None,
    output_dir: Path | None,
    reasoning_effort: str | None,
    compile_timeout: float | None,
) -> Settings:
    updates = {
        key: value
        for key, value in (
            ("provider", provider),
            ("output_dir", output_dir),
            ("openai_reasoning_effort", reasoning_effort),
            ("compile_timeout_seconds", compile_timeout),
        )
        if value is not None
    }
    try:
        settings = get_settings()
    except ValidationError as exc:
        _report_settings_error(exc)
        raise typer.Exit(1) from None
    settings = settings.model_copy(update=updates)

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
        print_settings_error(
            detail=(
                "unsupported Anthropic model: "
                f"{settings.anthropic_model}. Supported models: {', '.join(ANTHROPIC_MODELS)}"
            )
        )
        raise typer.Exit(1)

    if (
        settings.provider == "gemini"
        and gemini_context_window_tokens_for(settings.gemini_model) is None
    ):
        print_settings_error(
            detail=(
                "unsupported Gemini model: "
                f"{settings.gemini_model}. Supported models: gemini-3.1-pro-preview, "
                "gemini-3.6-flash"
            )
        )
        raise typer.Exit(1)

    missing = _missing_provider_settings(settings)
    if missing:
        print_settings_error(missing=missing)
        raise typer.Exit(1)
    return settings


def _missing_provider_settings(settings: Settings) -> list[str]:
    if settings.provider == "anthropic":
        return [] if anthropic_api_key_value(settings) else ["ANTHROPIC_API_KEY"]
    if settings.provider == "gemini":
        return [] if (settings.gemini_api_key or "").strip() else ["GEMINI_API_KEY"]
    return [] if settings.openai_api_key else ["OPENAI_API_KEY"]


def _report_settings_error(exc: ValidationError) -> None:
    missing = [
        str(error["loc"][0])
        for error in exc.errors()
        if error.get("type") == "missing" and error.get("loc")
    ]
    if missing:
        print_settings_error(missing=missing)
        return
    print_settings_error(detail=str(exc))


def _print_result(result: dict[str, object]) -> None:
    typer.echo(f"status: {result.get('status', '')}")
    typer.echo(f"run: {result.get('run', '')}")
    if result.get("result"):
        typer.echo(f"result: {result['result']}")
    if result.get("message"):
        typer.echo(str(result["message"]))
    if result.get("error"):
        typer.echo(f"error: {result['error']}", err=True)
    if result.get("status") != "success":
        raise typer.Exit(1)


def main() -> None:
    app(args=_app_args(sys.argv[1:]), prog_name="mini-articraft")


def _app_args(argv: list[str]) -> list[str]:
    if not argv or argv[0] in COMMANDS or argv[0] in {"--help", "-h"}:
        return argv
    return ["generate", *argv]


if __name__ == "__main__":
    main()
