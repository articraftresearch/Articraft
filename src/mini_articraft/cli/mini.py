from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import typer
from pydantic import ValidationError

from mini_articraft.agent import Agent, events
from mini_articraft.cli.tui import print_settings_error, replay_run, run_live
from mini_articraft.environments import LocalEnvironment
from mini_articraft.environments.worker import texture_run
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
from mini_articraft.viewer import load_viewer_run, serve_viewer

app = typer.Typer(help="Generate articulated objects with mini-articraft.", add_completion=False)
COMMANDS = {"generate", "replay", "view", "simulate", "texture"}


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
    effort: str | None = typer.Option(
        None,
        "--effort",
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
    textures: bool = typer.Option(
        False,
        "--textures",
        help="Apply texture maps to the generated result.",
    ),
    physics: bool = typer.Option(
        False,
        "--physics",
        help="Require every part to declare mass properties, and export them.",
    ),
) -> None:
    """Generate an object from a prompt."""
    settings = _settings(provider, model, output_dir, effort, compile_timeout, physics)
    use_tui = tui if tui is not None else sys.stdout.isatty()
    try:
        result = _run_generation(
            settings,
            prompt,
            image,
            use_tui=use_tui,
        )
        if textures:
            _apply_textures(result)
        if not use_tui:
            _print_result(result)
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


@app.command()
def simulate(
    run: str = typer.Argument(
        ..., help="Run id under the output directory, or a path to a run directory."
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Run output directory."),
    seconds: float = typer.Option(3.0, "--seconds", help="How long to simulate."),
    scenario: str = typer.Option(
        "drop",
        "--scenario",
        help="'drop' to settle on a floor, 'tilt' to find where it slides, "
        "'release' to let the joints fall.",
    ),
) -> None:
    """Run a run's latest USDZ in a physics engine and report whether it behaves.

    OpenUSD validation says the stage is well formed; this says whether the
    object stands up, stays together, and settles. 'tilt' tips the floor until it
    slides, which measures the friction its materials authored. 'release' lets
    every joint fall from mid-travel, which is the motion worth watching.
    """
    from mini_articraft.simulate import SimulationUnavailable, simulate_usdz

    run_dir = _resolve_run_dir(run, output_dir)
    try:
        viewer_run = load_viewer_run(run_dir)
        usdz = viewer_run.files[str(viewer_run.versions[0]["id"])]
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    try:
        simulation_dir = run_dir / "result" / "simulation"
        result = simulate_usdz(usdz, simulation_dir, seconds=seconds, scenario=scenario)
    except SimulationUnavailable as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except ValueError as exc:
        typer.echo(f"could not simulate {usdz.name}: {exc}", err=True)
        raise typer.Exit(1) from None

    if result.trajectory is not None:
        # Keyed by USDZ so the viewer plays the motion belonging to the version
        # it is showing.
        record = simulation_dir / f"{usdz.stem}.trajectory.json"
        record.write_text(json.dumps(result.trajectory.to_payload()), encoding="utf-8")
        typer.echo(f"recorded motion for the viewer: {record}")

    typer.echo(result.summary())
    if not result.stood_up:
        raise typer.Exit(1)


@app.command()
def texture(
    run: str = typer.Argument(
        ..., help="Run id under the output directory, or a path to a run directory."
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Run output directory."),
) -> None:
    """Apply texture maps to an already-generated run.

    Re-exports the run's result with texture maps, so generation can stay local
    and a completed run can be enhanced afterward.
    """
    run_dir = _resolve_run_dir(run, output_dir)
    if not (run_dir / "workspace" / "main.py").is_file():
        typer.echo(f"no generated run at {run_dir}", err=True)
        raise typer.Exit(1)
    outcome = texture_run(run_dir)
    if outcome.applied:
        typer.echo(
            f"applied texture maps to {outcome.textured_shapes}/"
            f"{outcome.requested_shapes} surfaces in {run_dir}"
        )
        for error in outcome.errors:
            typer.echo(f"note: {error}", err=True)
        typer.echo(f"view it:  uv run mini-articraft view {run}")
    else:
        detail = outcome.error or "; ".join(outcome.errors) or "no textures were requested"
        typer.echo(f"could not apply texture maps to {run_dir}: {detail}", err=True)
        raise typer.Exit(1)


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


def _run_generation(
    settings: Settings,
    prompt: str,
    image_path: Path | None,
    *,
    use_tui: bool,
) -> dict[str, Any]:
    if use_tui:
        return _generate_with_tui(settings, prompt, image_path)
    return asyncio.run(_generate(settings, prompt, image_path=image_path))


def _generate_with_tui(settings: Settings, prompt: str, image_path: Path | None) -> dict[str, Any]:
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
    return result


def _apply_textures(result: dict[str, Any]) -> None:
    """Re-export a completed run with texture maps when available.

    A failure keeps the parametric result and does not change generation status.
    """

    if str(result.get("status")) != "success":
        return
    run = result.get("run")
    if not run:
        return
    outcome = texture_run(Path(str(run)))
    if outcome.applied:
        if outcome.usdz is not None:
            result["result"] = outcome.usdz.relative_to(Path(str(run)).resolve()).as_posix()
        typer.echo(
            f"applied texture maps to {outcome.textured_shapes}/{outcome.requested_shapes} surfaces"
        )
        for error in outcome.errors:
            typer.echo(f"note: {error}", err=True)
    else:
        detail = outcome.error or "; ".join(outcome.errors) or "no textures were requested"
        typer.echo(f"note: kept parametric result ({detail})", err=True)


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
    effort: str | None,
    compile_timeout: float | None,
    physics: bool = False,
) -> Settings:
    updates = {
        key: value
        for key, value in (
            ("provider", provider),
            ("output_dir", output_dir),
            ("openai_reasoning_effort", effort),
            ("compile_timeout_seconds", compile_timeout),
            # The flag only turns the lane on; leaving it off keeps whatever the
            # environment or .env already said.
            ("physics_enabled", True if physics else None),
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
