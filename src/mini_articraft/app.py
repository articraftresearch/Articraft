from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

import typer
from pydantic import ValidationError

from mini_articraft import api
from mini_articraft.agent.record import Record
from mini_articraft.compiler.worker import texture_run
from mini_articraft.settings import DEFAULT_OUTPUT_DIR, Settings, get_settings
from mini_articraft.tui import print_settings_error, replay_run, run_live
from mini_articraft.viewer import load_viewer_run, serve_viewer

cli = typer.Typer(help="Generate articulated objects with mini-articraft.", add_completion=False)
COMMANDS = {"generate", "replay", "view", "simulate", "texture"}


@cli.command()
def generate(
    prompt: str,
    image: Path | None = typer.Option(
        None,
        "--image",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Local reference image for reconstruction.",
    ),
    source: Path | None = typer.Option(
        None,
        "--source",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Existing run main.py to modify instead of starting from scratch.",
    ),
    provider: Literal["openai", "gemini", "anthropic", "openrouter"] | None = typer.Option(
        None,
        "--provider",
        case_sensitive=False,
        help="Model provider to use: openai, gemini, anthropic, or openrouter.",
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
    if image is not None and not image.exists():
        typer.echo(f"reference image does not exist: {image}", err=True)
        raise typer.Exit(2)
    if source is not None and not source.exists():
        typer.echo(f"source script does not exist: {source}", err=True)
        raise typer.Exit(2)
    settings = _settings(provider, model, output_dir, effort, compile_timeout, physics)
    if image is not None and settings.provider == "openrouter":
        typer.echo("OpenRouter does not support reference images.", err=True)
        raise typer.Exit(1)
    use_tui = tui if tui is not None else sys.stdout.isatty()
    try:
        result = _run_generation(
            settings,
            prompt,
            image,
            source,
            use_tui=use_tui,
        )
        if textures:
            _apply_textures(result)
        if not use_tui:
            _print_result(result)
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


@cli.command()
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


@cli.command()
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


@cli.command()
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


@cli.command()
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
        if outcome.usdz is not None:
            _point_record_at(run_dir, outcome.usdz)
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


def _run_generation(
    settings: Settings,
    prompt: str,
    image_path: Path | None,
    source_path: Path | None = None,
    *,
    use_tui: bool,
) -> dict[str, Any]:
    if use_tui:
        return _generate_with_tui(settings, prompt, image_path, source_path)
    return asyncio.run(
        api._run_generation(settings, prompt, image_path=image_path, source_path=source_path)
    )


def _generate_with_tui(
    settings: Settings,
    prompt: str,
    image_path: Path | None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    try:
        result = run_live(
            lambda on_event: api._run_generation(
                settings,
                prompt,
                image_path=image_path,
                source_path=source_path,
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
            run_dir = Path(str(run)).resolve()
            result["result"] = outcome.usdz.relative_to(run_dir).as_posix()
            _point_record_at(run_dir, outcome.usdz)
        typer.echo(
            f"applied texture maps to {outcome.textured_shapes}/{outcome.requested_shapes} surfaces"
        )
        for error in outcome.errors:
            typer.echo(f"note: {error}", err=True)
    else:
        detail = outcome.error or "; ".join(outcome.errors) or "no textures were requested"
        typer.echo(f"note: kept parametric result ({detail})", err=True)


def _point_record_at(run_dir: Path, usdz: Path) -> None:
    """Aim the run's record at a newly exported usdz.

    The compiler produces the file and says where it went; the record is the
    agent's trace, so updating it belongs to whoever drove the run.

    A trace that cannot be updated is worth saying out loud but not worth
    failing over: the export already succeeded and the usdz is on disk. The
    compiler used to do this inside its own try/except, which reported the
    whole texture run as failed when only the bookkeeping broke.
    """

    record_path = run_dir / "record.json"
    if not record_path.is_file():
        return
    try:
        record = Record.load(record_path)
        record.result = usdz.relative_to(run_dir).as_posix()
        record.save(record_path)
    except (OSError, ValueError) as exc:
        typer.echo(f"note: could not update {record_path.name} ({exc})", err=True)


def _resolve_run_dir(run: str, output_dir: Path | None) -> Path:
    candidate = Path(run)
    return candidate if candidate.is_dir() else (output_dir or _default_output_dir()) / run


def _default_output_dir() -> Path:
    try:
        return get_settings().output_dir
    except ValidationError:
        return DEFAULT_OUTPUT_DIR


def _settings(
    provider: Literal["openai", "gemini", "anthropic", "openrouter"] | None,
    model: str | None,
    output_dir: Path | None,
    effort: str | None,
    compile_timeout: float | None,
    physics: bool = False,
) -> Settings:
    try:
        settings = get_settings()
    except ValidationError as exc:
        _report_settings_error(exc)
        raise typer.Exit(1) from None

    try:
        settings = api._resolved_settings(
            settings,
            provider=provider,
            model=model,
            output_dir=output_dir,
            effort=effort,
            compile_timeout=compile_timeout,
            physics=physics,
        )
    except ValueError as exc:
        print_settings_error(detail=str(exc))
        raise typer.Exit(1) from None

    if missing := api._missing_provider_settings(settings):
        print_settings_error(missing=missing)
        raise typer.Exit(1)
    return settings


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
    cli(args=_app_args(sys.argv[1:]), prog_name="mini-articraft")


def _app_args(argv: list[str]) -> list[str]:
    if not argv or argv[0] in COMMANDS or argv[0] in {"--help", "-h"}:
        return argv
    return ["generate", *argv]


if __name__ == "__main__":
    main()
