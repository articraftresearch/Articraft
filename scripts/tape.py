"""Manage generation tapes: list, show, record, replay, erase.

The tape library lives in tests/tapes by default. Recording costs a
real generation (OPENAI_API_KEY); replaying is free. Examples:

    uv run python scripts/tape.py list
    uv run python scripts/tape.py show box
    uv run python scripts/tape.py record box "a small box"     # live, pays
    uv run python scripts/tape.py replay box                    # replays offline
    uv run python scripts/tape.py erase box
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import harness
from harness import TAPE_ROOT, ReplayHarness, WarmEnvironment

app = typer.Typer(help=__doc__, no_args_is_help=True)

Root = Annotated[Path, typer.Option("--root", "-r", help="Tape library root.")]


@app.command()
def list(root: Root = TAPE_ROOT) -> None:
    """List recordings in the library."""
    library = ReplayHarness(root)
    names = library.names()
    if not names:
        typer.echo(f"(no recordings in {library.root})")
        return
    for name in names:
        rows = [row for row in library.entries(name) if "response" in row]
        prompt = library.meta(name).get("prompt", "")
        suffix = f"  prompt={prompt!r}" if prompt else ""
        typer.echo(f"{name}: {len(rows)} exchange(s){suffix}")


@app.command()
def show(name: str, root: Root = TAPE_ROOT) -> None:
    """Show one recording's exchanges."""
    library = ReplayHarness(root)
    meta = library.meta(name)
    if meta:
        typer.echo(f"meta: {json.dumps(meta, default=str)}")
    turn = 0
    for row in library.entries(name):
        if "response" not in row:
            continue
        turn += 1
        response = row["response"]
        tool_names = [call.get("name") for call in response.get("tool_calls") or []]
        preview = (response.get("text") or "")[:80]
        typer.echo(
            f"turn {turn}: tools={tool_names} cost={response.get('cost', 0.0)} text={preview!r}"
        )


@app.command()
def record(
    name: str,
    prompt: str,
    root: Root = TAPE_ROOT,
    max_turns: int = 100,
) -> None:
    """Run a live generation and record it as a tape (pays for model calls)."""
    from mini_articraft.agent import Agent
    from mini_articraft.agent.provider.openai import OpenAIModel
    from mini_articraft.compiler.runner import LocalEnvironment
    from mini_articraft.settings import get_settings

    settings = get_settings()
    library = ReplayHarness(root)
    meta = {"prompt": prompt, "model": settings.openai_model}
    with library.record(name, OpenAIModel(settings), meta=meta) as model:
        result = harness.run(
            Agent(model, LocalEnvironment(), max_turns=max_turns, on_event=_print_event).run(prompt)
        )
    _print_result(result)
    raise typer.Exit(0 if result["status"] == "success" else 1)


@app.command()
def replay(
    name: str,
    root: Root = TAPE_ROOT,
    prompt: Annotated[str | None, typer.Option("--prompt", "-p")] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("runs"),
) -> None:
    """Replay a recorded generation offline, end to end, for free."""
    library = ReplayHarness(root)
    actual_prompt = prompt or library.meta(name).get("prompt")
    if not actual_prompt:
        raise typer.BadParameter(f"tape {name!r} has no recorded prompt; pass --prompt")
    run_id = f"tape-{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    artifacts = harness.run_scenario(
        actual_prompt,
        model=library.replay(name),
        env=WarmEnvironment(output_dir=output_dir),
        run_id=run_id,
        on_event=_print_event,
    )
    finished = artifacts.recorder.finished
    if finished is not None:
        typer.echo(f"turns={finished.turns} duration={finished.duration}s")
    _print_result(artifacts.result)
    raise typer.Exit(0 if artifacts.record.status == "success" else 1)


@app.command()
def erase(name: str, root: Root = TAPE_ROOT) -> None:
    """Remove one recording."""
    library = ReplayHarness(root)
    if library.erase(name):
        typer.echo(f"erased {name}")
        return
    raise typer.BadParameter(f"unknown tape: {name!r}")


def _print_event(event: harness.events.Event) -> None:
    if isinstance(event, harness.events.RunStarted):
        typer.echo(f"run {event.run_id} model={event.model}")
    elif isinstance(event, harness.events.AssistantMessage):
        preview = event.text.strip().splitlines()[0][:80] if event.text.strip() else ""
        tool_names = [call.get("name") for call in event.tool_calls]
        if tool_names:
            typer.echo(f"turn {event.turn}: tools={tool_names}")
        elif preview:
            typer.echo(f"turn {event.turn}: {preview}")


def _print_result(result: dict[str, Any]) -> None:
    typer.echo(
        f"status={result['status']} cost=${result.get('cost', 0.0):.4f} result={result.get('result') or '-'}"
    )
    if result.get("error"):
        typer.echo(f"error={result['error']}")


def main() -> None:
    asyncio.set_event_loop(asyncio.new_event_loop())
    app()


if __name__ == "__main__":
    main()
