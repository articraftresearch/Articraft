"""Live/e2e generation tests backed by the tape lane.

Default: always live (real model). Tapes are opt-in via flags:

    uv run pytest tests/test_live_generation.py --replay
    OPENAI_API_KEY=... uv run pytest tests/test_live_generation.py --record

Tape names default to the test function name
(``tape_model("latest")`` for an explicit name).
"""

from __future__ import annotations

from pathlib import Path

from harness import WarmEnvironment, run_scenario

from articraft.settings import DEFAULT_MAX_TURNS


def test_box_generation(tape_model, tmp_path: Path) -> None:
    with tape_model() as model:
        artifacts = run_scenario(
            "a simple box",
            model=model,
            env=WarmEnvironment(output_dir=tmp_path),
            max_turns=DEFAULT_MAX_TURNS,
        )
    assert artifacts.record.status == "success"
    assert artifacts.record.result.endswith(".usdz")
