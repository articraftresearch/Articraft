"""The canonical examples compile cleanly and ship a USDZ.

These pin the intended public SDK path end to end (object model, authored
tests, baseline physical checks, export) and double as the compile
performance probe: watch them with ``--durations`` to spot compile-time
regressions before they reach the agent loop.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from harness import WarmEnvironment

from articraft.compiler.result import CompilePayload

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES = ("hinged_box", "mesh_knob")


def compile_example(env: WarmEnvironment, name: str) -> tuple[CompilePayload, float]:
    run_dir = env.create_run(name)
    shutil.copy(EXAMPLES_DIR / name / "main.py", run_dir / "workspace" / "main.py")
    started = time.perf_counter()
    payload = env.compile_path(run_dir)
    return payload, time.perf_counter() - started


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_compiles_cleanly(tmp_path: Path, name: str) -> None:
    env = WarmEnvironment(output_dir=tmp_path)

    payload, duration = compile_example(env, name)

    assert payload["status"] == "success", payload["compile_report"]["signals_text"]
    assert Path(payload["usdz"]).is_file()
    counts = payload["compile_report"]["counts"]
    assert counts["failures"] == 0, payload["compile_report"]["signals_text"]
    assert counts["warnings"] == 0, payload["compile_report"]["signals_text"]
    print(f"\n{name} compiled in {duration:.2f}s")
