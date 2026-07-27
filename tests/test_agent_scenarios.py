"""Deep agent-loop scenarios: the full pipeline with a scripted model, for $0.

Every scenario runs the real tools and the real compile worker (via
``LocalEnvironment``), so the model-facing machinery --
compile signals, repeat-failure guidance, allowances, reminders, the run
record, the event stream -- is exercised end to end without a paid
generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness import (
    GOOD_MAIN_PY,
    EventRecorder,
    ModelQuery,
    ReplayHarness,
    Response,
    WarmEnvironment,
    calls,
    run_scenario,
    text,
    tool_call,
)

from mini_articraft.agent.events import AssistantMessage, RunFinished, RunStarted, ToolStarted

BROKEN_NO_RUN_TESTS = """
from build123d import Box

from mini_articraft.sdk import ArticulatedObject


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("box")
    base = model.part("base")
    base.add(Box(0.2, 0.2, 0.1), name="body")
    return model


object_model = build_object_model()
"""

OVERLAP_MAIN = """
from build123d import Box

from mini_articraft.sdk import ArticulatedObject, TestContext, TestReport


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("press_fit")
    base = model.part("base")
    base.add(Box(0.2, 0.2, 0.1), name="body")
    pin = model.part("pin")
    pin.add(Box(0.05, 0.05, 0.2), name="body")
    model.articulation("press_fit_pin", "fixed", parent="base", child="pin")
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.expect_no_collision("base", "pin", shape_a="body", shape_b="body")
    return ctx.report()
"""

OVERLAP_ALLOWED_MAIN = OVERLAP_MAIN.replace(
    '    ctx.expect_no_collision("base", "pin", shape_a="body", shape_b="body")',
    """    ctx.allow_overlap(
        "base",
        "pin",
        reason="intentional press-fit embed",
        shape_a="body",
        shape_b="body",
    )""",
)


def write_main(content: str) -> Response:
    return calls(tool_call("write", {"path": "main.py", "content": content}))


def compile_workspace() -> Response:
    return calls(tool_call("compile"))


def compile_signals_shown(tool_outputs: list[dict[str, Any]]) -> list[str]:
    """The rendered <compile_signals> blocks the model was shown, in order."""
    return [
        output["result"]["compile_signals"]
        for output in tool_outputs
        if isinstance(output.get("result"), dict) and "compile_signals" in output["result"]
    ]


def event_signal_codes(recorder: EventRecorder) -> list[str]:
    """Machine-readable signal codes from the full compile results in events."""
    codes: list[str] = []
    for finished in recorder.tool_finishes("compile"):
        bundle = finished.payload["result"]["compile_report"]["signal_bundle"]
        codes.extend(signal["code"] for signal in bundle["signals"])
    return codes


def test_agent_repairs_a_missing_run_tests_with_real_signals(tmp_path: Path) -> None:
    env = WarmEnvironment(output_dir=tmp_path)

    def repair(query: ModelQuery) -> Response:
        signals = compile_signals_shown(query.tool_outputs())
        assert signals and "[missing_run_tests]" in signals[-1]
        return write_main(GOOD_MAIN_PY)

    artifacts = run_scenario(
        "a box",
        [
            write_main(BROKEN_NO_RUN_TESTS),
            compile_workspace(),
            repair,
            compile_workspace(),
            text("done"),
        ],
        env=env,
        max_turns=5,
    )

    assert artifacts.record.status == "success"
    assert artifacts.result["message"] == "done"
    assert env.compile_count == 2
    codes = event_signal_codes(artifacts.recorder)
    assert "COMPILE_MISSING_RUN_TESTS" in codes


def test_repeat_failure_guidance_escalates_across_compiles(tmp_path: Path) -> None:
    env = WarmEnvironment(output_dir=tmp_path)

    artifacts = run_scenario(
        "a box",
        [
            write_main(BROKEN_NO_RUN_TESTS),
            compile_workspace(),
            write_main(BROKEN_NO_RUN_TESTS),
            compile_workspace(),
            write_main(BROKEN_NO_RUN_TESTS),
            compile_workspace(),
            write_main(GOOD_MAIN_PY),
            compile_workspace(),
            text("done"),
        ],
        env=env,
        max_turns=9,
    )

    assert artifacts.record.status == "success"
    assert env.compile_count == 4
    signals = compile_signals_shown(artifacts.tool_outputs())
    assert len(signals) == 4
    assert "matches the previous compile attempt" not in signals[0]
    assert "matches the previous compile attempt" in signals[1]
    assert "compile failure 3 in a row" in signals[2]


def test_a_cached_success_does_not_recompile(tmp_path: Path) -> None:
    """Reads and no-op inspections keep a fresh compile: no second worker call."""
    env = WarmEnvironment(output_dir=tmp_path)

    artifacts = run_scenario(
        "a box",
        [
            write_main(GOOD_MAIN_PY),
            compile_workspace(),
            calls(tool_call("read", {"path": "main.py"})),
            text("done"),
        ],
        env=env,
        max_turns=4,
    )

    assert artifacts.record.status == "success"
    assert env.compile_count == 1


def test_agent_can_extend_locally_and_inspect_visual_evidence(tmp_path: Path) -> None:
    env = WarmEnvironment(output_dir=tmp_path)
    helper = """from build123d import Box


def body_shape():
    return Box(0.4, 0.3, 0.2)
"""
    main = """from geometry_helpers import body_shape
from mini_articraft.sdk import ArticulatedObject, TestContext, TestReport

object_model = ArticulatedObject("local_extension")
object_model.part("body").add(body_shape(), name="shell")

def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    metrics = ctx.measure_geometry()
    ctx.expect_metric("triangle count", metrics.triangle_count, minimum=12)
    ctx.attach_artifact(
        "qa/previews/body_section.png",
        name="body section",
    )
    return ctx.report()
"""
    previews = """from main import object_model
from mini_articraft.sdk import SectionView, render_view


render_view(
    object_model,
    SectionView(plane_normal=(0.0, 1.0, 0.0)),
    "qa/previews/body_section.png",
)
"""

    def inspect_working_preview(query: ModelQuery) -> Response:
        assert not compile_signals_shown(query.tool_outputs())
        return calls(
            tool_call(
                "view_image",
                {"path": "qa/previews/body_section.png"},
            )
        )

    artifacts = run_scenario(
        "a checked box",
        [
            calls(tool_call("write", {"path": "geometry_helpers.py", "content": helper})),
            write_main(main),
            calls(tool_call("write", {"path": "previews.py", "content": previews})),
            calls(
                tool_call(
                    "exec_command",
                    {"command": '"$MINI_ARTICRAFT_PYTHON" previews.py'},
                )
            ),
            inspect_working_preview,
            compile_workspace(),
            text("done"),
        ],
        env=env,
        max_turns=7,
    )

    assert artifacts.record.status == "success"
    assert artifacts.workspace.joinpath("geometry_helpers.py").is_file()
    assert artifacts.workspace.joinpath("previews.py").is_file()
    assert len(artifacts.recorder.tool_finishes("view_image")) == 1
    tool_names = [event.name for event in artifacts.recorder.of(ToolStarted)]
    assert tool_names.index("exec_command") < tool_names.index("compile")
    assert tool_names.index("view_image") < tool_names.index("compile")
    compile_outputs = [
        event
        for event in artifacts.conversation
        if event.get("type") == "function_call_output"
        and "compile_signals" in str(event.get("output"))
    ]
    assert compile_outputs
    assert "workspace_path=qa/previews/body_section.png" in str(compile_outputs[-1]["output"])
    assert all("input_image" not in str(event["output"]) for event in compile_outputs)
    assert all("data:image" not in str(event["output"]) for event in compile_outputs)
    assert {
        path.relative_to(artifacts.workspace).as_posix()
        for path in artifacts.workspace.rglob("*.png")
    } == {"qa/previews/body_section.png"}
    assert not list(artifacts.run_dir.joinpath("result").rglob("*.png"))


def test_failure_streak_resets_after_a_successful_compile(tmp_path: Path) -> None:
    """A success clears the streak: the next failure is not flagged as repeated."""
    env = WarmEnvironment(output_dir=tmp_path)

    artifacts = run_scenario(
        "a box",
        [
            write_main(BROKEN_NO_RUN_TESTS),
            compile_workspace(),
            write_main(GOOD_MAIN_PY),
            compile_workspace(),
            write_main(BROKEN_NO_RUN_TESTS),
            compile_workspace(),
            write_main(GOOD_MAIN_PY),
            compile_workspace(),
            text("done"),
        ],
        env=env,
        max_turns=9,
    )

    assert artifacts.record.status == "success"
    # the final write restores the exact content of the earlier successful
    # compile, so the freshness cache serves it without a worker call
    assert env.compile_count == 3
    signals = compile_signals_shown(artifacts.tool_outputs())
    assert "matches the previous compile attempt" not in signals[0]
    assert "matches the previous compile attempt" not in signals[2]


def test_overlap_allowance_flows_through_the_real_worker(tmp_path: Path) -> None:
    env = WarmEnvironment(output_dir=tmp_path)

    def allow_the_overlap(query: ModelQuery) -> Response:
        signals = compile_signals_shown(query.tool_outputs())
        assert signals and "[real_overlap]" in signals[-1]
        return write_main(OVERLAP_ALLOWED_MAIN)

    artifacts = run_scenario(
        "a press-fit pin",
        [
            write_main(OVERLAP_MAIN),
            compile_workspace(),
            allow_the_overlap,
            compile_workspace(),
            text("done"),
        ],
        env=env,
        max_turns=5,
    )

    assert artifacts.record.status == "success"
    assert env.compile_count == 2
    codes = event_signal_codes(artifacts.recorder)
    assert "TEST_REAL_OVERLAP" in codes
    assert "NOTE_ALLOWED_OVERLAP" in codes
    final_signals = compile_signals_shown(artifacts.tool_outputs())[-1]
    assert "allowed by justification" in final_signals


def test_event_stream_is_ordered_and_complete(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [write_main(GOOD_MAIN_PY), compile_workspace(), text("done")],
        env=WarmEnvironment(output_dir=tmp_path),
    )

    stream = artifacts.recorder.events
    assert isinstance(stream[0], RunStarted)
    assert isinstance(stream[-1], RunFinished)
    assert len(artifacts.recorder.of(AssistantMessage)) == 3
    for name in ("write", "compile"):
        started = next(
            index
            for index, event in enumerate(stream)
            if isinstance(event, ToolStarted) and event.name == name
        )
        finished = artifacts.recorder.tool_finishes(name)[0]
        assert started < stream.index(finished)


def test_agent_hitting_max_turns_is_a_terminal_error(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [write_main(GOOD_MAIN_PY), compile_workspace()],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=2,
    )

    assert artifacts.record.status == "error"
    assert artifacts.record.error == "agent hit max turns limit"
    finished = artifacts.recorder.finished
    assert finished is not None
    assert finished.turns == 2


def test_script_exhaustion_surfaces_as_a_model_failure(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [text("no compile happened")],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=3,
    )

    assert artifacts.record.status == "error"
    assert "ScriptExhaustedError" in artifacts.record.error


def test_hand_authored_tape_drives_a_real_run(
    tmp_path: Path, replay_harness: ReplayHarness
) -> None:
    replay_harness.set(
        "authored-box",
        [write_main(GOOD_MAIN_PY), compile_workspace(), text("done from tape")],
    )

    artifacts = run_scenario(
        "a box",
        model=replay_harness.replay("authored-box"),
        env=WarmEnvironment(output_dir=tmp_path),
    )

    assert artifacts.record.status == "success"
    assert artifacts.result["message"] == "done from tape"
