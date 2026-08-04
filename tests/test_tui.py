from __future__ import annotations

import io
import json

from rich.console import Console

from mini_articraft.agent import events
from mini_articraft.tui import PRIMARY_STYLE, RunRenderer


def _renderer() -> tuple[RunRenderer, Console]:
    console = Console(file=io.StringIO(), width=120)
    return RunRenderer(console), console


def _text(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def test_renderer_primary_status_style_is_white() -> None:
    renderer, _console = _renderer()

    assert str(renderer.status_text().style) == PRIMARY_STYLE


def test_renderer_streams_transcript() -> None:
    renderer, console = _renderer()
    sequence = [
        events.RunStarted("run-x", "gpt-5.5", "a box", "high"),
        events.TurnStarted(1),
        events.AssistantMessage(1, "I'll write the file.", [{"id": "c1", "name": "write"}]),
        events.ToolStarted("c1", "write", '{"path": "main.py"}'),
        events.ToolFinished("c1", "write", {"result": {"path": "main.py", "bytes": 412}}, 0.01),
    ]
    for event in sequence:
        renderer.handle(event)

    out = _text(console)
    assert "run-x" in out
    assert "gpt-5.5" in out
    assert "reasoning high" in out
    assert "I'll write the file." in out
    assert "write(main.py)" in out
    assert "main.py (412 bytes)" in out


def test_renderer_shows_compile_and_errors() -> None:
    renderer, console = _renderer()
    renderer.handle(
        events.ToolFinished(
            "c1",
            "compile",
            {"result": {"status": "error", "error": "NameError: foo", "stderr": "Traceback\nboom"}},
            0.1,
        )
    )
    renderer.handle(events.ToolFinished("c2", "read", {"error": "file not found"}, 0.0))

    out = _text(console)
    assert "compile error" in out
    assert "NameError: foo" in out
    assert "Traceback" in out
    assert "boom" in out
    assert "read error: file not found" in out


def test_renderer_shows_compile_duration() -> None:
    renderer, console = _renderer()
    renderer.handle(
        events.ToolFinished(
            "c1",
            "compile",
            {"result": {"status": "success"}},
            12.34,
        )
    )

    assert "compile ok (12.3s)" in _text(console)


def test_renderer_shows_full_compile_traceback_without_truncation() -> None:
    renderer, console = _renderer()
    traceback_lines = ["Traceback (most recent call last):"]
    traceback_lines.extend(f'  File "main.py", line {line}' for line in range(1, 30))
    traceback_lines.append("ValueError: " + "bad loft " * 40)
    traceback_text = "\n".join(traceback_lines)

    renderer.handle(
        events.ToolFinished(
            "c1",
            "compile",
            {
                "result": {
                    "status": "error",
                    "error": "ValueError: bad loft",
                    "traceback": traceback_text,
                    "stderr": "compiler warning on stderr",
                }
            },
            0.1,
        )
    )

    out = _text(console)
    assert "compile error" in out
    assert 'File "main.py", line 29' in out
    assert ("bad loft " * 40).strip() in " ".join(out.split())
    assert "compiler warning on stderr" in out
    assert "…" not in out


def test_renderer_shows_token_usage_bar() -> None:
    renderer, console = _renderer()

    renderer.handle(events.RunStarted("run-x", "gpt-5.4-mini", "a box", "high", 400_000))
    renderer.handle(events.TurnStarted(1))
    renderer.handle(
        events.AssistantMessage(
            1,
            "done",
            [],
            {
                "input_tokens": 1_000,
                "cached_input_tokens": 100,
                "output_tokens": 250,
                "total_tokens": 1_250,
            },
        )
    )

    out = _text(console)
    assert "tokens 1.2k" in out
    assert "/ 400k" in out
    assert "(0.3%)" in out


def test_renderer_uses_gpt_5_6_sol_context_window() -> None:
    renderer, console = _renderer()

    renderer.handle(events.RunStarted("run-x", "gpt-5.6-sol", "a box", "low"))
    renderer.handle(events.TurnStarted(1))
    renderer.handle(
        events.AssistantMessage(
            1,
            "done",
            [],
            {
                "input_tokens": 10_000,
                "cached_input_tokens": 1_000,
                "output_tokens": 500,
                "total_tokens": 10_500,
            },
        )
    )

    out = _text(console)
    assert "tokens 10.5k" in out
    assert "/ 272k" in out
    assert "(3.9%)" in out


def test_renderer_shows_full_compile_signals_without_protocol_tags_or_truncation() -> None:
    renderer, console = _renderer()
    detail_lines = "\n".join(f"  detail line {index}" for index in range(20))
    signals_text = (
        "<compile_signals>\n"
        "<summary>\n"
        "status=failure failures=1 warnings=0 notes=0\n"
        "Primary issue: compiler-owned geometry checks found floating islands inside a part.\n"
        "</summary>\n"
        "\n"
        "<failures>\n"
        "Failures (blocking):\n"
        "- FAILURE [disconnected_geometry] Disconnected geometry islands were found inside a part.\n"
        f"{detail_lines}\n"
        "</failures>\n"
        "\n"
        "<response_rules>\n"
        "Suggested next steps:\n"
        "- Fix or explicitly justify the reported part relationship before adding more geometry.\n"
        "</response_rules>\n"
        "</compile_signals>"
    )

    renderer.handle(
        events.ToolFinished(
            "c1",
            "compile",
            {
                "result": {
                    "status": "error",
                    "error": "SDK tests failed",
                    "compile_report": {"signals_text": signals_text},
                }
            },
            0.1,
        )
    )

    out = _text(console)
    assert "compile signals" in out
    assert "status=failure failures=1 warnings=0 notes=0" in out
    assert "Failures (blocking):" in out
    assert "detail line 19" in out
    assert "Suggested next steps:" in out
    assert "<compile_signals>" not in out
    assert "… (truncated)" not in out


def test_renderer_truncates_long_output() -> None:
    renderer, console = _renderer()
    stdout = "\n".join(f"line {i}" for i in range(20))
    renderer.handle(
        events.ToolFinished(
            "c1",
            "exec_command",
            {
                "result": {
                    "returncode": 0,
                    "running": False,
                    "wall_time_seconds": 0.2,
                    "stdout": stdout,
                }
            },
            0.2,
        )
    )

    out = _text(console)
    assert "exec rc=0" in out
    assert "line 0" in out
    assert "… (truncated)" in out
    assert "line 19" not in out


def test_replay_shows_reference_image_metadata() -> None:
    renderer, console = _renderer()

    renderer.render_row(
        {
            "role": "user",
            "content": "reconstruct this",
            "images": [{"path": "input/reference.png", "width": 1200, "height": 800}],
        }
    )

    out = _text(console)
    assert "reconstruct this" in out
    assert "input/reference.png, 1200x800" in out


def test_final_summary_success() -> None:
    renderer, console = _renderer()
    summary = renderer.final_summary(
        events.RunFinished(
            status="success",
            run="runs/run-x",
            result="result/model.usdz",
            turns=3,
            duration=18.4,
            cost=0.012345,
            token_usage={"total_tokens": 1_250},
        )
    )
    console.print(summary)

    out = _text(console)
    assert "success" in out
    assert "runs/run-x" in out
    assert "3 turns" in out
    assert "cost $0.012345" in out
    assert "tokens 1.2k" in out


def test_final_summary_error_includes_message() -> None:
    renderer, console = _renderer()
    summary = renderer.final_summary(
        events.RunFinished(status="error", run="runs/run-x", error="compile failed", turns=5)
    )
    console.print(summary)

    out = _text(console)
    assert "error" in out
    assert "compile failed" in out


def test_replay_rows_resolve_tool_names() -> None:
    renderer, console = _renderer()
    rows = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "make a box"},
        {
            "role": "assistant",
            "content": "writing",
            "tool_calls": [
                {"id": "c1", "name": "write", "arguments": json.dumps({"path": "main.py"})}
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": json.dumps({"result": {"path": "main.py", "bytes": 120}}),
        },
        {"role": "compiler", "status": "success", "error": ""},
    ]
    for row in rows:
        renderer.render_row(row)

    out = _text(console)
    assert "make a box" in out
    assert "write(main.py)" in out
    assert "main.py (120 bytes)" in out
    assert "compile ok" in out
    assert renderer.turn == 1
