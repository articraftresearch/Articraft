from __future__ import annotations

import asyncio
import shlex
import sys
from datetime import datetime

import pytest
from harness import (
    GOOD_MAIN_PY,
    ModelQuery,
    Response,
    ScriptedModel,
    calls,
    compile_success_tool,
    run,
    stub_schema,
    text,
    tool_call,
)
from PIL import Image

import mini_articraft.agent.tools as agent_tools
from mini_articraft.agent import Agent, events
from mini_articraft.agent.harness import (
    PROMPT_SLUG_MAX_LENGTH,
    _prompt_slug,
    _read_sdk_quickstart,
    _run_id_for_prompt,
)
from mini_articraft.agent.tools import Tool, ToolContext, ToolResult
from mini_articraft.agent.tools._core import workspace_digest
from mini_articraft.environments.local import DEFAULT_MAIN_PY, LocalEnvironment
from mini_articraft.record import Record, read_conversation


def test_agent_writes_compiles_and_returns_final_response(tmp_path) -> None:
    model = ScriptedModel(
        [
            {
                "text": "",
                "cost": 0.25,
                "token_usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
                "tool_calls": [
                    tool_call(
                        "write",
                        {"path": "main.py", "content": GOOD_MAIN_PY},
                        call_id="call_1",
                    )
                ],
            },
            {
                "text": "",
                "cost": 0.25,
                "token_usage": {
                    "input_tokens": 200,
                    "cached_input_tokens": 20,
                    "output_tokens": 30,
                    "total_tokens": 230,
                },
                "tool_calls": [tool_call("compile", {}, call_id="call_2")],
            },
            {
                "text": "done",
                "cost": 0.5,
                "token_usage": {
                    "input_tokens": 300,
                    "cached_input_tokens": 30,
                    "output_tokens": 40,
                    "total_tokens": 340,
                },
                "tool_calls": [],
            },
        ]
    )
    env = LocalEnvironment(output_dir=tmp_path)
    agent = Agent(model, env, max_turns=3)

    result = run(agent.run("a box", run_id="box"))

    assert result["status"] == "success"
    assert result["cost"] == 1.0
    assert result["token_usage"] == {
        "input_tokens": 600,
        "cached_input_tokens": 60,
        "output_tokens": 90,
        "total_tokens": 690,
    }
    assert result["message"] == "done"
    assert result["compile_report"]["status"] == "success"
    assert tmp_path.joinpath("box", "workspace", "main.py").is_file()
    record = Record.load(tmp_path / "box" / "record.json")
    assert record.status == "success"
    assert record.result == "result/usdz/0000.usdz"
    assert record.cost == 1.0
    assert record.token_usage["total_tokens"] == 690
    first_query = model.queries[0]
    assert "<sdk_docs>" not in first_query.messages[0]["content"]
    assert [message["role"] for message in first_query.messages[:3]] == ["system", "user", "user"]
    assert first_query.messages[1]["content"].startswith("<sdk_quickstart>")
    assert "docs/sdk/common/30_articulated_object.md" in first_query.messages[1]["content"]
    assert first_query.messages[2]["content"].startswith("<task>")
    assert "a box" in first_query.messages[2]["content"]
    assert {tool["name"] for tool in first_query.tools} == {
        "read",
        "view_image",
        "edit",
        "write",
        "exec_command",
        "write_stdin",
        "compile",
    }


def test_agent_sends_typed_image_content_but_records_only_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    async def run_view_image(context, args):
        return ToolResult(
            {"path": args["path"], "mime_type": "image/png", "bytes": 5},
            [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aW1hZ2U=",
                    "detail": "high",
                }
            ],
        )

    async def run_write(context, args):
        context.workspace.joinpath(args["path"]).write_text(args["content"], encoding="utf-8")
        return {"path": args["path"], "bytes": len(args["content"])}

    def inspect_image(query: ModelQuery) -> Response:
        item = next(
            message for message in query.messages if message.get("type") == "function_call_output"
        )
        assert item["call_id"] == "call_image"
        assert item["output"][0]["type"] == "input_text"
        assert item["output"][1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,aW1hZ2U=",
            "detail": "high",
        }
        return calls(
            tool_call(
                "write",
                {"path": "main.py", "content": GOOD_MAIN_PY},
                call_id="call_write",
            )
        )

    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        {
            "view_image": Tool("view_image", stub_schema("view_image"), run_view_image),
            "write": Tool("write", stub_schema("write"), run_write),
            "compile": compile_success_tool(),
        },
    )
    model = ScriptedModel(
        [
            calls(
                tool_call(
                    "view_image",
                    {"path": "docs/sdk/build123d/assets/AngleIron.png"},
                    call_id="call_image",
                )
            ),
            inspect_image,
            calls(tool_call("compile", {}, call_id="call_compile")),
            text("done"),
        ]
    )
    agent = Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=4)

    result = run(agent.run("a box", run_id="box"))

    assert result["status"] == "success"
    image_event = next(
        event
        for event in read_conversation(tmp_path / "box" / "conversation.jsonl")
        if event.get("call_id") == "call_image"
    )
    assert isinstance(image_event["output"], str)
    assert "base64" not in image_event["output"]


def test_agent_sends_and_saves_initial_reference_image(tmp_path) -> None:
    image_path = tmp_path / "reference.png"
    Image.new("RGB", (12, 8), color="white").save(image_path)

    def inspect_image(query: ModelQuery) -> Response:
        content = query.messages[2]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "input_text"
        assert "a box" in content[0]["text"]
        assert content[1]["type"] == "input_image"
        assert content[1]["detail"] == "original"
        assert content[1]["image_url"].startswith("data:image/png;base64,")
        return calls(
            tool_call(
                "write",
                {"path": "main.py", "content": GOOD_MAIN_PY},
                call_id="call_write",
            )
        )

    model = ScriptedModel(
        [
            inspect_image,
            calls(tool_call("compile", {}, call_id="call_compile")),
            text("done"),
        ]
    )
    run_dir = tmp_path / "box"
    result = run(
        Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=3).run(
            "a box",
            run_id="box",
            image_path=image_path,
        )
    )

    assert result["status"] == "success"
    saved_image = run_dir / "input" / "reference.png"
    assert saved_image.read_bytes() == image_path.read_bytes()
    conversation_text = run_dir.joinpath("conversation.jsonl").read_text(encoding="utf-8")
    assert "base64" not in conversation_text
    image_event = next(
        event for event in read_conversation(run_dir / "conversation.jsonl") if event.get("images")
    )
    metadata = image_event["images"][0]
    assert {key: value for key, value in metadata.items() if key != "sha256"} == {
        "path": "input/reference.png",
        "mime_type": "image/png",
        "bytes": len(image_path.read_bytes()),
        "width": 12,
        "height": 8,
        "detail": "original",
    }
    assert len(metadata["sha256"]) == 64


def test_agent_requires_compile_before_final_response(tmp_path) -> None:
    model = ScriptedModel(
        [
            calls(
                tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY}, call_id="call_1")
            ),
            text("done too early"),
            calls(tool_call("compile", {}, call_id="call_2")),
            text("done"),
        ]
    )
    env = LocalEnvironment(output_dir=tmp_path)
    agent = Agent(model, env, max_turns=4)

    result = run(agent.run("a box", run_id="box"))

    assert result["status"] == "success"
    assert result["message"] == "done"
    assert model.queries[2].contains(
        "<compile_required>", "No successful compile has completed yet."
    )
    conversation = read_conversation(tmp_path / "box" / "conversation.jsonl")
    assert any(
        "<compile_required>" in str(event.get("content"))
        and "No successful compile has completed yet." in str(event.get("content"))
        for event in conversation
    )


def test_agent_keeps_compile_fresh_after_read(
    monkeypatch,
    tmp_path,
) -> None:
    async def run_write(context, args):
        context.workspace.joinpath(args["path"]).write_text(args["content"], encoding="utf-8")
        return {"path": args["path"], "bytes": len(args["content"])}

    async def run_read(context, args):
        return {"path": args["path"], "text": "L1: ok"}

    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        {
            "write": Tool("write", stub_schema("write"), run_write),
            "read": Tool("read", stub_schema("read"), run_read, supports_parallel=True),
            "compile": compile_success_tool(),
        },
    )
    model = ScriptedModel(
        [
            calls(tool_call("write", {"path": "main.py", "content": "x"}, call_id="call_1")),
            calls(tool_call("compile", {}, call_id="call_2")),
            calls(tool_call("read", {"path": "main.py"}, call_id="call_3")),
            text("done after read"),
        ]
    )
    env = LocalEnvironment(output_dir=tmp_path)
    agent = Agent(model, env, max_turns=4)

    result = run(agent.run("a box", run_id="box"))

    assert result["status"] == "success"
    assert result["message"] == "done after read"
    assert len(model.queries) == 4


def test_agent_keeps_compile_fresh_after_inspection_and_noop_edits(
    monkeypatch,
    tmp_path,
) -> None:
    selected = {name: agent_tools.get(name) for name in ("write", "edit", "exec_command")}
    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        selected | {"compile": compile_success_tool()},
    )
    model = ScriptedModel(
        [
            calls(tool_call("write", {"path": "main.py", "content": "x"}, call_id="call_1")),
            calls(tool_call("compile", {}, call_id="call_2")),
            calls(
                tool_call(
                    "edit",
                    {"path": "main.py", "old_text": "x", "new_text": "x"},
                    call_id="call_3",
                )
            ),
            calls(
                tool_call(
                    "edit",
                    {"path": "main.py", "old_text": "missing", "new_text": "y"},
                    call_id="call_4",
                )
            ),
            calls(tool_call("exec_command", {"command": "printf inspected"}, call_id="call_5")),
            text("done"),
        ]
    )

    result = run(Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=6).run("box"))

    assert result["status"] == "success"
    assert result["message"] == "done"


def test_agent_requires_new_compile_after_real_file_change(monkeypatch, tmp_path) -> None:
    write_tool = agent_tools.get("write")
    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        {"write": write_tool, "compile": compile_success_tool()},
    )
    model = ScriptedModel(
        [
            calls(tool_call("write", {"path": "main.py", "content": "x"}, call_id="call_1")),
            calls(tool_call("compile", {}, call_id="call_2")),
            calls(tool_call("write", {"path": "main.py", "content": "y"}, call_id="call_3")),
            text("too early"),
            calls(tool_call("compile", {}, call_id="call_4")),
            text("done"),
        ]
    )

    result = run(Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=6).run("box"))

    assert result["status"] == "success"
    assert result["message"] == "done"
    assert model.queries[4].contains(
        "<compile_required>", "changed since the last successful compile"
    )


def test_running_exec_session_blocks_compile_and_finalization(monkeypatch, tmp_path) -> None:
    selected = {name: agent_tools.get(name) for name in ("exec_command", "write_stdin")}
    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        selected | {"compile": compile_success_tool()},
    )

    def send_stdin(query: ModelQuery) -> Response:
        session_id = next(
            output["result"]["session_id"]
            for output in query.tool_outputs()
            if output.get("result", {}).get("session_id")
        )
        return calls(
            tool_call(
                "write_stdin",
                {"session_id": session_id, "chars": "go\n", "yield_time_ms": 1000},
                call_id="call_3",
            )
        )

    model = ScriptedModel(
        [
            calls(
                tool_call(
                    "exec_command",
                    {
                        "command": "read value; printf changed > main.py",
                        "yield_time_ms": 10,
                    },
                    call_id="call_1",
                )
            ),
            calls(tool_call("compile", {}, call_id="call_2")),
            text("too early"),
            send_stdin,
            calls(tool_call("compile", {}, call_id="call_4")),
            text("done"),
        ]
    )

    result = run(
        Agent(
            model,
            LocalEnvironment(output_dir=tmp_path),
            max_turns=6,
        ).run("box", run_id="box")
    )

    assert result["status"] == "success"
    assert result["message"] == "done"
    conversation = read_conversation(tmp_path / "box" / "conversation.jsonl")
    assert any(
        "finish the running exec_command" in str(event.get("output")) for event in conversation
    )
    assert any("is still running" in str(event.get("content")) for event in conversation)


def test_agent_requires_visible_final_after_fresh_compile(monkeypatch, tmp_path) -> None:
    write_tool = agent_tools.get("write")
    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        {"write": write_tool, "compile": compile_success_tool()},
    )
    model = ScriptedModel(
        [
            calls(tool_call("write", {"path": "main.py", "content": "x"}, call_id="call_1")),
            calls(tool_call("compile", {}, call_id="call_2")),
            text(""),
            text("done"),
        ]
    )

    result = run(Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=4).run("box"))

    assert result["message"] == "done"
    assert model.queries[3].contains("<final_response_required>")


def test_agent_does_not_finalize_success_without_a_usdz_result(monkeypatch, tmp_path) -> None:
    async def run_compile(context, args):
        result = {"status": "success"}
        context.compile_result = result
        context.successful_compile_result = result
        context.successful_compile_digest = workspace_digest(context.workspace)
        return result

    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        {"compile": Tool("compile", stub_schema("compile"), run_compile)},
    )
    model = ScriptedModel(
        [
            calls(tool_call("compile", {}, call_id="call_1")),
            text("done"),
        ]
    )

    result = run(Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=2).run("box"))

    assert result["status"] == "error"
    assert result["result"] == ""
    assert result["error"] == "fresh compile did not produce a USDZ result"


def test_cached_success_survives_a_later_failed_compile(monkeypatch, tmp_path) -> None:
    attempts = 0

    async def run_compile(context, args):
        nonlocal attempts
        if context.refresh_compile_freshness():
            return context.successful_compile_result
        attempts += 1
        if attempts == 1:
            usdz = context.run_dir / "result" / "usdz" / "0000.usdz"
            usdz.parent.mkdir(parents=True, exist_ok=True)
            usdz.write_bytes(b"test-usdz")
            result = {"status": "success", "usdz": str(usdz)}
            context.compile_result = result
            context.successful_compile_result = result
            context.successful_compile_digest = workspace_digest(context.workspace)
            return result
        result = {"status": "error", "error": "bad geometry"}
        context.compile_result = result
        return result

    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        {
            "write": agent_tools.get("write"),
            "compile": Tool("compile", stub_schema("compile"), run_compile),
        },
    )
    model = ScriptedModel(
        [
            calls(tool_call("compile", {}, call_id="compile_1")),
            calls(tool_call("write", {"path": "main.py", "content": "bad"}, call_id="change")),
            calls(tool_call("compile", {}, call_id="compile_2")),
            calls(
                tool_call(
                    "write",
                    {"path": "main.py", "content": DEFAULT_MAIN_PY},
                    call_id="restore",
                )
            ),
            calls(tool_call("compile", {}, call_id="compile_3")),
            text("done"),
        ]
    )

    result = run(
        Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=6).run("box", run_id="cached")
    )

    assert attempts == 2
    assert result["status"] == "success"
    assert result["result"] == "result/usdz/0000.usdz"


def test_agent_cancellation_terminates_a_live_exec_session(monkeypatch, tmp_path) -> None:
    command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)'"
    waiting = asyncio.Event()
    contexts: list[ToolContext] = []

    real_context = ToolContext

    def spy_context(*args, **kwargs):
        context = real_context(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr("mini_articraft.agent.harness.ToolContext", spy_context)

    async def block_forever(query: ModelQuery) -> Response:
        waiting.set()
        pending: asyncio.Future[None] = asyncio.Future()
        await pending
        raise AssertionError("blocking query unexpectedly completed")

    model = ScriptedModel(
        [
            calls(
                tool_call(
                    "exec_command",
                    {"command": command, "yield_time_ms": 10},
                    call_id="exec",
                )
            ),
            block_forever,
        ]
    )

    async def exercise() -> None:
        env = LocalEnvironment(output_dir=tmp_path)
        task = asyncio.create_task(Agent(model, env, max_turns=3).run("box", run_id="cancel"))
        await asyncio.wait_for(waiting.wait(), timeout=5)
        assert contexts[0].exec_sessions.live_ids()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert contexts[0].exec_sessions.live_ids() == ()

    run(exercise())


def test_agent_fails_third_consecutive_empty_response(tmp_path) -> None:
    model = ScriptedModel([text("") for _ in range(3)])

    result = run(Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=3).run("box"))

    assert result["status"] == "error"
    assert "three consecutive responses" in result["error"]
    assert model.queries[2].contains("Do not continue with reasoning-only output.")


def test_agent_records_model_query_failure(tmp_path) -> None:
    def fail(query: ModelQuery) -> Response:
        raise RuntimeError("socket closed")

    model = ScriptedModel([fail])

    result = run(Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=3).run("box"))

    assert result["status"] == "error"
    assert result["result"] == ""
    assert result["error"] == "model query failed: RuntimeError: socket closed"


def test_agent_closes_the_model_after_a_successful_run(tmp_path) -> None:
    model = ScriptedModel(
        [
            calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
            calls(tool_call("compile")),
            text("done"),
        ]
    )

    result = run(Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=3).run("box"))

    assert result["status"] == "success"
    assert model.close_calls == 1


def test_agent_closes_the_model_after_a_model_failure(tmp_path) -> None:
    def fail(query: ModelQuery) -> Response:
        raise RuntimeError("socket closed")

    model = ScriptedModel([fail])

    run(Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=3).run("box"))

    assert model.close_calls == 1


def test_agent_closes_the_model_on_cancellation(tmp_path) -> None:
    waiting = asyncio.Event()

    async def block_forever(query: ModelQuery) -> Response:
        waiting.set()
        pending: asyncio.Future[None] = asyncio.Future()
        await pending
        raise AssertionError("blocking query unexpectedly completed")

    model = ScriptedModel([block_forever])

    async def exercise() -> None:
        env = LocalEnvironment(output_dir=tmp_path)
        task = asyncio.create_task(Agent(model, env, max_turns=3).run("box", run_id="cancel"))
        await asyncio.wait_for(waiting.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(exercise())
    assert model.close_calls == 1


def test_agent_emits_run_events(tmp_path) -> None:
    model = ScriptedModel(
        [
            calls(
                tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY}, call_id="call_1")
            ),
            calls(tool_call("compile", {}, call_id="call_2")),
            text("done"),
        ]
    )
    env = LocalEnvironment(output_dir=tmp_path)
    captured: list[events.Event] = []
    agent = Agent(model, env, max_turns=3, on_event=captured.append)

    result = run(agent.run("a box", run_id="box"))

    assert result["status"] == "success"
    assert isinstance(captured[0], events.RunStarted)
    assert captured[0].model == "gpt-test"
    assert captured[0].reasoning_effort == "low"
    assert isinstance(captured[-1], events.RunFinished)
    kinds = {type(event) for event in captured}
    assert events.TurnStarted in kinds
    assert events.AssistantMessage in kinds
    assert events.ToolStarted in kinds
    assert events.ToolFinished in kinds

    write_finished = next(
        event
        for event in captured
        if isinstance(event, events.ToolFinished) and event.name == "write"
    )
    assert write_finished.payload["result"]["path"] == "main.py"
    assert write_finished.payload["result"]["bytes"] > 0

    compile_finished = next(
        event
        for event in captured
        if isinstance(event, events.ToolFinished) and event.name == "compile"
    )
    assert "compile_report" in compile_finished.payload["result"]
    assert "compile_signals" not in compile_finished.payload["result"]

    run_finished = captured[-1]
    assert isinstance(run_finished, events.RunFinished)
    assert run_finished.status == "success"
    assert run_finished.turns == 3


def test_agent_runs_parallel_safe_tools_concurrently(monkeypatch, tmp_path) -> None:
    active = 0
    max_active = 0

    async def run_read(context, args):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return {"path": args["path"]}

    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        {
            "read": Tool("read", stub_schema("read"), run_read, supports_parallel=True),
            "compile": compile_success_tool(),
        },
    )
    model = ScriptedModel(
        [
            calls(
                tool_call("read", {"path": "a.py"}, call_id="call_1"),
                tool_call("read", {"path": "b.py"}, call_id="call_2"),
            ),
            calls(tool_call("compile", {}, call_id="call_3")),
            text("done"),
        ]
    )
    env = LocalEnvironment(output_dir=tmp_path)
    agent = Agent(model, env, max_turns=3)

    result = run(agent.run("a box", run_id="box"))

    assert result["status"] == "success"
    assert max_active == 2
    outputs = [
        event
        for event in read_conversation(tmp_path / "box" / "conversation.jsonl")
        if event.get("type") == "function_call_output"
    ]
    assert [event["call_id"] for event in outputs[:2]] == ["call_1", "call_2"]


def test_agent_serializes_non_parallel_tools(monkeypatch, tmp_path) -> None:
    active = 0
    max_active = 0

    async def run_write(context, args):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return {"path": args["path"], "bytes": 1}

    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        {
            "write": Tool("write", stub_schema("write"), run_write, supports_parallel=True),
            "compile": compile_success_tool(),
        },
    )
    model = ScriptedModel(
        [
            calls(
                tool_call("write", {"path": "a.py"}, call_id="call_1"),
                tool_call("write", {"path": "b.py"}, call_id="call_2"),
            ),
            calls(tool_call("compile", {}, call_id="call_3")),
            text("done"),
        ]
    )
    env = LocalEnvironment(output_dir=tmp_path)
    agent = Agent(model, env, max_turns=3)

    result = run(agent.run("a box", run_id="box"))

    assert result["status"] == "success"
    assert max_active == 1


def test_sdk_quickstart_user_message_is_preloaded() -> None:
    content = _read_sdk_quickstart()

    assert content.startswith("<sdk_quickstart>")
    assert "This SDK quickstart is preloaded for the run." in content
    assert "plausible build123d and mesh approaches" in content
    assert "parallel reads" in content
    assert "Do not stop at the first workable API" in content
    assert "# SDK quickstart" in content
    assert "`docs/sdk/common/40_testing.md`" in content
    assert content.endswith("</sdk_quickstart>")


def test_default_run_id_uses_datetime_and_clipped_prompt() -> None:
    prompt = "Make a tiny articulated desk lamp with a compliant hinge and a wide base"

    run_id = _run_id_for_prompt(prompt, now=datetime(2026, 6, 30, 14, 5, 9))

    assert run_id == "20260630-140509-make-a-tiny-articulated-desk-lamp-with-a-complia"


def test_prompt_slug_sanitizes_and_falls_back() -> None:
    assert _prompt_slug("  Drawer: 2-stage / slide!  ") == "drawer-2-stage-slide"
    assert _prompt_slug("...") == "prompt"
    assert len(_prompt_slug("x" * 200)) == PROMPT_SLUG_MAX_LENGTH
