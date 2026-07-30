"""Tests for the modular test environment itself (tests/harness.py)."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import _compile_server
import harness
import pytest
from harness import (
    GOOD_MAIN_PY,
    CompileServerError,
    ModelQuery,
    ReplayHarness,
    Response,
    ScriptedModel,
    ScriptExhaustedError,
    TapeError,
    TapeMismatchError,
    WarmEnvironment,
    calls,
    compile_success_tool,
    run,
    run_scenario,
    stub_schema,
    text,
    tool_call,
)

from mini_articraft.agent.events import TurnStarted
from mini_articraft.agent.tools import ToolContext
from mini_articraft.agent.workspace.local import LocalWorkspace
from mini_articraft.record import Record


def write_good_main() -> Response:
    return calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY}))


# ---------------------------------------------------------------------------
# ScriptedModel
# ---------------------------------------------------------------------------


def test_scripted_model_plays_steps_and_fills_defaults() -> None:
    model = ScriptedModel([text("hello"), calls(tool_call("read", {"path": "a.py"}))])

    first = run(model.query([{"role": "user", "content": "go"}]))
    assert first == {"text": "hello", "tool_calls": [], "cost": 0.0, "token_usage": {}}
    assert model.remaining == 1

    second = run(model.query([{"role": "user", "content": "go"}]))
    assert second["tool_calls"][0]["name"] == "read"
    assert model.remaining == 0
    assert [query.turn for query in model.queries] == [1, 2]
    assert model.queries[0].messages == [{"role": "user", "content": "go"}]
    model.assert_exhausted()


def test_scripted_model_exhaustion_raises_a_clear_error() -> None:
    model = ScriptedModel([text("only")])
    run(model.query([]))
    with pytest.raises(ScriptExhaustedError, match="turn 2 beyond the 1 scripted step"):
        run(model.query([]))


def test_assert_exhausted_catches_unplayed_steps() -> None:
    model = ScriptedModel([text("a"), text("b")])
    run(model.query([]))
    with pytest.raises(AssertionError, match="1 scripted step"):
        model.assert_exhausted()


def test_step_exceptions_propagate_as_model_failures() -> None:
    def fail(query: ModelQuery) -> Response:
        raise RuntimeError("socket closed")

    model = ScriptedModel([fail])
    with pytest.raises(RuntimeError, match="socket closed"):
        run(model.query([]))


def test_normalized_responses_do_not_mutate_step_dicts() -> None:
    step = {"text": "hi", "tool_calls": []}
    model = ScriptedModel([step])
    run(model.query([]))
    assert step == {"text": "hi", "tool_calls": []}


def test_model_query_contains_matches_a_single_message() -> None:
    query = ModelQuery(
        turn=1,
        messages=[
            {"role": "user", "content": "alpha only"},
            {"role": "user", "content": "alpha and beta"},
        ],
        tools=[],
    )
    assert query.contains("alpha")
    assert query.contains("alpha", "beta")  # both needles in one message
    assert not query.contains("beta", "gamma")
    # needles split across messages do not match: containment is per message
    split = ModelQuery(
        turn=1,
        messages=[{"role": "user", "content": "alpha"}, {"role": "user", "content": "beta"}],
        tools=[],
    )
    assert not split.contains("alpha", "beta")
    with pytest.raises(ValueError, match="at least one needle"):
        query.contains()


def test_model_query_tool_outputs_parses_only_tool_results() -> None:
    query = ModelQuery(
        turn=1,
        messages=[
            {"role": "assistant", "content": "working"},
            {"type": "function_call_output", "output": '{"result": {"path": "main.py"}}'},
        ],
        tools=[],
    )
    assert query.tool_outputs() == [{"result": {"path": "main.py"}}]


def test_tool_call_ids_are_unique_and_explicit_ids_win() -> None:
    first, second = tool_call("read"), tool_call("read")
    assert first["id"] != second["id"]
    assert tool_call("read", call_id="chosen")["id"] == "chosen"
    assert json.loads(tool_call("write", {"path": "a.py"})["arguments"]) == {"path": "a.py"}


def test_text_and_calls_carry_optional_fields() -> None:
    usage = {"input_tokens": 1, "output_tokens": 2}
    assert text("hi", cost=0.5, token_usage=usage)["cost"] == 0.5
    response = calls(tool_call("compile"), token_usage=usage)
    assert response["token_usage"] == usage
    assert response["tool_calls"][0]["name"] == "compile"


def test_non_dict_steps_are_a_clear_type_error() -> None:
    model = ScriptedModel([["not", "a", "dict"]])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="model responses must be dicts"):
        run(model.query([]))


def test_async_steps_are_awaited() -> None:
    async def slow(query: ModelQuery) -> Response:
        await asyncio.sleep(0)
        return text("late")

    model = ScriptedModel([slow])
    assert run(model.query([]))["text"] == "late"


def test_model_identity_is_configurable() -> None:
    model = ScriptedModel(
        [],
        model_name="gpt-custom",
        reasoning_effort="minimal",
        context_window_tokens=1234,
    )
    assert model.config.openai_model == "gpt-custom"
    assert model.config.openai_reasoning_effort == "minimal"
    assert model.context_window_tokens == 1234


# ---------------------------------------------------------------------------
# WarmEnvironment
# ---------------------------------------------------------------------------


def test_warm_environment_compiles_via_the_shared_worker(tmp_path: Path) -> None:
    env = WarmEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("box")
    (run_dir / "workspace" / "main.py").write_text(GOOD_MAIN_PY, encoding="utf-8")

    payload = env.compile_path(run_dir)

    assert payload["status"] == "success"
    assert payload["returncode"] == 0
    assert env.compile_count == 1
    assert Path(payload["usdz"]).is_file()
    assert payload["compile_report"]["status"] == "success"
    assert Record.load(run_dir / "record.json").attempts == 1


def test_warm_environment_surfaces_failures_with_signals(tmp_path: Path) -> None:
    env = WarmEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("broken")
    (run_dir / "workspace" / "main.py").write_text(
        "object_model = 'not a model'\n", encoding="utf-8"
    )

    payload = env.compile_path(run_dir)

    assert payload["status"] == "error"
    assert payload["returncode"] == 1
    report = payload["compile_report"]
    codes = {signal["code"] for signal in report["signal_bundle"]["signals"]}
    assert codes == {"COMPILE_RUNTIME_FAILURE"}
    assert "compile_runtime" in report["signals_text"]


def test_warm_environment_requires_main_py(tmp_path: Path) -> None:
    env = WarmEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("empty")
    (run_dir / "workspace" / "main.py").unlink()

    payload = env.compile_path(run_dir)

    assert payload["status"] == "error"
    assert "workspace/main.py is required" in payload["error"]
    assert env.compile_count == 0


def test_warm_environment_matches_the_cold_subprocess_contract(tmp_path: Path) -> None:
    outcomes = {}
    environments = {
        "cold": LocalWorkspace(output_dir=tmp_path / "cold"),
        "warm": WarmEnvironment(output_dir=tmp_path / "warm"),
    }
    for lane, env in environments.items():
        run_dir = env.create_run("box")
        (run_dir / "workspace" / "main.py").write_text(GOOD_MAIN_PY, encoding="utf-8")
        payload = env.compile_path(run_dir)
        outcomes[lane] = (
            payload["status"],
            payload["compile_report"]["status"],
            Path(payload["usdz"]).is_file(),
        )
    assert outcomes["cold"] == outcomes["warm"] == ("success", "success", True)


def test_cold_and_warm_lanes_fail_identically(tmp_path: Path) -> None:
    """Failure payloads also match across lanes: status, signals, and codes."""
    outcomes = {}
    environments = {
        "cold": LocalWorkspace(output_dir=tmp_path / "cold"),
        "warm": WarmEnvironment(output_dir=tmp_path / "warm"),
    }
    for lane, env in environments.items():
        run_dir = env.create_run("broken")
        (run_dir / "workspace" / "main.py").write_text(
            "object_model = 'not a model'\n", encoding="utf-8"
        )
        payload = env.compile_path(run_dir)
        bundle = payload["compile_report"]["signal_bundle"]
        outcomes[lane] = (
            payload["status"],
            payload["compile_report"]["status"],
            tuple(signal["code"] for signal in bundle["signals"]),
        )
    assert outcomes["cold"] == outcomes["warm"]
    assert outcomes["cold"][0] == "error"
    assert outcomes["cold"][2] == ("COMPILE_RUNTIME_FAILURE",)


HELPER_MAIN = """
import helper

from build123d import Box

from mini_articraft.sdk import ArticulatedObject, TestContext, TestReport

print(f"helper says {helper.VALUE}")


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("box")
    base = model.part("base")
    base.add(Box(0.2, 0.2, 0.1), name="body")
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    return TestContext(object_model).report()
"""


def _compile_helper_run(env: WarmEnvironment, run_id: str, value: str) -> dict:
    run_dir = env.create_run(run_id)
    workspace = run_dir / "workspace"
    workspace.joinpath("helper.py").write_text(f'VALUE = "{value}"\n', encoding="utf-8")
    workspace.joinpath("main.py").write_text(HELPER_MAIN, encoding="utf-8")
    return env.compile_path(run_dir)


def test_warm_environment_isolates_workspace_modules_between_runs(tmp_path: Path) -> None:
    """Two runs may both define helper.py; each compile must see its own."""
    env = WarmEnvironment(output_dir=tmp_path)

    first = _compile_helper_run(env, "first", "first-run")
    second = _compile_helper_run(env, "second", "second-run")

    assert first["status"] == second["status"] == "success"
    assert "helper says first-run" in first["stdout"]
    assert "helper says second-run" in second["stdout"]


def test_warm_environment_survives_workspace_code_that_exits(tmp_path: Path) -> None:
    """os._exit in workspace code kills the worker, not the test process."""
    env = WarmEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("selfish")
    (run_dir / "workspace" / "main.py").write_text("import os\nos._exit(1)\n", encoding="utf-8")

    payload = env.compile_path(run_dir)

    assert payload["status"] == "error"
    assert "exited mid-compile" in payload["error"]

    healthy = env.create_run("healthy")
    (healthy / "workspace" / "main.py").write_text(GOOD_MAIN_PY, encoding="utf-8")
    assert env.compile_path(healthy)["status"] == "success"  # a fresh worker took over


def test_compile_server_ignores_stale_generation_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Lines from a killed worker generation can never pose as the response."""
    env = WarmEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("box")
    (run_dir / "workspace" / "main.py").write_text(GOOD_MAIN_PY, encoding="utf-8")
    server = env._shared_server()
    monkeypatch.setattr(server, "_drain_stale_lines", lambda: None)
    server._lines.put((-1, '{"status": "stale"}'))
    server._lines.put((-1, None))

    status, payload = server.compile(run_dir, timeout_seconds=30)

    assert status == "ok"
    assert payload is not None
    assert payload["status"] == "success"


def test_compile_server_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A protocol violation is an infrastructure error, not a compile result."""
    env = WarmEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("box")
    (run_dir / "workspace" / "main.py").write_text(GOOD_MAIN_PY, encoding="utf-8")
    server = env._shared_server()
    monkeypatch.setattr(server, "_drain_stale_lines", lambda: None)
    status, _ = server.compile(run_dir, timeout_seconds=30)  # worker at a known generation
    assert status == "ok"
    server._lines.put((server._generation, "this is not json"))

    with pytest.raises(CompileServerError, match="invalid JSON"):
        server.compile(run_dir, timeout_seconds=30)


def test_compile_server_stop_is_idempotent() -> None:
    """Stopping with no worker is a no-op, as is stopping twice."""
    server = harness._CompileServer()
    server._stop()
    server._stop()


def test_compile_server_parses_requests() -> None:
    assert _compile_server._parse_request("not json") is None
    assert _compile_server._parse_request('{"wrong": "shape"}') is None
    assert _compile_server._parse_request('{"run_dir": 42}') is None
    parsed = _compile_server._parse_request('{"run_dir": "/tmp/some-run"}')
    assert parsed == Path("/tmp/some-run").resolve()


def test_compile_server_evicts_only_workspace_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "run" / "workspace"
    workspace.mkdir(parents=True)
    inside = workspace / "helper.py"
    inside.write_text("", encoding="utf-8")
    outside = tmp_path / "other.py"
    outside.write_text("", encoding="utf-8")
    for name, file in (
        ("fake_inside", inside),
        ("fake_outside", outside),
        ("fake_builtin", None),
    ):
        module = types.ModuleType(name)
        module.__file__ = None if file is None else str(file)
        monkeypatch.setitem(sys.modules, name, module)

    _compile_server._evict_workspace_modules(workspace)

    assert "fake_inside" not in sys.modules
    assert "fake_outside" in sys.modules
    assert "fake_builtin" in sys.modules


def test_warm_environment_enforces_the_timeout_contract(tmp_path: Path) -> None:
    env = WarmEnvironment(output_dir=tmp_path, timeout_seconds=0.5)
    run_dir = env.create_run("slow")
    (run_dir / "workspace" / "main.py").write_text(
        "import time\ntime.sleep(30)\n", encoding="utf-8"
    )

    payload = env.compile_path(run_dir)

    assert payload["status"] == "error"
    assert "timed out after 0.5s" in payload["error"]

    healthy = env.create_run("healthy")
    (healthy / "workspace" / "main.py").write_text(GOOD_MAIN_PY, encoding="utf-8")
    assert env.compile_path(healthy)["status"] == "success"  # a fresh worker took over


# ---------------------------------------------------------------------------
# ReplayHarness
# ---------------------------------------------------------------------------


def test_record_set_replay_erase_cycle(replay_harness: ReplayHarness) -> None:
    assert replay_harness.names() == []

    with replay_harness.record("box", ScriptedModel([text("hi")])) as recording:
        run(recording.query([{"role": "user", "content": "go"}]))
    assert replay_harness.names() == ["box"]
    assert replay_harness.entries("box")[0]["response"]["text"] == "hi"

    replay = replay_harness.replay("box")
    assert run(replay.query([{"role": "user", "content": "go"}]))["text"] == "hi"
    replay.assert_exhausted()

    assert replay_harness.erase("box") is True
    assert replay_harness.erase("box") is False
    assert not replay_harness.has("box")


def test_record_replaces_existing_and_rejects_empty_records(
    replay_harness: ReplayHarness,
) -> None:
    replay_harness.set("run", [text("v1")])
    with (
        pytest.raises(TapeError, match="recorded no exchanges"),
        replay_harness.record("run", ScriptedModel([text("v2")])),
    ):
        pass  # never queried -> a broken record
    assert not replay_harness.has("run")  # the old tape was replaced up front


def test_set_installs_hand_authored_rows(replay_harness: ReplayHarness) -> None:
    path = replay_harness.set("authored", [calls(tool_call("compile")), text("done", cost=0.5)])

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["fingerprint"] for row in rows] == ["", ""]
    assert rows[0]["response"]["tool_calls"][0]["name"] == "compile"
    assert rows[1]["response"]["cost"] == 0.5

    replay = replay_harness.replay("authored")  # strict, but fingerprintless rows match anything
    assert run(replay.query([{"role": "user", "content": "anything"}]))["tool_calls"]
    with pytest.raises(AssertionError, match="never consumed"):
        replay.assert_exhausted()


def test_set_preserves_full_rows_and_refuses_empty(replay_harness: ReplayHarness) -> None:
    replay_harness.set("original", [text("hi")])
    rows = replay_harness.entries("original")
    replay_harness.set("copy", rows)
    assert replay_harness.entries("copy") == rows
    with pytest.raises(TapeError, match="empty"):
        replay_harness.set("empty", [])


def test_library_rejects_unsafe_names(replay_harness: ReplayHarness) -> None:
    for bad in ("../up", "a/b", "", "with space"):
        with pytest.raises(ValueError, match="simple file name"):
            replay_harness.path(bad)


def test_clear_wipes_the_library(replay_harness: ReplayHarness) -> None:
    replay_harness.set("a", [text("a")])
    replay_harness.set("b", [text("b")])
    assert replay_harness.names() == ["a", "b"]
    assert replay_harness.clear() == 2
    assert replay_harness.names() == []


def test_strict_replay_detects_trajectory_drift(replay_harness: ReplayHarness) -> None:
    messages_1 = [{"role": "user", "content": "go"}]
    messages_2 = [*messages_1, {"role": "assistant", "content": "a"}]
    with replay_harness.record("run", ScriptedModel([text("a"), text("b")])) as recording:
        run(recording.query(messages_1))
        run(recording.query(messages_2))

    replay = replay_harness.replay("run")
    assert run(replay.query(messages_1))["text"] == "a"
    with pytest.raises(TapeMismatchError, match="turn 2 diverged"):
        run(replay.query([{"role": "user", "content": "different"}]))


def test_strict_replay_detects_tool_set_drift(replay_harness: ReplayHarness) -> None:
    recorded_tools = [{"name": "compile"}]
    with replay_harness.record("run", ScriptedModel([text("a"), text("b")])) as recording:
        run(recording.query([{"role": "user", "content": "go"}], tools=recorded_tools))
        run(recording.query([{"role": "user", "content": "go"}], tools=recorded_tools))

    replay = replay_harness.replay("run")
    first = run(replay.query([{"role": "user", "content": "go"}], tools=recorded_tools))
    assert first["text"] == "a"
    with pytest.raises(TapeMismatchError, match="turn 2 diverged"):
        run(replay.query([{"role": "user", "content": "go"}], tools=[{"name": "different"}]))


def test_strict_replay_ignores_payload_text(replay_harness: ReplayHarness) -> None:
    """Same structure, different words: payload text is not fingerprinted."""
    with replay_harness.record("run", ScriptedModel([text("a"), text("b")])) as recording:
        run(recording.query([{"role": "user", "content": "first wording"}]))
        run(
            recording.query(
                [
                    {"role": "user", "content": "first wording"},
                    {"role": "assistant", "content": "a"},
                ]
            )
        )

    replay = replay_harness.replay("run")
    assert run(replay.query([{"role": "user", "content": "DIFFERENT WORDING"}]))["text"] == "a"
    second = run(
        replay.query(
            [
                {"role": "user", "content": "DIFFERENT WORDING"},
                {"role": "assistant", "content": "a"},
            ]
        )
    )
    assert second["text"] == "b"


def test_non_strict_replay_skips_fingerprint_checks(replay_harness: ReplayHarness) -> None:
    with replay_harness.record("run", ScriptedModel([text("a")])) as recording:
        run(recording.query([{"role": "user", "content": "go"}]))

    replay = replay_harness.replay("run", strict=False)
    assert run(replay.query([{"role": "assistant", "content": "anything"}]))["text"] == "a"


def test_recording_model_delegates_exhaustion_to_finite_models(
    replay_harness: ReplayHarness,
) -> None:
    inner = ScriptedModel([text("a"), text("leftover")])
    with replay_harness.record("run", inner) as recording:
        run(recording.query([]))
    with pytest.raises(AssertionError, match="never consumed"):
        recording.assert_exhausted()


def test_a_failed_record_keeps_the_partial_tape(replay_harness: ReplayHarness) -> None:
    with (
        pytest.raises(RuntimeError, match="boom"),
        replay_harness.record("run", ScriptedModel([text("a")])) as recording,
    ):
        run(recording.query([]))
        raise RuntimeError("boom")
    assert replay_harness.entries("run")  # the partial tape stays for inspection

    with replay_harness.record("run", ScriptedModel([text("b")])) as recording:
        run(recording.query([]))
    assert [row["response"]["text"] for row in replay_harness.entries("run")] == ["b"]


def test_malformed_tape_rows_report_their_location(replay_harness: ReplayHarness) -> None:
    path = replay_harness.set("broken", [text("ok")])
    path.write_text(
        '{"fingerprint": "", "response": {"text": "ok"}}\nnot json\n',
        encoding="utf-8",
    )
    with pytest.raises(TapeError, match=r"invalid tape row .*:2"):
        replay_harness.entries("broken")


def test_set_normalizes_plain_responses(replay_harness: ReplayHarness) -> None:
    replay_harness.set("norm", [{"text": "hi"}])
    row = replay_harness.entries("norm")[0]
    assert row["response"] == {"text": "hi", "tool_calls": [], "cost": 0.0, "token_usage": {}}


def test_replay_beyond_the_tape_is_a_clear_error(replay_harness: ReplayHarness) -> None:
    replay_harness.set("short", [text("only")])
    replay = replay_harness.replay("short")
    run(replay.query([]))
    with pytest.raises(ScriptExhaustedError, match="beyond the 1 tape row"):
        run(replay.query([]))


def test_unknown_recording_is_a_clear_error(replay_harness: ReplayHarness) -> None:
    with pytest.raises(TapeError, match="unknown recording"):
        replay_harness.entries("missing")
    with pytest.raises(TapeError, match="unknown recording"):
        replay_harness.replay("missing")


def test_record_stores_metadata_and_replay_skips_it(replay_harness: ReplayHarness) -> None:
    with replay_harness.record(
        "run",
        ScriptedModel([text("a")]),
        meta={"prompt": "a box"},
    ) as recording:
        run(recording.query([]))

    assert replay_harness.meta("run") == {"prompt": "a box"}
    assert replay_harness.entries("run")[0]["meta"] == {"prompt": "a box"}
    replay = replay_harness.replay("run")
    assert run(replay.query([]))["text"] == "a"  # only exchange rows are replayed
    replay.assert_exhausted()


def test_set_preserves_metadata_rows_but_refuses_meta_only(
    replay_harness: ReplayHarness,
) -> None:
    replay_harness.set("with-meta", [{"meta": {"prompt": "x"}}, text("a")])
    rows = replay_harness.entries("with-meta")
    assert rows[0]["meta"] == {"prompt": "x"}
    assert rows[1]["response"]["text"] == "a"
    with pytest.raises(TapeError, match="empty"):
        replay_harness.set("meta-only", [{"meta": {"prompt": "x"}}])


# ---------------------------------------------------------------------------
# run_scenario
# ---------------------------------------------------------------------------


def test_run_scenario_returns_deep_artifacts(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [write_good_main(), calls(tool_call("compile")), text("done")],
        env=WarmEnvironment(output_dir=tmp_path),
    )

    assert artifacts.record.status == "success"
    assert artifacts.result["message"] == "done"
    assert (artifacts.workspace / "main.py").is_file()
    assert [event.turn for event in artifacts.recorder.of(TurnStarted)] == [1, 2, 3]
    assert len(artifacts.recorder.tool_finishes("compile")) == 1
    finished = artifacts.recorder.finished
    assert finished is not None
    assert finished.status == "success"
    assert artifacts.tool_outputs()[0]["result"]["path"] == "main.py"
    assert isinstance(artifacts.model, ScriptedModel)
    assert artifacts.model.queries[-1].turn == 3


def test_reactive_steps_can_assert_on_tool_outputs(tmp_path: Path) -> None:
    def check_write(query: ModelQuery) -> Response:
        outputs = query.tool_outputs()
        assert outputs[-1]["result"]["path"] == "main.py"
        return calls(tool_call("compile"))

    artifacts = run_scenario(
        "a box",
        [write_good_main(), check_write, text("done")],
        env=WarmEnvironment(output_dir=tmp_path),
    )
    assert artifacts.record.status == "success"


def test_run_scenario_fails_when_the_script_is_not_consumed(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="never consumed"):
        run_scenario(
            "a box",
            [text("done"), text("extra")],
            env=WarmEnvironment(output_dir=tmp_path),
            max_turns=1,
        )


def test_run_scenario_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="script= or model="):
        run_scenario("a box")
    with pytest.raises(ValueError, match="not both"):
        run_scenario("a box", [text("x")], model=ScriptedModel([text("y")]))
    with pytest.raises(ValueError, match="env= or tmp_path="):
        run_scenario("a box", [text("x")])


def test_run_scenario_allows_open_ended_scripts(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [write_good_main(), calls(tool_call("compile")), text("done"), text("unused")],
        env=WarmEnvironment(output_dir=tmp_path),
        assert_exhausted=False,
    )
    assert artifacts.record.status == "success"
    assert isinstance(artifacts.model, ScriptedModel)
    assert len(artifacts.model.queries) == 3


def test_run_scenario_honors_run_id(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [write_good_main(), calls(tool_call("compile")), text("done")],
        env=WarmEnvironment(output_dir=tmp_path),
        run_id="custom-run",
    )
    assert artifacts.run_dir.name == "custom-run"
    assert artifacts.record.run_id == "custom-run"


def test_replay_leftovers_fail_the_scenario(tmp_path: Path, replay_harness: ReplayHarness) -> None:
    replay_harness.set(
        "long",
        [write_good_main(), calls(tool_call("compile")), text("done"), text("unused")],
    )
    with pytest.raises(AssertionError, match="tape row"):
        run_scenario(
            "a box",
            model=replay_harness.replay("long"),
            env=WarmEnvironment(output_dir=tmp_path),
        )


def test_compile_success_tool_mints_a_usdz_and_marks_freshness(tmp_path: Path) -> None:
    env = LocalWorkspace(output_dir=tmp_path)
    run_dir = env.create_run("box")
    context = ToolContext(env, run_dir, run_dir / "workspace")

    result = run(compile_success_tool().run(context, {}))

    assert result["status"] == "success"
    assert Path(result["usdz"]).is_file()
    assert context.refresh_compile_freshness()


def test_stub_schema_has_the_function_wire_shape() -> None:
    schema = stub_schema("compile")
    assert schema["type"] == "function"
    assert schema["name"] == "compile"
    assert schema["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_recorded_tape_replays_a_full_scenario(
    tmp_path: Path, replay_harness: ReplayHarness
) -> None:
    script = [write_good_main(), calls(tool_call("compile")), text("done", cost=0.25)]
    with replay_harness.record("box", ScriptedModel(script)) as recording:
        recorded = run_scenario(
            "a box",
            model=recording,
            env=WarmEnvironment(output_dir=tmp_path / "a"),
        )
    assert recorded.record.status == "success"

    replayed = run_scenario(
        "a box",
        model=replay_harness.replay("box"),
        env=WarmEnvironment(output_dir=tmp_path / "b"),
    )
    assert replayed.record.status == "success"
    assert replayed.result["message"] == "done"
    assert replayed.result["cost"] == 0.25
