from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from harness import GOOD_MAIN_PY, ScriptedModel, WarmEnvironment, calls, run, text, tool_call
from pydantic import ValidationError

import mini_articraft
from mini_articraft import api
from mini_articraft.agent import events
from mini_articraft.agent.record import Record
from mini_articraft.compiler.result import CompilePayload
from mini_articraft.settings import Settings, get_settings


@pytest.mark.parametrize(
    ("provider", "model", "field"),
    [
        ("openai", "gpt-future-preview", "openai_model"),
        ("gemini", "gemini-future-preview", "gemini_model"),
        ("anthropic", "claude-future-preview", "anthropic_model"),
    ],
)
def test_resolved_settings_routes_model_to_selected_provider(
    provider: str, model: str, field: str
) -> None:
    base = Settings(openai_api_key="sk-test")

    settings = api._resolved_settings(base, provider=provider, model=model)

    assert settings.provider == provider
    assert getattr(settings, field) == model
    assert settings.selected_model == model


def test_resolved_settings_validates_overrides(tmp_path: Path) -> None:
    base = Settings(openai_api_key="sk-test")

    settings = api._resolved_settings(base, output_dir=tmp_path, compile_timeout=2.5)

    assert settings.output_dir == tmp_path
    assert settings.compile_timeout_seconds == 2.5
    with pytest.raises(ValidationError, match="greater than 0"):
        api._resolved_settings(base, compile_timeout=0)


def test_resolved_settings_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unsupported provider: mistral"):
        api._resolved_settings(Settings(openai_api_key="sk-test"), provider="mistral")


def test_missing_provider_settings_treats_whitespace_as_missing() -> None:
    base = Settings(openai_api_key=" ", gemini_api_key=" ", anthropic_api_key=" ")

    assert api._missing_provider_settings(base) == ["OPENAI_API_KEY"]
    assert api._missing_provider_settings(base.model_copy(update={"provider": "gemini"})) == [
        "GEMINI_API_KEY"
    ]
    assert api._missing_provider_settings(base.model_copy(update={"provider": "anthropic"})) == [
        "ANTHROPIC_API_KEY"
    ]


def test_generate_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt must not be empty"):
        mini_articraft.generate("  ")


def test_generate_requires_provider_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        mini_articraft.generate("a box")


def test_generate_rejects_missing_reference_image(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    with pytest.raises(FileNotFoundError, match=r"missing\.png"):
        mini_articraft.generate("a box", image=tmp_path / "missing.png")


def test_generate_routes_inputs_and_returns_typed_paths(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, settings: Settings):
            captured["settings"] = settings
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    class FakeEnvironment:
        def __init__(self, **kwargs: Any):
            captured["env_kwargs"] = kwargs

    class FakeAgent:
        def __init__(self, model: Any, env: Any, **kwargs: Any):
            captured["agent_kwargs"] = kwargs

        async def run(
            self,
            prompt: str,
            *,
            image_path: Path | None = None,
            source: Path | None = None,
        ) -> dict[str, Any]:
            captured["prompt"] = prompt
            captured["image_path"] = image_path
            captured["source"] = source
            return {
                "status": "success",
                "run_id": "test-run",
                "run": str(tmp_path / "runs" / "test-run"),
                "result": "result/model.usdz",
                "message": "done",
                "attempts": 2,
                "cost": 1.25,
                "token_usage": {"input_tokens": 10},
                "compile_report": {"shapes": 3},
            }

    monkeypatch.setattr(api, "create_model", FakeModel)
    monkeypatch.setattr(api, "LocalWorkspace", FakeEnvironment)
    monkeypatch.setattr(api, "Agent", FakeAgent)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(anthropic_api_key="sk-test"))

    image = tmp_path / "reference.png"
    image.write_bytes(b"png")
    seen: list[events.Event] = []

    result = mini_articraft.generate(
        "a fan",
        provider="anthropic",
        model="claude-opus-5",
        image=image,
        output_dir=tmp_path / "runs",
        on_event=seen.append,
    )

    assert isinstance(result, mini_articraft.GenerationResult)
    assert result.succeeded
    assert result.status == "success"
    assert result.run_id == "test-run"
    assert result.run_dir == tmp_path / "runs" / "test-run"
    assert result.artifact == result.run_dir / "result/model.usdz"
    assert result.message == "done"
    assert result.attempts == 2
    assert result.cost == 1.25
    assert result.token_usage == {"input_tokens": 10}
    assert result.compile_report == {"shapes": 3}
    assert captured["settings"].provider == "anthropic"
    assert captured["settings"].anthropic_model == "claude-opus-5"
    assert captured["settings"].output_dir == tmp_path / "runs"
    assert captured["env_kwargs"] == {
        "output_dir": tmp_path / "runs",
        "timeout_seconds": captured["settings"].compile_timeout_seconds,
        "physics_enabled": False,
    }
    assert captured["agent_kwargs"]["max_turns"] == 100
    assert captured["agent_kwargs"]["on_event"] == seen.append
    assert captured["prompt"] == "a fan"
    assert captured["image_path"] == image


def test_generate_end_to_end_with_scripted_model(monkeypatch, tmp_path: Path) -> None:
    script = [
        calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
        calls(tool_call("compile")),
        text("done"),
    ]
    model = ScriptedModel(script)
    monkeypatch.setattr(api, "create_model", lambda settings: model)
    monkeypatch.setattr(api, "LocalWorkspace", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))
    seen: list[events.Event] = []

    result = mini_articraft.generate(
        "a box",
        output_dir=tmp_path / "runs",
        on_event=seen.append,
    )

    assert result.succeeded
    assert result.run_dir.parent == tmp_path / "runs"
    assert result.artifact is not None
    assert result.artifact.is_file()
    assert result.artifact.suffix == ".usdz"
    assert isinstance(seen[0], events.RunStarted)
    assert isinstance(seen[-1], events.RunFinished)
    assert seen[-1].status == "success"
    assert model.close_calls == 1


def test_error_result_has_no_artifact() -> None:
    result = api._result_from_payload(
        {
            "status": "error",
            "run_id": "failed-run",
            "run": "runs/failed-run",
            "error": "agent hit max turns limit",
        }
    )

    assert not result.succeeded
    assert result.artifact is None
    assert result.error == "agent hit max turns limit"


def test_result_rejects_incomplete_internal_payload() -> None:
    with pytest.raises(ValueError, match="unexpected generation status"):
        api._result_from_payload({"run": "runs/test"})
    with pytest.raises(ValueError, match="missing its run directory"):
        api._result_from_payload({"status": "error"})
    with pytest.raises(ValueError, match="missing its run directory"):
        api._result_from_payload({"status": "error", "run": "."})
    with pytest.raises(ValueError, match="invalid run directory"):
        api._result_from_payload({"status": "error", "run": "/"})
    with pytest.raises(ValueError, match="missing its artifact"):
        api._result_from_payload({"status": "success", "run": "runs/test"})
    with pytest.raises(ValueError, match="file beneath"):
        api._result_from_payload({"status": "success", "run": "runs/test", "result": "."})
    with pytest.raises(ValueError, match="unexpectedly contains an artifact"):
        api._result_from_payload(
            {"status": "error", "run": "runs/test", "result": "result/model.usdz"}
        )
    with pytest.raises(ValueError, match="must stay inside"):
        api._result_from_payload(
            {"status": "success", "run": "runs/test", "result": "../outside.usdz"}
        )
    with pytest.raises(ValueError, match="run id does not match"):
        api._result_from_payload(
            {
                "status": "success",
                "run_id": "other",
                "run": "runs/test",
                "result": "result/model.usdz",
            }
        )


def test_result_paths_remain_stable_after_cwd_changes(monkeypatch, tmp_path: Path) -> None:
    original_cwd = tmp_path / "original"
    original_cwd.mkdir()
    monkeypatch.chdir(original_cwd)

    result = api._result_from_payload(
        {
            "status": "success",
            "run": "runs/test",
            "result": "result/model.usdz",
        }
    )
    monkeypatch.chdir(tmp_path)

    assert result.run_dir == original_cwd / "runs/test"
    assert result.artifact == original_cwd / "runs/test/result/model.usdz"

    absolute = api._result_from_payload(
        {
            "status": "success",
            "run": str(original_cwd / "runs/test"),
            "result": str(original_cwd / "runs/test/result/model.usdz"),
        }
    )
    assert absolute.artifact == original_cwd / "runs/test/result/model.usdz"

    whitespace_run = original_cwd / " run "
    whitespace = api._result_from_payload(
        {
            "status": "success",
            "run": str(whitespace_run),
            "result": "result/model.usdz",
        }
    )
    assert whitespace.run_dir == whitespace_run
    assert whitespace.artifact == whitespace_run / "result/model.usdz"


def test_generate_async_runs_on_the_ambient_loop(monkeypatch, tmp_path: Path) -> None:
    script = [
        calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
        calls(tool_call("compile")),
        text("done"),
    ]
    monkeypatch.setattr(api, "create_model", lambda settings: ScriptedModel(script))
    monkeypatch.setattr(api, "LocalWorkspace", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))
    seen: list[events.Event] = []

    result = run(
        mini_articraft.generate_async(
            "a box",
            output_dir=tmp_path / "runs",
            on_event=seen.append,
        )
    )

    assert result.succeeded
    assert isinstance(seen[-1], events.RunFinished)


def _hanging_model_step(query: Any) -> Any:
    async def hang() -> dict[str, Any]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return hang()


def test_generate_async_uses_native_task_cancellation(monkeypatch, tmp_path: Path) -> None:
    model = ScriptedModel([_hanging_model_step])
    monkeypatch.setattr(api, "create_model", lambda settings: model)
    monkeypatch.setattr(api, "LocalWorkspace", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    async def cancel_running_generation() -> None:
        started = asyncio.Event()

        def on_event(event: events.Event) -> None:
            if isinstance(event, events.RunStarted):
                started.set()

        task = asyncio.create_task(
            mini_articraft.generate_async(
                "a box",
                output_dir=tmp_path / "runs",
                on_event=on_event,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(cancel_running_generation())
    record_path = next((tmp_path / "runs").glob("*/record.json"))
    assert model.close_calls == 1
    assert Record.load(record_path).status == "error"
    assert Record.load(record_path).error == "generation cancelled"


@pytest.mark.parametrize("compile_fails", [False, True])
def test_generate_async_finishes_active_compile_before_cancelling(
    monkeypatch, tmp_path: Path, compile_fails: bool
) -> None:
    compile_started = threading.Event()
    release_compile = threading.Event()
    model = ScriptedModel(
        [
            calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
            calls(tool_call("compile")),
        ]
    )

    class BlockingEnvironment(WarmEnvironment):
        def compile_path(self, run_dir: Path | str) -> CompilePayload:
            compile_started.set()
            if not release_compile.wait(timeout=5):
                raise TimeoutError("test did not release compile")
            if compile_fails:
                raise RuntimeError("compile failed during cancellation")
            return super().compile_path(run_dir)

    monkeypatch.setattr(api, "create_model", lambda settings: model)
    monkeypatch.setattr(api, "LocalWorkspace", BlockingEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    async def cancel_during_compile() -> None:
        task = asyncio.create_task(
            mini_articraft.generate_async("a box", output_dir=tmp_path / "runs")
        )
        assert await asyncio.to_thread(compile_started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_compile.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        run(cancel_during_compile())
    finally:
        release_compile.set()

    record_path = next((tmp_path / "runs").glob("*/record.json"))
    assert model.close_calls == 1
    assert Record.load(record_path).error == "generation cancelled"


def test_generate_async_supports_concurrent_identical_prompts(monkeypatch, tmp_path: Path) -> None:
    script = [
        calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
        calls(tool_call("compile")),
        text("done"),
    ]
    monkeypatch.setattr(api, "create_model", lambda settings: ScriptedModel(list(script)))
    monkeypatch.setattr(api, "LocalWorkspace", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    async def generate_twice() -> list[api.GenerationResult]:
        return list(
            await asyncio.gather(
                mini_articraft.generate_async("a box", output_dir=tmp_path / "runs"),
                mini_articraft.generate_async("a box", output_dir=tmp_path / "runs"),
            )
        )

    first, second = run(generate_twice())

    assert first.succeeded and second.succeeded
    assert first.run_id != second.run_id
    assert first.run_dir != second.run_dir


def test_generate_rejects_active_event_loop() -> None:
    async def call_sync_api() -> None:
        with pytest.raises(RuntimeError, match=r"await generate_async\(\) instead"):
            mini_articraft.generate("a box")

    run(call_sync_api())


def test_root_import_is_lazy_and_exports_python_api() -> None:
    code = "\n".join(
        [
            "import sys",
            "import mini_articraft",
            "heavy = [name for name in (",
            "    'mini_articraft.api', 'mini_articraft.agent.provider', 'mini_articraft.agent',",
            "    'PIL', 'anthropic', 'websockets',",
            ") if name in sys.modules]",
            "assert not heavy, heavy",
            "public = {'Event', 'EventHandler', 'GenerationResult', 'GenerationStatus',",
            "          'Provider', 'generate', 'generate_async'}",
            "assert public <= set(dir(mini_articraft))",
            "assert 'mini_articraft.api' not in sys.modules",
            "assert callable(mini_articraft.generate)",
            "assert callable(mini_articraft.generate_async)",
            "assert callable(mini_articraft.GenerationResult)",
            "assert 'mini_articraft.api' in sys.modules",
            "assert mini_articraft.generate is mini_articraft.__dict__['generate']",
        ]
    )

    subprocess.run([sys.executable, "-c", code], check=True)
