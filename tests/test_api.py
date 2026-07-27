from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from harness import GOOD_MAIN_PY, ScriptedModel, WarmEnvironment, calls, run, text, tool_call
from pydantic import ValidationError

import mini_articraft
from mini_articraft import api
from mini_articraft.agent import events
from mini_articraft.settings import Settings, get_settings


@pytest.mark.parametrize(
    ("provider", "model", "field"),
    [
        ("openai", "gpt-test", "openai_model"),
        ("gemini", "gemini-3.6-flash", "gemini_model"),
        ("anthropic", "claude-opus-5", "anthropic_model"),
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


@pytest.mark.parametrize(
    ("provider", "model", "message"),
    [
        ("anthropic", "claude-haiku-4-5", "unsupported Anthropic model"),
        ("gemini", "gemini-1.5-flash", "unsupported Gemini model"),
    ],
)
def test_resolved_settings_rejects_unsupported_models(
    provider: str, model: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        api._resolved_settings(
            Settings(
                openai_api_key="sk-test",
                anthropic_api_key="sk-test",
                gemini_api_key="sk-test",
            ),
            provider=provider,
            model=model,
        )


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

        async def run(self, prompt: str, *, image_path: Path | None = None) -> dict[str, Any]:
            captured["prompt"] = prompt
            captured["image_path"] = image_path
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
    monkeypatch.setattr(api, "LocalEnvironment", FakeEnvironment)
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
    monkeypatch.setattr(api, "LocalEnvironment", WarmEnvironment)
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
    assert model.close_calls >= 1


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


def test_generate_async_runs_on_the_ambient_loop(monkeypatch, tmp_path: Path) -> None:
    script = [
        calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
        calls(tool_call("compile")),
        text("done"),
    ]
    monkeypatch.setattr(api, "create_model", lambda settings: ScriptedModel(script))
    monkeypatch.setattr(api, "LocalEnvironment", WarmEnvironment)
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
    monkeypatch.setattr(api, "LocalEnvironment", WarmEnvironment)
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
    assert model.close_calls >= 1


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
            "    'mini_articraft.api', 'mini_articraft.models', 'mini_articraft.agent',",
            "    'PIL', 'anthropic', 'websockets',",
            ") if name in sys.modules]",
            "assert not heavy, heavy",
            "assert callable(mini_articraft.generate)",
            "assert callable(mini_articraft.generate_async)",
            "assert callable(mini_articraft.GenerationResult)",
            "assert 'mini_articraft.api' in sys.modules",
        ]
    )

    subprocess.run([sys.executable, "-c", code], check=True)
