from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from harness import GOOD_MAIN_PY, ScriptedModel, WarmEnvironment, calls, run, text, tool_call

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

    settings = api.resolved_settings(base, provider=provider, model=model)

    assert settings.provider == provider
    assert getattr(settings, field) == model
    assert settings.selected_model == model


def test_resolved_settings_applies_output_dir(tmp_path: Path) -> None:
    settings = api.resolved_settings(Settings(openai_api_key="sk-test"), output_dir=tmp_path)

    assert settings.output_dir == tmp_path


def test_resolved_settings_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unsupported provider: mistral"):
        api.resolved_settings(Settings(openai_api_key="sk-test"), provider="mistral")


def test_resolved_settings_rejects_unsupported_anthropic_model() -> None:
    with pytest.raises(ValueError, match="unsupported Anthropic model: claude-haiku-4-5"):
        api.resolved_settings(
            Settings(anthropic_api_key="sk-test"),
            provider="anthropic",
            model="claude-haiku-4-5",
        )


def test_resolved_settings_rejects_unsupported_gemini_model() -> None:
    with pytest.raises(ValueError, match=r"unsupported Gemini model: gemini-1\.5-flash"):
        api.resolved_settings(
            Settings(gemini_api_key="sk-test"),
            provider="gemini",
            model="gemini-1.5-flash",
        )


def test_missing_provider_settings_names_the_required_key() -> None:
    base = Settings(openai_api_key=None, gemini_api_key=None, anthropic_api_key=None)

    assert api.missing_provider_settings(base) == ["OPENAI_API_KEY"]
    assert api.missing_provider_settings(base.model_copy(update={"provider": "gemini"})) == [
        "GEMINI_API_KEY"
    ]
    assert api.missing_provider_settings(base.model_copy(update={"provider": "anthropic"})) == [
        "ANTHROPIC_API_KEY"
    ]


def test_start_requires_provider_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        mini_articraft.Generation("a box").start()


def test_start_rejects_missing_reference_image(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    generation = mini_articraft.Generation("a box", image=tmp_path / "missing.png")

    with pytest.raises(FileNotFoundError, match=r"missing\.png"):
        generation.start()


def test_generation_routes_settings_and_image_to_agent(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, settings: Settings):
            captured["settings"] = settings
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeEnvironment:
        def __init__(self, **kwargs: Any):
            captured["env_kwargs"] = kwargs

    class FakeAgent:
        def __init__(self, model: Any, env: Any, **kwargs: Any):
            captured["agent_kwargs"] = kwargs

        async def run(self, prompt: str, *, image_path: Path | None = None) -> dict[str, Any]:
            captured["prompt"] = prompt
            captured["image_path"] = image_path
            return {"status": "success", "run": "/tmp/run", "result": "result/model.usdz"}

    monkeypatch.setattr(api, "create_model", FakeModel)
    monkeypatch.setattr(api, "LocalEnvironment", FakeEnvironment)
    monkeypatch.setattr(api, "Agent", FakeAgent)
    monkeypatch.setattr(
        api, "get_settings", lambda: Settings(anthropic_api_key="sk-test", max_turns=7)
    )

    image = tmp_path / "reference.png"
    image.write_bytes(b"png")
    base = mini_articraft.Generation(
        "a fan",
        provider="anthropic",
        model="claude-opus-5",
        output_dir=tmp_path / "out",
    )
    generation = base.with_image(str(image))
    assert generation is not base
    assert base.image is None
    assert generation.image == str(image)

    result = generation.run()

    assert result["status"] == "success"
    assert captured["settings"].provider == "anthropic"
    assert captured["settings"].anthropic_model == "claude-opus-5"
    assert captured["settings"].output_dir == tmp_path / "out"
    assert captured["env_kwargs"] == {
        "output_dir": tmp_path / "out",
        "timeout_seconds": captured["settings"].compile_timeout_seconds,
        "physics_enabled": False,
    }
    assert captured["agent_kwargs"]["max_turns"] == 7
    assert captured["prompt"] == "a fan"
    assert captured["image_path"] == image


def test_generation_end_to_end_with_scripted_model(monkeypatch, tmp_path: Path) -> None:
    script = [
        calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
        calls(tool_call("compile")),
        text("done"),
    ]
    monkeypatch.setattr(api, "create_model", lambda settings: ScriptedModel(script))
    monkeypatch.setattr(api, "LocalEnvironment", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    generation_run = mini_articraft.Generation("a box", output_dir=tmp_path / "runs").start()
    seen = list(generation_run.watch())
    result = generation_run.wait()

    assert generation_run.done
    assert result["status"] == "success"
    run_dir = Path(str(result["run"]))
    assert run_dir.parent == tmp_path / "runs"
    usdz = run_dir / str(result["result"])
    assert usdz.is_file()
    assert usdz.suffix == ".usdz"
    assert isinstance(seen[0], events.RunStarted)
    assert isinstance(seen[-1], events.RunFinished)
    assert seen[-1].status == "success"


def test_watch_after_completion_still_sees_all_events(monkeypatch, tmp_path: Path) -> None:
    script = [
        calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
        calls(tool_call("compile")),
        text("done"),
    ]
    monkeypatch.setattr(api, "create_model", lambda settings: ScriptedModel(script))
    monkeypatch.setattr(api, "LocalEnvironment", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    generation_run = mini_articraft.Generation("a box", output_dir=tmp_path / "runs").start()
    result = generation_run.wait()

    seen = list(generation_run.watch())
    assert result["status"] == "success"
    assert isinstance(seen[0], events.RunStarted)
    assert isinstance(seen[-1], events.RunFinished)
    assert list(generation_run.watch()) == []


def test_generation_spec_is_immutable_and_reusable(monkeypatch, tmp_path: Path) -> None:
    script = [
        calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
        calls(tool_call("compile")),
        text("done"),
    ]
    monkeypatch.setattr(api, "create_model", lambda settings: ScriptedModel(script))
    monkeypatch.setattr(api, "LocalEnvironment", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    base = mini_articraft.Generation("a box")
    assert base.with_model("gpt-test") == base.with_model("gpt-test")
    assert base.with_model("gpt-test") != base

    first = base.with_output_dir(tmp_path / "a").start()
    second = base.with_output_dir(tmp_path / "b").start()
    results = [first.wait(), second.wait()]

    assert base.output_dir is None
    assert [result["status"] for result in results] == ["success", "success"]
    run_dirs = {Path(str(result["run"])).parent for result in results}
    assert run_dirs == {tmp_path / "a", tmp_path / "b"}


def test_wait_reraises_background_failure(monkeypatch, tmp_path: Path) -> None:
    def explode(settings: Settings) -> Any:
        raise RuntimeError("adapter exploded")

    monkeypatch.setattr(api, "create_model", explode)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    generation_run = mini_articraft.Generation("a box", output_dir=tmp_path / "runs").start()

    assert list(generation_run.watch()) == []
    with pytest.raises(RuntimeError, match="adapter exploded"):
        generation_run.wait()


def _hanging_model_step(query: Any) -> Any:
    async def hang() -> dict[str, Any]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return hang()


def test_cancel_stops_a_running_generation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api, "create_model", lambda settings: ScriptedModel([_hanging_model_step]))
    monkeypatch.setattr(api, "LocalEnvironment", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    generation_run = mini_articraft.Generation("a box", output_dir=tmp_path / "runs").start()
    watcher = generation_run.watch()
    assert isinstance(next(watcher), events.RunStarted)

    generation_run.cancel()
    list(watcher)
    with pytest.raises(mini_articraft.RunCancelledError):
        generation_run.wait(timeout=30)
    assert generation_run.done
    generation_run.cancel()


def test_run_context_manager_waits_for_completion(monkeypatch, tmp_path: Path) -> None:
    script = [
        calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
        calls(tool_call("compile")),
        text("done"),
    ]
    monkeypatch.setattr(api, "create_model", lambda settings: ScriptedModel(script))
    monkeypatch.setattr(api, "LocalEnvironment", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    generation = mini_articraft.Generation("a box", output_dir=tmp_path / "runs")
    with generation.start() as generation_run:
        seen = list(generation_run.watch())

    assert generation_run.done
    assert generation_run.wait()["status"] == "success"
    assert isinstance(seen[-1], events.RunFinished)


def test_run_context_manager_cancels_when_the_body_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api, "create_model", lambda settings: ScriptedModel([_hanging_model_step]))
    monkeypatch.setattr(api, "LocalEnvironment", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    generation = mini_articraft.Generation("a box", output_dir=tmp_path / "runs")
    with pytest.raises(RuntimeError, match="boom"), generation.start() as generation_run:
        assert isinstance(next(generation_run.watch()), events.RunStarted)
        raise RuntimeError("boom")

    assert generation_run.done
    with pytest.raises(mini_articraft.RunCancelledError):
        generation_run.wait()


def test_run_async_runs_on_the_ambient_loop(monkeypatch, tmp_path: Path) -> None:
    script = [
        calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
        calls(tool_call("compile")),
        text("done"),
    ]
    monkeypatch.setattr(api, "create_model", lambda settings: ScriptedModel(script))
    monkeypatch.setattr(api, "LocalEnvironment", WarmEnvironment)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    seen: list[events.Event] = []
    generation = mini_articraft.Generation("a box", output_dir=tmp_path / "runs")
    result = run(generation.run_async(on_event=seen.append))

    assert result["status"] == "success"
    assert isinstance(seen[-1], events.RunFinished)


def test_root_import_is_lazy_and_exports_generation() -> None:
    code = "\n".join(
        [
            "import sys",
            "import mini_articraft",
            "heavy = [name for name in (",
            "    'mini_articraft.api', 'mini_articraft.models', 'mini_articraft.agent',",
            "    'PIL', 'anthropic', 'websockets',",
            ") if name in sys.modules]",
            "assert not heavy, heavy",
            "assert callable(mini_articraft.Generation)",
            "assert callable(mini_articraft.Run)",
            "assert issubclass(mini_articraft.RunCancelledError, RuntimeError)",
            "assert 'mini_articraft.api' in sys.modules",
        ]
    )

    subprocess.run([sys.executable, "-c", code], check=True)
