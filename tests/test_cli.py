from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from typer.testing import CliRunner

from mini_articraft.cli import mini
from mini_articraft.record import Record, append_conversation
from mini_articraft.settings import Settings, get_settings


class FakeOpenAIModel:
    instances: ClassVar[list[FakeOpenAIModel]] = []

    def __init__(self, settings: Settings):
        self.settings = settings
        self.closed = False
        self.instances.append(self)

    async def close(self) -> None:
        self.closed = True


class FakeEnvironment:
    instances: ClassVar[list[FakeEnvironment]] = []

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.instances.append(self)


class FakeAgent:
    instances: ClassVar[list[FakeAgent]] = []
    result: ClassVar[dict[str, object]] = {
        "status": "success",
        "run": "/tmp/run",
        "result": "result/model.usdz",
        "message": "done",
    }

    def __init__(self, model: FakeOpenAIModel, env: FakeEnvironment, **kwargs: Any):
        self.model = model
        self.env = env
        self.kwargs = kwargs
        self.prompt = ""
        self.instances.append(self)

    async def run(self, prompt: str) -> dict[str, object]:
        self.prompt = prompt
        return self.result


def reset_fakes() -> None:
    FakeOpenAIModel.instances = []
    FakeEnvironment.instances = []
    FakeAgent.instances = []
    FakeAgent.result = {
        "status": "success",
        "run": "/tmp/run",
        "result": "result/model.usdz",
        "message": "done",
    }


def test_cli_runs_agent_with_only_core_overrides(monkeypatch, tmp_path: Path) -> None:
    reset_fakes()
    monkeypatch.setattr(mini, "create_model", FakeOpenAIModel)
    monkeypatch.setattr(mini, "LocalEnvironment", FakeEnvironment)
    monkeypatch.setattr(mini, "Agent", FakeAgent)
    monkeypatch.setattr(
        mini, "get_settings", lambda: Settings(openai_api_key="sk-test", max_turns=123)
    )

    output_dir = tmp_path / "runs"
    result = CliRunner().invoke(
        mini.app,
        [
            "generate",
            "make a hinge",
            "--model",
            "gpt-test",
            "--output-dir",
            str(output_dir),
            "--reasoning-effort",
            "low",
            "--compile-timeout",
            "45",
        ],
    )

    assert result.exit_code == 0
    assert FakeOpenAIModel.instances[0].settings.openai_model == "gpt-test"
    assert FakeOpenAIModel.instances[0].settings.output_dir == output_dir
    assert FakeOpenAIModel.instances[0].settings.openai_reasoning_effort == "low"
    assert FakeOpenAIModel.instances[0].closed is True
    assert FakeOpenAIModel.instances[0].settings.compile_timeout_seconds == 45
    assert FakeEnvironment.instances[0].kwargs == {
        "output_dir": output_dir,
        "timeout_seconds": 45,
    }
    assert FakeAgent.instances[0].kwargs == {"max_turns": 123}
    assert FakeAgent.instances[0].prompt == "make a hinge"
    assert "status: success" in result.output
    assert "run: /tmp/run" in result.output


def test_cli_selects_gemini_provider(monkeypatch, tmp_path: Path) -> None:
    reset_fakes()
    monkeypatch.setattr(mini, "create_model", FakeOpenAIModel)
    monkeypatch.setattr(mini, "LocalEnvironment", FakeEnvironment)
    monkeypatch.setattr(mini, "Agent", FakeAgent)
    monkeypatch.setattr(
        mini,
        "get_settings",
        lambda: Settings(gemini_api_key="gemini-test", max_turns=123),
    )

    output_dir = tmp_path / "runs"
    result = CliRunner().invoke(
        mini.app,
        [
            "generate",
            "make a hinge",
            "--provider",
            "gemini",
            "--model",
            "gemini-3.1-pro-preview",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    settings = FakeOpenAIModel.instances[0].settings
    assert settings.provider == "gemini"
    assert settings.gemini_model == "gemini-3.1-pro-preview"
    assert settings.output_dir == output_dir
    assert settings.selected_model == "gemini-3.1-pro-preview"


def test_cli_warns_on_missing_required_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = CliRunner().invoke(mini.app, ["generate", "make a hinge", "--no-tui"])

    assert result.exit_code == 1
    assert "Missing required environment variable" in result.output
    assert "OPENAI_API_KEY" in result.output
    assert ".env.example" in result.output
    assert "setup needed" in result.output
    assert "Traceback" not in result.output
    assert "ValidationError" not in result.output


def test_cli_warns_on_missing_gemini_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = CliRunner().invoke(
        mini.app,
        ["generate", "make a hinge", "--provider", "gemini", "--no-tui"],
    )

    assert result.exit_code == 1
    assert "Missing required environment variable" in result.output
    assert "GEMINI_API_KEY" in result.output
    assert "Traceback" not in result.output


def test_cli_exits_nonzero_when_agent_fails(monkeypatch) -> None:
    reset_fakes()
    FakeAgent.result = {"status": "error", "run": "/tmp/run", "error": "compile failed"}
    monkeypatch.setattr(mini, "create_model", FakeOpenAIModel)
    monkeypatch.setattr(mini, "LocalEnvironment", FakeEnvironment)
    monkeypatch.setattr(mini, "Agent", FakeAgent)
    monkeypatch.setattr(mini, "get_settings", lambda: Settings(openai_api_key="sk-test"))

    result = CliRunner().invoke(mini.app, ["generate", "make a hinge"])

    assert result.exit_code == 1
    assert "error: compile failed" in result.output


def test_cli_replays_recorded_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-demo"
    run_dir.mkdir()
    conversation = run_dir / "conversation.jsonl"
    append_conversation(conversation, {"role": "user", "content": "make a box"})
    append_conversation(
        conversation,
        {
            "role": "assistant",
            "content": "writing the file",
            "tool_calls": [
                {"id": "c1", "name": "write", "arguments": json.dumps({"path": "main.py"})}
            ],
        },
    )
    append_conversation(
        conversation,
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": json.dumps({"result": {"path": "main.py", "bytes": 120}}),
        },
    )
    append_conversation(conversation, {"role": "compiler", "status": "success", "error": ""})
    Record(run_id="run-demo", status="success", result="result/model.usdz").save(
        run_dir / "record.json"
    )

    result = CliRunner().invoke(mini.app, ["replay", str(run_dir)])

    assert result.exit_code == 0
    assert "make a box" in result.output
    assert "write(main.py)" in result.output
    assert "compile ok" in result.output
    assert "success" in result.output


def test_cli_replay_missing_run_exits_nonzero(tmp_path: Path) -> None:
    result = CliRunner().invoke(mini.app, ["replay", str(tmp_path / "nope")])

    assert result.exit_code == 1
    assert "no conversation log" in result.output


def test_cli_view_opens_resolved_run(monkeypatch, tmp_path: Path) -> None:
    viewed: list[Path] = []

    def view_run(run_dir: Path) -> None:
        viewed.append(run_dir)

    monkeypatch.setattr(mini, "serve_viewer", view_run)

    result = CliRunner().invoke(
        mini.app,
        ["view", "run-demo", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert viewed == [tmp_path / "run-demo"]


def test_cli_view_reports_invalid_run(monkeypatch, tmp_path: Path) -> None:
    def fail(_run_dir: Path) -> None:
        raise ValueError("no USDZ outputs")

    monkeypatch.setattr(mini, "serve_viewer", fail)
    result = CliRunner().invoke(mini.app, ["view", str(tmp_path / "missing")])

    assert result.exit_code == 1
    assert "no USDZ outputs" in result.output


def test_main_args_accept_bare_prompt() -> None:
    assert mini._app_args(["articulated lamp"]) == ["generate", "articulated lamp"]
    assert mini._app_args(["articulated lamp", "--model", "gpt-test"]) == [
        "generate",
        "articulated lamp",
        "--model",
        "gpt-test",
    ]
    assert mini._app_args(["--model", "gpt-test", "articulated lamp"]) == [
        "generate",
        "--model",
        "gpt-test",
        "articulated lamp",
    ]


def test_main_args_keep_commands_and_help() -> None:
    assert mini._app_args(["generate", "articulated lamp"]) == ["generate", "articulated lamp"]
    assert mini._app_args(["replay", "run-x"]) == ["replay", "run-x"]
    assert mini._app_args(["view", "run-x"]) == ["view", "run-x"]
    assert mini._app_args(["--help"]) == ["--help"]
