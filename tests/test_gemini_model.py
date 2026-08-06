from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import interactions
from pydantic import TypeAdapter

from mini_articraft.agent.provider.gemini import GeminiModel, context_window_tokens_for
from mini_articraft.errors import ModelError
from mini_articraft.settings import DEFAULT_GEMINI_MODEL, Settings, get_settings


def run(awaitable):
    return asyncio.get_event_loop().run_until_complete(awaitable)


class FakeInteractions:
    def __init__(self, responses: list[Any]):
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list[Any]):
        self.interactions = FakeInteractions(responses)
        self.aio = SimpleNamespace(interactions=self.interactions, aclose=self.aclose)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def text_response(
    text: str,
    *,
    input_tokens: int = 0,
    cached_tokens: int = 0,
    output_tokens: int = 0,
    thought_tokens: int = 0,
) -> Any:
    return SimpleNamespace(
        status="completed",
        output_text=text,
        steps=[
            {
                "type": "model_output",
                "content": [{"type": "text", "text": text}],
            }
        ],
        usage=SimpleNamespace(
            total_input_tokens=input_tokens,
            total_cached_tokens=cached_tokens,
            total_output_tokens=output_tokens,
            total_thought_tokens=thought_tokens,
            total_tokens=input_tokens + output_tokens + thought_tokens,
        ),
    )


def function_call_response(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str,
    text: str = "",
) -> Any:
    steps = []
    if text:
        steps.append(
            {
                "type": "model_output",
                "content": [{"type": "text", "text": text}],
            }
        )
    steps.extend(
        [
            {"type": "thought", "signature": "opaque"},
            {
                "type": "function_call",
                "id": call_id,
                "name": name,
                "arguments": arguments,
            },
        ]
    )
    return SimpleNamespace(
        status="requires_action",
        output_text="",
        steps=steps,
        usage=None,
    )


def gemini_model(
    responses: list[Any],
    **kwargs: Any,
) -> tuple[GeminiModel, FakeClient]:
    kwargs.setdefault("provider", "gemini")
    kwargs.setdefault("gemini_api_key", "gemini-test")
    kwargs.setdefault("gemini_model", DEFAULT_GEMINI_MODEL)
    client = FakeClient(responses)
    return GeminiModel(Settings(**kwargs), client=client), client


def test_gemini_model_sends_messages_tools_and_returns_usage() -> None:
    model, client = gemini_model(
        [
            text_response(
                "result",
                input_tokens=1_000,
                cached_tokens=100,
                output_tokens=20,
                thought_tokens=30,
            )
        ]
    )
    tool = {
        "type": "function",
        "name": "write",
        "description": "write a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": False,
    }

    result = run(
        model.query(
            [
                {"role": "system", "content": "write clean code"},
                {"role": "user", "content": "build a hinge"},
            ],
            tools=[tool],
        )
    )

    assert result["text"] == "result"
    assert result["tool_calls"] == []
    assert result["cost"] == 0.00174
    assert result["token_usage"] == {
        "input_tokens": 1_000,
        "cached_input_tokens": 100,
        "output_tokens": 20,
        "thought_tokens": 30,
        "total_tokens": 1_050,
    }
    request = client.interactions.requests[0]
    assert request["model"] == "gemini-3.6-flash"
    assert request["input"] == [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "build a hinge"}],
        }
    ]
    assert request["system_instruction"] == "write clean code"
    assert request["store"] is False
    assert request["tools"] == [
        {
            "type": "function",
            "name": "write",
            "description": "write a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]


def test_gemini_model_sends_initial_reference_image() -> None:
    model, client = gemini_model([text_response("result")])

    run(
        model.query(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "reconstruct this"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,YWJj",
                            "detail": "original",
                        },
                    ],
                }
            ]
        )
    )

    assert client.interactions.requests[0]["input"] == [
        {
            "type": "user_input",
            "content": [
                {"type": "text", "text": "reconstruct this"},
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "data": "YWJj",
                    "resolution": "ultra_high",
                },
            ],
        }
    ]


def test_gemini_model_preserves_response_steps_for_tool_results() -> None:
    model, client = gemini_model(
        [
            function_call_response(
                "compile",
                {},
                call_id="call_compile",
                text="I will compile the model.",
            ),
            text_response("done"),
        ]
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "build"}]

    first = run(model.query(messages, tools=[]))
    assert first["text"] == "I will compile the model."
    assert first["tool_calls"] == [{"id": "call_compile", "name": "compile", "arguments": "{}"}]

    messages.extend(
        [
            {
                "role": "assistant",
                "content": first["text"],
                "tool_calls": first["tool_calls"],
                "provider_content": first["provider_content"],
            },
            {
                "type": "function_call_output",
                "call_id": "call_compile",
                "output": json.dumps({"result": {"status": "success"}}),
            },
        ]
    )
    second = run(model.query(messages, tools=[]))

    assert second["text"] == "done"
    assert client.interactions.requests[1]["input"] == [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "build"}],
        },
        {
            "type": "model_output",
            "content": [{"type": "text", "text": "I will compile the model."}],
        },
        {"type": "thought", "signature": "opaque"},
        {
            "type": "function_call",
            "id": "call_compile",
            "name": "compile",
            "arguments": {},
        },
        {
            "type": "function_result",
            "name": "compile",
            "call_id": "call_compile",
            "result": [
                {
                    "type": "text",
                    "text": '{"result": {"status": "success"}}',
                }
            ],
        },
    ]
    TypeAdapter(interactions.InteractionsInput).validate_python(
        client.interactions.requests[1]["input"]
    )


def test_gemini_model_marks_tool_errors() -> None:
    model, client = gemini_model(
        [
            function_call_response("write", {"path": "main.py"}, call_id="call_write"),
            text_response("done"),
        ]
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "build"}]
    first = run(model.query(messages))
    messages.extend(
        [
            {
                "role": "assistant",
                "content": first["text"],
                "tool_calls": first["tool_calls"],
                "provider_content": first["provider_content"],
            },
            {
                "type": "function_call_output",
                "call_id": "call_write",
                "output": '{"error": "write failed"}',
            },
        ]
    )

    run(model.query(messages))

    result = client.interactions.requests[1]["input"][-1]
    assert result["is_error"] is True
    TypeAdapter(interactions.InteractionsInput).validate_python([result])


def test_gemini_model_preserves_unknown_response_steps() -> None:
    raw = {"type": "future_step", "payload": {"opaque": "value"}}
    response = text_response("done")
    response.steps.insert(0, interactions.UnknownStep(raw=raw))
    model, _client = gemini_model([response])

    result = run(model.query([{"role": "user", "content": "build"}]))

    assert result["provider_content"][0] == raw


def test_gemini_summary_request() -> None:
    model, client = gemini_model([text_response("checkpoint", input_tokens=100, output_tokens=20)])

    result = run(
        model.summarize_context(
            [
                {"role": "system", "content": "summarize"},
                {"role": "user", "content": "old work"},
            ],
            max_output_tokens=8_192,
        )
    )

    assert result["text"] == "checkpoint"
    request = client.interactions.requests[0]
    assert request["generation_config"] == {"max_output_tokens": 8_192}
    assert request["system_instruction"] == "summarize"
    assert request["tools"] is None


def test_gemini_model_converts_image_tool_results() -> None:
    model, client = gemini_model(
        [
            function_call_response("view_image", {"path": "preview.png"}, call_id="call_image"),
            text_response("done"),
        ]
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "inspect"}]
    first = run(model.query(messages))
    messages.extend(
        [
            {
                "role": "assistant",
                "content": first["text"],
                "tool_calls": first["tool_calls"],
                "provider_content": first["provider_content"],
            },
            {
                "type": "function_call_output",
                "call_id": "call_image",
                "output": [
                    {"type": "input_text", "text": '{"width": 1, "height": 1}'},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,YWJj",
                        "detail": "high",
                    },
                ],
            },
        ]
    )

    run(model.query(messages))

    assert client.interactions.requests[1]["input"][-1] == {
        "type": "function_result",
        "name": "view_image",
        "call_id": "call_image",
        "result": [
            {"type": "text", "text": '{"width": 1, "height": 1}'},
            {
                "type": "image",
                "mime_type": "image/png",
                "data": "YWJj",
                "resolution": "high",
            },
        ],
    }


def test_gemini_model_charges_pro_long_context_and_thought_tokens() -> None:
    model, _client = gemini_model(
        [
            text_response(
                "result",
                input_tokens=200_001,
                cached_tokens=100_000,
                output_tokens=1_000,
                thought_tokens=500,
            )
        ],
        gemini_model="gemini-3.1-pro-preview",
    )

    result = run(model.query([{"role": "user", "content": "build"}]))

    assert result["cost"] == 0.467004


def test_gemini_model_rejects_unsupported_models() -> None:
    with pytest.raises(ModelError, match="Unsupported Gemini model"):
        GeminiModel(
            Settings(
                provider="gemini",
                gemini_api_key="gemini-test",
                gemini_model="gemini-3.5-flash",
            )
        )


def test_gemini_model_exposes_conservative_context_window() -> None:
    model, _client = gemini_model([text_response("done")])

    assert model.context_window_tokens == 272_000
    assert context_window_tokens_for("gemini-3.6-flash") == 272_000
    assert context_window_tokens_for("gemini-3.1-pro-preview") == 272_000
    assert context_window_tokens_for("gemini-3.5-flash") is None


def test_gemini_model_wraps_provider_errors_and_closes_client() -> None:
    class AuthError(Exception):
        pass

    model, client = gemini_model([AuthError("bad key")])

    with pytest.raises(ModelError, match=r"Gemini request failed: AuthError.*bad key"):
        run(model.query([{"role": "user", "content": "build"}]))

    assert len(client.interactions.requests) == 1
    run(model.close())
    run(model.close())
    assert client.closed is True


def test_gemini_model_rejects_incomplete_interactions() -> None:
    response = text_response("partial")
    response.status = "incomplete"
    model, _client = gemini_model([response])

    with pytest.raises(ModelError, match="ended with status incomplete"):
        run(model.query([{"role": "user", "content": "build"}]))


def test_gemini_model_loads_dotenv_key(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")
    get_settings.cache_clear()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    tmp_path.joinpath(".env").write_text(
        "\n".join(
            [
                "MINI_ARTICRAFT_PROVIDER=gemini",
                "GEMINI_API_KEY=gemini-test",
                "MINI_ARTICRAFT_GEMINI_MODEL=gemini-3.1-pro-preview",
            ]
        )
    )

    settings = get_settings()

    assert settings.provider == "gemini"
    assert settings.gemini_model == "gemini-3.1-pro-preview"
    assert settings.gemini_api_key == "gemini-test"
