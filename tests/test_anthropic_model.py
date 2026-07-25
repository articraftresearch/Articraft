from __future__ import annotations

import asyncio
import json
from datetime import date
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from mini_articraft.errors import ModelError
from mini_articraft.models.anthropic import (
    AnthropicModel,
    _response_cost,
    _response_token_usage,
    anthropic_api_key_value,
    context_window_tokens_for,
    max_output_tokens_for,
)
from mini_articraft.settings import DEFAULT_ANTHROPIC_MODEL, Settings, get_settings


def run(awaitable):
    return asyncio.get_event_loop().run_until_complete(awaitable)


class FakeMessages:
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
        self.messages = FakeMessages(responses)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def text_response(
    text: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> Any:
    return SimpleNamespace(
        content=[{"type": "text", "text": text}],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
    )


def tool_use_response(name: str, tool_input: dict[str, Any], *, call_id: str) -> Any:
    return SimpleNamespace(
        content=[
            {
                "type": "thinking",
                "thinking": "",
                "signature": "signed-thinking",
            },
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": tool_input,
            },
        ],
    )


def anthropic_model(
    responses: list[Any],
    **kwargs: Any,
) -> tuple[AnthropicModel, FakeClient]:
    kwargs.setdefault("provider", "anthropic")
    kwargs.setdefault("anthropic_api_key", "anthropic-test")
    kwargs.setdefault("anthropic_model", DEFAULT_ANTHROPIC_MODEL)
    client = FakeClient(responses)
    return AnthropicModel(Settings(**kwargs), client=client), client


def test_anthropic_model_sends_messages_tools_and_returns_text_and_cost() -> None:
    model, client = anthropic_model(
        [
            text_response(
                "result",
                input_tokens=1_000,
                output_tokens=20,
                cache_creation_input_tokens=100,
                cache_read_input_tokens=50,
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
    assert result["provider_content"] == [{"type": "text", "text": "result"}]
    assert result["cost"] == 0.00246
    assert result["token_usage"] == {
        "input_tokens": 1_000,
        "output_tokens": 20,
        "cache_creation_input_tokens": 100,
        "cache_creation_5m_input_tokens": 100,
        "cache_creation_1h_input_tokens": 0,
        "cache_read_input_tokens": 50,
        "cached_tokens": 50,
        "cached_input_tokens": 50,
        "total_tokens": 1_170,
    }
    request = client.messages.requests[0]
    assert request["model"] == "claude-sonnet-5"
    assert request["max_tokens"] == 128_000
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["system"] == "write clean code"
    assert request["messages"] == [{"role": "user", "content": "build a hinge"}]
    assert request["tools"] == [
        {
            "name": "write",
            "description": "write a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            "strict": False,
        }
    ]


def test_anthropic_model_converts_tool_calls_and_tool_results() -> None:
    model, client = anthropic_model(
        [
            tool_use_response("compile", {}, call_id="call_compile"),
            text_response("done"),
        ]
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "build"}]

    first = run(model.query(messages, tools=[]))
    assert first["tool_calls"] == [{"id": "call_compile", "name": "compile", "arguments": "{}"}]
    assert first["provider_content"] == [
        {
            "type": "thinking",
            "thinking": "",
            "signature": "signed-thinking",
        },
        {
            "type": "tool_use",
            "id": "call_compile",
            "name": "compile",
            "input": {},
        },
    ]

    messages.extend(
        [
            {"role": "assistant", "content": "", "tool_calls": first["tool_calls"]},
            {
                "type": "function_call_output",
                "call_id": "call_compile",
                "output": json.dumps({"result": {"status": "success"}}),
            },
        ]
    )
    second = run(model.query(messages, tools=[]))

    assert second["text"] == "done"
    assert client.messages.requests[1]["messages"] == [
        {"role": "user", "content": "build"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "signed-thinking",
                },
                {
                    "type": "tool_use",
                    "id": "call_compile",
                    "name": "compile",
                    "input": {},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_compile",
                    "content": '{"result": {"status": "success"}}',
                }
            ],
        },
    ]


def test_anthropic_model_preserves_all_thinking_across_tool_rounds() -> None:
    first_content = [
        {
            "type": "thinking",
            "thinking": "",
            "signature": "first-signature",
        },
        {"type": "text", "text": "I will inspect the file."},
        {
            "type": "tool_use",
            "id": "call_read",
            "name": "read",
            "input": {"path": "main.py"},
        },
    ]
    second_content = [
        {
            "type": "redacted_thinking",
            "data": "redacted-thinking-data",
        },
        {
            "type": "thinking",
            "thinking": "",
            "signature": "second-signature",
        },
        {
            "type": "tool_use",
            "id": "call_compile",
            "name": "compile",
            "input": {},
        },
    ]
    model, client = anthropic_model(
        [
            SimpleNamespace(content=first_content),
            SimpleNamespace(content=second_content),
            text_response("done"),
        ]
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "build"}]

    first = run(model.query(messages, tools=[]))
    messages.extend(
        [
            {
                "role": "assistant",
                "content": first["text"],
                "tool_calls": first["tool_calls"],
            },
            {
                "type": "function_call_output",
                "call_id": "call_read",
                "output": '{"result": "file contents"}',
            },
        ]
    )
    second = run(model.query(messages, tools=[]))
    messages.extend(
        [
            {
                "role": "assistant",
                "content": second["text"],
                "tool_calls": second["tool_calls"],
            },
            {
                "type": "function_call_output",
                "call_id": "call_compile",
                "output": '{"result": {"status": "success"}}',
            },
        ]
    )

    run(model.query(messages, tools=[]))

    third_messages = client.messages.requests[2]["messages"]
    assert third_messages[1] == {"role": "assistant", "content": first_content}
    assert third_messages[3] == {"role": "assistant", "content": second_content}


def test_anthropic_model_groups_consecutive_tool_results() -> None:
    model, client = anthropic_model([text_response("done")])

    run(
        model.query(
            [
                {"role": "user", "content": "build"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_write", "name": "write", "arguments": '{"path": "main.py"}'},
                        {"id": "call_compile", "name": "compile", "arguments": "{}"},
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_write",
                    "output": '{"ok": true}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_compile",
                    "output": '{"status": "success"}',
                },
            ],
            tools=[],
        )
    )

    assert client.messages.requests[0]["messages"][2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_write",
                "content": '{"ok": true}',
            },
            {
                "type": "tool_result",
                "tool_use_id": "call_compile",
                "content": '{"status": "success"}',
            },
        ],
    }


def test_anthropic_model_sends_initial_reference_image() -> None:
    model, client = anthropic_model([text_response("done")])

    run(
        model.query(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "reconstruct this"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,aW1hZ2U=",
                            "detail": "original",
                        },
                    ],
                },
            ],
            tools=[],
        )
    )

    assert client.messages.requests[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "reconstruct this"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "aW1hZ2U=",
                    },
                },
            ],
        }
    ]


def test_anthropic_model_sends_url_image() -> None:
    model, client = anthropic_model([text_response("done")])

    run(
        model.query(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/reference.webp",
                        },
                        {"type": "input_text", "text": "reconstruct this"},
                    ],
                },
            ],
            tools=[],
        )
    )

    assert client.messages.requests[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/reference.webp",
                    },
                },
                {"type": "text", "text": "reconstruct this"},
            ],
        }
    ]


def test_anthropic_model_sends_typed_image_tool_output() -> None:
    model, client = anthropic_model([text_response("done")])

    run(
        model.query(
            [
                {"role": "user", "content": "inspect"},
                {
                    "type": "function_call_output",
                    "call_id": "call_image",
                    "output": [
                        {"type": "input_text", "text": '{"result": {"path": "image.png"}}'},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,aW1hZ2U=",
                            "detail": "high",
                        },
                    ],
                },
            ],
            tools=[],
        )
    )

    assert client.messages.requests[0]["messages"][1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_image",
                "content": [
                    {"type": "text", "text": '{"result": {"path": "image.png"}}'},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aW1hZ2U=",
                        },
                    },
                ],
            }
        ],
    }


def test_anthropic_model_marks_tool_errors() -> None:
    model, client = anthropic_model([text_response("done")])

    run(
        model.query(
            [
                {"role": "user", "content": "build"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_write", "name": "write", "arguments": "{}"},
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_write",
                    "output": '{"error": "write failed"}',
                },
            ],
            tools=[],
        )
    )

    assert client.messages.requests[0]["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_write",
        "content": '{"error": "write failed"}',
        "is_error": True,
    }


def test_anthropic_model_prices_mixed_cache_durations() -> None:
    usage = {
        "input_tokens": 1_000,
        "output_tokens": 20,
        "cache_creation_input_tokens": 150,
        "cache_creation_5m_input_tokens": 100,
        "cache_creation_1h_input_tokens": 50,
        "cache_read_input_tokens": 25,
    }

    assert _response_cost("claude-opus-5", usage) == 0.0066375


def test_anthropic_model_reads_cache_duration_breakdown() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1_000,
            output_tokens=20,
            cache_creation_input_tokens=150,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=100,
                ephemeral_1h_input_tokens=50,
            ),
            cache_read_input_tokens=25,
        )
    )

    assert _response_token_usage(response) == {
        "input_tokens": 1_000,
        "output_tokens": 20,
        "cache_creation_input_tokens": 150,
        "cache_creation_5m_input_tokens": 100,
        "cache_creation_1h_input_tokens": 50,
        "cache_read_input_tokens": 25,
        "cached_tokens": 25,
        "cached_input_tokens": 25,
        "total_tokens": 1_195,
    }


def test_anthropic_model_returns_sonnet_standard_cost_after_price_change() -> None:
    usage = {
        "input_tokens": 1_000,
        "output_tokens": 20,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 50,
    }

    assert _response_cost("claude-sonnet-5", usage, today=date(2026, 8, 31)) == 0.00246
    assert _response_cost("claude-sonnet-5", usage, today=date(2026, 9, 1)) == 0.00369


def test_anthropic_model_rejects_unsupported_models() -> None:
    with pytest.raises(ModelError, match="Unsupported Anthropic model"):
        AnthropicModel(
            Settings(
                provider="anthropic",
                anthropic_api_key="anthropic-test",
                anthropic_model="claude-haiku-4-5",
            )
        )

    assert context_window_tokens_for("claude-sonnet-5") == 1_000_000
    assert context_window_tokens_for("claude-opus-5") == 1_000_000
    assert context_window_tokens_for("claude-haiku-4-5") is None
    assert max_output_tokens_for("claude-sonnet-5") == 128_000


def test_anthropic_model_rejects_too_many_output_tokens() -> None:
    with pytest.raises(ModelError, match="cannot exceed 128000"):
        AnthropicModel(
            Settings(
                provider="anthropic",
                anthropic_api_key="anthropic-test",
                anthropic_model="claude-opus-5",
                anthropic_max_output_tokens=128_001,
            )
        )


def test_anthropic_model_configures_sdk_retries_and_timeout(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    client = SimpleNamespace()

    def create_client(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", create_client)
    model = AnthropicModel(
        Settings(
            provider="anthropic",
            anthropic_api_key="anthropic-test",
            anthropic_max_attempts=4,
            anthropic_request_timeout_seconds=123.0,
        )
    )

    assert model._client_or_create() is client
    assert captured == {
        "api_key": "anthropic-test",
        "max_retries": 3,
        "timeout": 123.0,
    }


def test_anthropic_model_wraps_provider_errors() -> None:
    class AuthError(Exception):
        status_code = 401

    model, client = anthropic_model([AuthError("bad key")])

    with pytest.raises(ModelError, match=r"Anthropic request failed: AuthError.*bad key"):
        run(model.query([{"role": "user", "content": "build"}]))

    assert len(client.messages.requests) == 1


def test_anthropic_model_loads_dotenv_key(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tmp_path.joinpath(".env").write_text(
        "\n".join(
            [
                "MINI_ARTICRAFT_PROVIDER=anthropic",
                "ANTHROPIC_API_KEY=anthropic-test",
                "MINI_ARTICRAFT_ANTHROPIC_MODEL=claude-opus-5",
            ]
        )
    )

    settings = get_settings()

    assert settings.provider == "anthropic"
    assert settings.anthropic_model == "claude-opus-5"
    assert anthropic_api_key_value(settings) == "anthropic-test"
