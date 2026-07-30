from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from harness import GOOD_MAIN_PY

from mini_articraft.agent import Agent
from mini_articraft.environments import LocalEnvironment
from mini_articraft.errors import ModelError
from mini_articraft.models import create_model
from mini_articraft.models.openrouter import OpenRouterModel
from mini_articraft.settings import DEFAULT_OPENROUTER_MODEL, Settings, get_settings


def run(awaitable):
    return asyncio.get_event_loop().run_until_complete(awaitable)


def response(
    *,
    text: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    reasoning: str | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": "gen-test",
        "model": "vendor/arbitrary-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls or [],
                    **({"reasoning": reasoning} if reasoning is not None else {}),
                    **(
                        {"reasoning_details": reasoning_details}
                        if reasoning_details is not None
                        else {}
                    ),
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        **({"usage": usage} if usage is not None else {}),
    }


def function_call(name: str, arguments: dict[str, Any], *, call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def model_with_responses(
    responses: list[dict[str, Any] | httpx.Response | BaseException],
    **settings: Any,
) -> tuple[OpenRouterModel, httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(200, json=item)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    settings.setdefault("provider", "openrouter")
    settings.setdefault("openrouter_api_key", "or-test")
    settings.setdefault("openrouter_model", "vendor/arbitrary-model")
    return OpenRouterModel(Settings(**settings), client=client), client, requests


def request_json(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def test_openrouter_model_sends_messages_tools_and_returns_usage_and_cost() -> None:
    usage = {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 20},
        "cost": 0.0042,
    }
    model, client, requests = model_with_responses(
        [response(text="done", usage=usage)],
        openrouter_http_referer="https://mini-articraft.example",
        openrouter_app_title="mini-articraft",
        openrouter_request_timeout_seconds=45,
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

    assert result["text"] == "done"
    assert result["tool_calls"] == []
    assert result["cost"] == 0.0042
    assert result["token_usage"] == {
        "input_tokens": 120,
        "cached_input_tokens": 20,
        "output_tokens": 30,
        "total_tokens": 150,
    }
    assert result["response"]["id"] == "gen-test"
    request = requests[0]
    assert request.url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer or-test"
    assert request.headers["HTTP-Referer"] == "https://mini-articraft.example"
    assert request.headers["X-OpenRouter-Title"] == "mini-articraft"
    assert request.extensions["timeout"] == {
        "connect": 45,
        "read": 45,
        "write": 45,
        "pool": 45,
    }
    assert request_json(request) == {
        "model": "vendor/arbitrary-model",
        "messages": [
            {"role": "system", "content": "write clean code"},
            {"role": "user", "content": "build a hinge"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "write a file",
                    "parameters": tool["parameters"],
                    "strict": False,
                },
            }
        ],
    }
    run(model.close())
    assert client.is_closed
    run(model.close())


def test_openrouter_model_converts_multiple_tool_calls_and_results() -> None:
    reasoning_details = [
        {
            "type": "reasoning.text",
            "text": "I should write the file first.",
            "id": "reasoning-1",
            "format": "unknown",
            "index": 0,
        }
    ]
    first_calls = [
        function_call("write", {"path": "main.py"}, call_id="call_write"),
        function_call("compile", {}, call_id="call_compile"),
    ]
    model, _, requests = model_with_responses(
        [
            response(
                tool_calls=first_calls,
                reasoning="I should write the file first.",
                reasoning_details=reasoning_details,
            ),
            response(text="done"),
        ]
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "build"}]

    first = run(model.query(messages, tools=[]))
    assert first["tool_calls"] == [
        {"id": "call_write", "name": "write", "arguments": '{"path": "main.py"}'},
        {"id": "call_compile", "name": "compile", "arguments": "{}"},
    ]
    assert first["provider_content"] == [
        {
            "type": "openrouter_reasoning",
            "reasoning": "I should write the file first.",
            "reasoning_details": reasoning_details,
        }
    ]

    messages.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": first["tool_calls"],
                "provider_content": first["provider_content"],
            },
            {
                "type": "function_call_output",
                "call_id": "call_write",
                "output": '{"result": {"path": "main.py"}}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_compile",
                "output": [{"type": "input_text", "text": '{"status": "success"}'}],
            },
        ]
    )
    second = run(model.query(messages, tools=[]))

    assert second["text"] == "done"
    assert request_json(requests[1])["messages"] == [
        {"role": "user", "content": "build"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": first_calls,
            "reasoning": "I should write the file first.",
            "reasoning_details": reasoning_details,
        },
        {
            "role": "tool",
            "tool_call_id": "call_write",
            "content": '{"result": {"path": "main.py"}}',
        },
        {
            "role": "tool",
            "tool_call_id": "call_compile",
            "content": '{"status": "success"}',
        },
    ]


def test_openrouter_model_runs_full_generate_compile_loop(tmp_path: Path) -> None:
    model, client, requests = model_with_responses(
        [
            response(
                tool_calls=[
                    function_call(
                        "write",
                        {"path": "main.py", "content": GOOD_MAIN_PY},
                        call_id="call_write",
                    )
                ],
                usage={"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001},
            ),
            response(
                tool_calls=[function_call("compile", {}, call_id="call_compile")],
                usage={"prompt_tokens": 200, "completion_tokens": 10, "cost": 0.002},
            ),
            response(
                text="done",
                usage={"prompt_tokens": 300, "completion_tokens": 5, "cost": 0.003},
            ),
        ]
    )
    agent = Agent(model, LocalEnvironment(output_dir=tmp_path), max_turns=3)

    result = run(agent.run("a box", run_id="openrouter-box"))

    assert result["status"] == "success"
    assert result["cost"] == 0.006
    assert result["token_usage"] == {
        "input_tokens": 600,
        "cached_input_tokens": 0,
        "output_tokens": 35,
        "total_tokens": 635,
    }
    assert len(requests) == 3
    assert request_json(requests[1])["messages"][-1]["role"] == "tool"
    assert request_json(requests[2])["messages"][-1]["role"] == "tool"
    assert client.is_closed


def test_openrouter_model_accepts_unknown_model_and_has_unknown_context() -> None:
    model, _, _ = model_with_responses([response(text="done")])

    assert model.config.openrouter_model == "vendor/arbitrary-model"
    assert model.context_window_tokens == 0
    result = run(model.query([{"role": "user", "content": "build"}]))
    assert result["token_usage"] == {}
    assert result["cost"] == 0.0


def test_openrouter_model_uses_default_model() -> None:
    model = OpenRouterModel(
        Settings(
            provider="openrouter",
            openrouter_api_key="or-test",
        )
    )

    assert model.config.openrouter_model == DEFAULT_OPENROUTER_MODEL
    run(model.close())


def test_model_factory_selects_openrouter() -> None:
    model = create_model(
        Settings(
            provider="openrouter",
            openrouter_api_key="or-test",
            openrouter_model="vendor/arbitrary-model",
        )
    )

    assert isinstance(model, OpenRouterModel)
    run(model.close())


def test_openrouter_settings_load_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MINI_ARTICRAFT_OPENROUTER_MODEL", raising=False)
    tmp_path.joinpath(".env").write_text(
        "\n".join(
            [
                "MINI_ARTICRAFT_PROVIDER=openrouter",
                "OPENROUTER_API_KEY=or-test",
                "MINI_ARTICRAFT_OPENROUTER_MODEL=vendor/new-model",
                "OPENROUTER_HTTP_REFERER=https://mini-articraft.example",
                "OPENROUTER_APP_TITLE=mini-articraft",
            ]
        )
    )

    settings = get_settings()

    assert settings.provider == "openrouter"
    assert settings.selected_model == "vendor/new-model"
    assert settings.openrouter_api_key == "or-test"
    assert settings.openrouter_http_referer == "https://mini-articraft.example"
    assert settings.openrouter_app_title == "mini-articraft"
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"openrouter_api_key": "", "openrouter_model": "vendor/model"},
            "OPENROUTER_API_KEY",
        ),
        (
            {"openrouter_api_key": "or-test", "openrouter_model": ""},
            "MINI_ARTICRAFT_OPENROUTER_MODEL",
        ),
    ],
)
def test_openrouter_model_requires_credentials_and_model(
    settings: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ModelError, match=message):
        OpenRouterModel(Settings.model_validate({"provider": "openrouter", **settings}))


def test_openrouter_model_rejects_image_content_without_request() -> None:
    model, _, requests = model_with_responses([response(text="unused")])

    with pytest.raises(ModelError, match="not images"):
        run(
            model.query(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "reconstruct"},
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,aW1hZ2U=",
                            },
                        ],
                    }
                ]
            )
        )

    assert requests == []


def test_openrouter_model_does_not_retry_permanent_provider_error() -> None:
    error = httpx.Response(
        401,
        json={"error": {"code": 401, "message": "invalid API key"}},
    )
    model, _, requests = model_with_responses([error])

    with pytest.raises(ModelError, match=r"HTTP 401.*invalid API key"):
        run(model.query([{"role": "user", "content": "build"}]))

    assert len(requests) == 1


def test_openrouter_model_retries_rate_limit_and_honors_retry_after(monkeypatch) -> None:
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("mini_articraft.models.openrouter.asyncio.sleep", sleep)
    rate_limit = httpx.Response(
        429,
        headers={"Retry-After": "2"},
        json={"error": {"code": 429, "message": "rate limited"}},
    )
    model, _, requests = model_with_responses([rate_limit, response(text="done")])

    result = run(model.query([{"role": "user", "content": "build"}]))

    assert result["text"] == "done"
    assert len(requests) == 2
    assert delays == [2.0]


def test_openrouter_model_wraps_transport_error_after_retry(monkeypatch) -> None:
    async def sleep(_delay: float) -> None:
        pass

    monkeypatch.setattr("mini_articraft.models.openrouter.asyncio.sleep", sleep)
    request = httpx.Request("POST", "https://openrouter.ai")
    model, _, requests = model_with_responses(
        [
            httpx.ConnectError("offline", request=request),
            httpx.ConnectError("offline", request=request),
        ],
        openrouter_max_attempts=2,
    )

    with pytest.raises(ModelError, match=r"ConnectError.*offline"):
        run(model.query([{"role": "user", "content": "build"}]))

    assert len(requests) == 2


def test_openrouter_model_reports_embedded_provider_error() -> None:
    payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "error",
                "error": {
                    "code": 502,
                    "message": "upstream disconnected",
                },
            }
        ]
    }
    model, _, _ = model_with_responses([payload])

    with pytest.raises(ModelError, match=r"provider code 502.*upstream disconnected"):
        run(model.query([{"role": "user", "content": "build"}]))
