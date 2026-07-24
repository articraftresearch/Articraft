from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from mini_articraft.errors import ModelError
from mini_articraft.models.openai import OpenAIModel, context_window_tokens_for
from mini_articraft.settings import DEFAULT_MAX_TURNS, Settings, get_settings


def run(awaitable):
    return asyncio.get_event_loop().run_until_complete(awaitable)


def response_event(
    text: str,
    *,
    model: str = "gpt-5.5-2026-04-23",
    response_id: str = "resp_1",
    status: str = "completed",
    incomplete_details: dict[str, object] | None = None,
    output: list[dict[str, object]] | None = None,
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    response = {
        "type": f"response.{status}",
        "response": {
            "id": response_id,
            "status": status,
            "model": model,
            "incomplete_details": incomplete_details,
            "output": output
            if output is not None
            else [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        },
    }
    if usage is not None:
        response["response"]["usage"] = usage
    return response


class FakeWebSocket:
    def __init__(self, events: list[dict[str, object]]):
        self.closed = False
        self.sent: list[dict[str, object]] = []
        self.events = [json.dumps(event) for event in events]

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def recv(self) -> str:
        return self.events.pop(0)

    async def close(self) -> None:
        self.closed = True


def patch_websocket(monkeypatch: pytest.MonkeyPatch, socket: FakeWebSocket) -> None:
    async def connect(
        url: str,
        *,
        additional_headers: dict[str, str],
        open_timeout: float,
        max_size: object,
    ):
        assert url == "wss://api.openai.com/v1/responses"
        assert additional_headers == {"Authorization": "Bearer sk-test"}
        assert open_timeout == 20.0
        assert max_size is None
        return socket

    monkeypatch.setattr("mini_articraft.models.openai.websockets.connect", connect)


def openai_model(**kwargs: Any) -> OpenAIModel:
    kwargs.setdefault("openai_model", "gpt-5.5-2026-04-23")
    kwargs.setdefault("openai_reasoning_effort", "high")
    return OpenAIModel(Settings(openai_api_key="sk-test", **kwargs))


def test_openai_model_uses_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket([response_event("result")])
    patch_websocket(monkeypatch, socket)

    result = run(
        openai_model().query(
            [
                {"role": "system", "content": "write clean code"},
                {"role": "user", "content": "build a hinge"},
            ]
        )
    )

    assert result["text"] == "result"
    assert socket.sent == [
        {
            "type": "response.create",
            "model": "gpt-5.5-2026-04-23",
            "input": [{"role": "user", "content": "build a hinge"}],
            "reasoning": {"effort": "high"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "max_output_tokens": 128_000,
            "instructions": "write clean code",
        }
    ]


def test_openai_model_sends_initial_reference_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket([response_event("result")])
    patch_websocket(monkeypatch, socket)
    content = [
        {"type": "input_text", "text": "reconstruct this"},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,YWJj",
            "detail": "original",
        },
    ]

    run(openai_model().query([{"role": "user", "content": content}]))

    assert socket.sent[0]["input"] == [{"role": "user", "content": content}]


def test_openai_model_sends_tools_and_returns_function_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket(
        [
            response_event(
                "",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "write",
                        "arguments": '{"path": "main.py", "content": "x"}',
                    }
                ],
            )
        ]
    )
    patch_websocket(monkeypatch, socket)
    tool = {
        "type": "function",
        "name": "write",
        "description": "write",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    result = run(openai_model().query([{"role": "user", "content": "build"}], tools=[tool]))

    assert result["text"] == ""
    assert result["tool_calls"] == [
        {
            "id": "call_1",
            "name": "write",
            "arguments": '{"path": "main.py", "content": "x"}',
        }
    ]
    assert socket.sent[0]["tools"] == [tool]
    assert socket.sent[0]["parallel_tool_calls"] is True


def test_openai_model_returns_estimated_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket(
        [
            response_event(
                "result",
                usage={
                    "input_tokens": 1_000,
                    "input_tokens_details": {"cached_tokens": 100},
                    "output_tokens": 20,
                },
            )
        ]
    )
    patch_websocket(monkeypatch, socket)

    result = run(openai_model().query([{"role": "user", "content": "build"}]))

    assert result["cost"] == 0.00515
    assert result["token_usage"] == {
        "input_tokens": 1_000,
        "cached_input_tokens": 100,
        "cache_write_tokens": 0,
        "output_tokens": 20,
        "total_tokens": 1_020,
    }


def test_openai_model_returns_estimated_cost_for_gpt_5_6_sol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket(
        [
            response_event(
                "result",
                model="gpt-5.6-sol",
                usage={
                    "input_tokens": 1_000,
                    "input_tokens_details": {
                        "cached_tokens": 100,
                        "cache_write_tokens": 200,
                    },
                    "output_tokens": 20,
                },
            )
        ]
    )
    patch_websocket(monkeypatch, socket)

    result = run(
        openai_model(openai_model="gpt-5.6-sol").query([{"role": "user", "content": "build"}])
    )

    assert result["cost"] == 0.0054
    assert result["token_usage"] == {
        "input_tokens": 1_000,
        "cached_input_tokens": 100,
        "cache_write_tokens": 200,
        "output_tokens": 20,
        "total_tokens": 1_020,
    }


def test_openai_model_exposes_context_window() -> None:
    assert openai_model().config.openai_model == "gpt-5.5-2026-04-23"
    assert DEFAULT_MAX_TURNS == 100
    assert openai_model().context_window_tokens == 272_000
    assert context_window_tokens_for("gpt-5.6-sol") == 272_000
    assert context_window_tokens_for("gpt-5.6") == 272_000
    assert context_window_tokens_for("gpt-5.6-terra") is None
    assert context_window_tokens_for("gpt-5.5-2026-04-23") == 272_000
    assert context_window_tokens_for("gpt-5.5-pro") == 272_000
    assert context_window_tokens_for("gpt-5.4-mini-2026-03-17") == 400_000
    assert context_window_tokens_for("gpt-test") is None


def test_openai_model_sends_function_call_outputs_with_previous_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket(
        [
            response_event(
                "",
                response_id="resp_1",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "compile",
                        "arguments": "{}",
                    }
                ],
            ),
            response_event("done", response_id="resp_2"),
        ]
    )
    patch_websocket(monkeypatch, socket)
    model = openai_model()
    messages: list[dict[str, Any]] = [{"role": "user", "content": "build"}]

    run(model.query(messages, tools=[]))
    messages.extend(
        [
            {"role": "assistant", "content": "", "tool_calls": []},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"ok": true}'},
        ]
    )
    run(model.query(messages, tools=[]))

    assert socket.sent[1]["previous_response_id"] == "resp_1"
    assert socket.sent[1]["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": '{"ok": true}'}
    ]


def test_openai_model_sends_typed_image_tool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket(
        [
            response_event(
                "",
                response_id="resp_1",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "call_image",
                        "name": "view_image",
                        "arguments": '{"path": "image.png"}',
                    }
                ],
            ),
            response_event("done", response_id="resp_2"),
        ]
    )
    patch_websocket(monkeypatch, socket)
    model = openai_model(openai_model="gpt-5.4-mini")
    messages: list[dict[str, Any]] = [{"role": "user", "content": "inspect"}]

    run(model.query(messages, tools=[]))
    image_output = {
        "type": "function_call_output",
        "call_id": "call_image",
        "output": [
            {"type": "input_text", "text": '{"result": {"path": "image.png"}}'},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,aW1hZ2U=",
                "detail": "original",
            },
        ],
    }
    messages.extend(
        [
            {"role": "assistant", "content": "", "tool_calls": []},
            image_output,
        ]
    )
    run(model.query(messages, tools=[]))

    assert socket.sent[1]["previous_response_id"] == "resp_1"
    sent_output = cast(list[dict[str, Any]], socket.sent[1]["input"])[0]
    assert sent_output["output"][1]["detail"] == "high"


def test_openai_model_uses_incremental_websocket_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket(
        [
            response_event(
                "first",
                response_id="resp_1",
                output=[
                    {"type": "reasoning", "encrypted_content": "encrypted"},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "first"}],
                    },
                ],
            ),
            response_event("second", response_id="resp_2"),
        ]
    )
    patch_websocket(monkeypatch, socket)
    model = openai_model()

    run(model.query([{"role": "system", "content": "system"}, {"role": "user", "content": "one"}]))
    run(
        model.query(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "two"},
            ]
        )
    )

    assert socket.sent[1]["previous_response_id"] == "resp_1"
    assert socket.sent[1]["input"] == [{"role": "user", "content": "two"}]
    assert model._input_items[1] == {"type": "reasoning", "encrypted_content": "encrypted"}


def test_openai_model_round_trips_response_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket([response_event("result")])
    patch_websocket(monkeypatch, socket)

    run(
        openai_model().query(
            [
                {"role": "system", "content": "system contract"},
                {"role": "assistant", "content": "I will inspect the compile error."},
                {"id": "rs_123", "type": "reasoning", "summary": []},
                {"role": "user", "content": "continue"},
            ],
        )
    )

    assert socket.sent[0]["input"] == [
        {"role": "assistant", "content": "I will inspect the compile error."},
        {"id": "rs_123", "type": "reasoning", "summary": []},
        {"role": "user", "content": "continue"},
    ]


def test_openai_model_loads_dotenv(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MINI_ARTICRAFT_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("MINI_ARTICRAFT_MODEL", raising=False)
    monkeypatch.delenv("MINI_ARTICRAFT_REASONING_EFFORT", raising=False)
    tmp_path.joinpath(".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test",
                "MINI_ARTICRAFT_OUTPUT_DIR=custom-runs",
                "MINI_ARTICRAFT_MODEL=gpt-test",
                "MINI_ARTICRAFT_REASONING_EFFORT=low",
            ]
        )
    )
    socket = FakeWebSocket([response_event("result")])
    patch_websocket(monkeypatch, socket)

    model = OpenAIModel()
    run(model.query([{"role": "user", "content": "hello"}]))

    assert model.config.output_dir.as_posix() == "custom-runs"
    assert socket.sent[0]["model"] == "gpt-test"
    assert socket.sent[0]["reasoning"] == {"effort": "low"}
    assert socket.sent[0]["max_output_tokens"] == 128_000
    assert socket.sent[0]["include"] == ["reasoning.encrypted_content"]
    assert socket.sent[0]["store"] is False


def test_openai_model_raises_on_incomplete_response(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket(
        [
            response_event(
                "",
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
                output=[],
            )
        ]
    )
    patch_websocket(monkeypatch, socket)

    with pytest.raises(ModelError, match=r"max_output_tokens.*no visible output"):
        run(openai_model().query([{"role": "user", "content": "hello"}]))


def test_openai_model_raises_on_partial_incomplete_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket(
        [
            response_event(
                "partial",
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            )
        ]
    )
    patch_websocket(monkeypatch, socket)

    with pytest.raises(ModelError, match="partial output returned"):
        run(openai_model().query([{"role": "user", "content": "hello"}]))


def test_openai_model_raises_on_websocket_error(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket(
        [
            {
                "type": "error",
                "status": 400,
                "error": {
                    "code": "previous_response_not_found",
                    "message": "missing response",
                },
            }
        ]
    )
    patch_websocket(monkeypatch, socket)

    with pytest.raises(ModelError, match="previous_response_not_found"):
        run(openai_model().query([{"role": "user", "content": "hello"}]))


def test_openai_model_retries_transient_errors_on_a_new_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakeWebSocket(
        [
            {
                "type": "error",
                "status": 503,
                "error": {"code": "server_error", "message": "try again"},
            }
        ]
    )
    second = FakeWebSocket([response_event("result")])
    sockets = iter([first, second])
    sleeps: list[float] = []

    async def connect(*args: Any, **kwargs: Any) -> FakeWebSocket:
        return next(sockets)

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("mini_articraft.models.openai.websockets.connect", connect)
    monkeypatch.setattr("mini_articraft.models.openai.asyncio.sleep", sleep)
    monkeypatch.setattr("mini_articraft.models.openai.random.random", lambda: 0.5)

    result = run(openai_model().query([{"role": "user", "content": "hello"}]))

    assert result["text"] == "result"
    assert first.closed is True
    assert sleeps == [0.25]
    assert first.sent == second.sent


def test_openai_model_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket(
        [
            {
                "type": "error",
                "status": 401,
                "error": {"code": "invalid_api_key", "message": "bad key"},
            }
        ]
    )
    patch_websocket(monkeypatch, socket)

    with pytest.raises(ModelError, match="invalid_api_key"):
        run(openai_model().query([{"role": "user", "content": "hello"}]))

    assert len(socket.sent) == 1


def test_openai_model_resends_full_context_when_previous_response_is_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakeWebSocket(
        [
            response_event("first", response_id="resp_1"),
            {
                "type": "error",
                "status": 400,
                "error": {
                    "code": "previous_response_not_found",
                    "message": "missing response",
                },
            },
        ]
    )
    second = FakeWebSocket([response_event("second", response_id="resp_2")])
    sockets = iter([first, second])

    async def connect(*args: Any, **kwargs: Any) -> FakeWebSocket:
        return next(sockets)

    monkeypatch.setattr("mini_articraft.models.openai.websockets.connect", connect)
    model = openai_model()
    messages = [{"role": "user", "content": "first question"}]

    run(model.query(messages))
    messages.extend(
        [
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "second question"},
        ]
    )
    result = run(model.query(messages))

    assert result["text"] == "second"
    assert first.closed is True
    assert first.sent[1]["previous_response_id"] == "resp_1"
    assert "previous_response_id" not in second.sent[0]
    assert second.sent[0]["input"] == [
        {"role": "user", "content": "first question"},
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "first"}],
        },
        {"role": "user", "content": "second question"},
    ]


def test_openai_model_raises_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeWebSocket([response_event("", output=[])])
    patch_websocket(monkeypatch, socket)

    with pytest.raises(ModelError, match="output_text"):
        run(openai_model().query([{"role": "user", "content": "hello"}]))


def test_openai_model_returns_completed_reasoning_only_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = FakeWebSocket(
        [
            response_event(
                "",
                output=[{"type": "reasoning", "encrypted_content": "encrypted"}],
            )
        ]
    )
    patch_websocket(monkeypatch, socket)

    result = run(openai_model().query([{"role": "user", "content": "hello"}]))

    assert result["text"] == ""
    assert "reasoning" not in result
    assert result["tool_calls"] == []


def test_close_closes_the_websocket_and_is_idempotent(monkeypatch) -> None:
    socket = FakeWebSocket([response_event("result")])
    patch_websocket(monkeypatch, socket)
    model = openai_model()

    run(model.query([{"role": "user", "content": "hi"}]))
    run(model.close())

    assert socket.closed is True
    run(model.close())  # idempotent: nothing left to close
