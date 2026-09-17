from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from articraft import api
from articraft.agent.provider import AtlasCloudModel, create_model
from articraft.app import cli
from articraft.errors import ModelError
from articraft.settings import Settings


def settings(**overrides) -> Settings:
    return Settings.model_validate(
        {"provider": "atlascloud", "atlascloud_api_key": "test-atlas-key", **overrides}
    )


def response(message=None, **overrides):
    return {
        "choices": [{"message": message or {"content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        **overrides,
    }


def test_request_converts_tools_and_preserves_the_conversation() -> None:
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=response(
                    {
                        "content": None,
                        "reasoning_content": "inspect the file",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read", "arguments": '{"path":"main.py"}'},
                            }
                        ],
                    },
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    },
                ),
            )
        return httpx.Response(200, json=response())

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        model = AtlasCloudModel(settings(), client=client)
        messages = [{"role": "user", "content": [{"type": "input_text", "text": "read the file"}]}]
        first = await model.query(
            messages,
            tools=[
                {
                    "type": "function",
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                }
            ],
        )
        assert first["text"] == ""
        assert first["tool_calls"] == [
            {"id": "call-1", "name": "read", "arguments": '{"path":"main.py"}'}
        ]
        assert first["token_usage"] == {
            "input_tokens": 10,
            "cached_input_tokens": 4,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        assert first["cost"] == 0.0
        messages += [
            {
                "role": "assistant",
                "content": first["text"],
                "tool_calls": first["tool_calls"],
                "provider_content": first["provider_content"],
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": {"text": "file contents"},
            },
        ]
        assert (await model.query(messages))["text"] == "done"
        await model.close()
        assert client.is_closed

    asyncio.run(run())
    assert str(requests[0].url) == "https://api.atlascloud.ai/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer test-atlas-key"
    body = json.loads(requests[0].content)
    assert body["model"] == "openai/gpt-4.1-mini"
    assert body["max_tokens"] == 8192
    assert body["tools"][0]["function"]["name"] == "read"
    assert "HTTP-Referer" not in requests[0].headers
    followup = json.loads(requests[1].content)["messages"]
    assert followup[1]["tool_calls"][0]["id"] == followup[2]["tool_call_id"] == "call-1"
    assert followup[1]["reasoning_content"] == "inspect the file"
    assert followup[2]["role"] == "tool"
    assert json.loads(followup[2]["content"]) == {"text": "file contents"}


@pytest.mark.parametrize("failure", [401, 429, 500, "timeout", "transport"])
def test_failed_generation_is_not_retried(failure) -> None:
    requests = []

    def handle(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("test-atlas-key", request=request)
        if failure == "transport":
            raise httpx.ConnectError("test-atlas-key", request=request)
        return httpx.Response(failure, json={"error": {"message": "test-atlas-key"}})

    async def run():
        model = AtlasCloudModel(
            settings(), client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        )
        with pytest.raises(ModelError) as error:
            await model.query([{"role": "user", "content": "hello"}])
        assert "test-atlas-key" not in str(error.value)
        await model.close()

    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"error": {"message": "upstream failed"}},
        {"choices": []},
        response({"content": ""}),
        response({"content": ["invalid"]}),
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
        response({"tool_calls": [{"function": {"name": "read"}}]}),
    ],
)
def test_invalid_or_incomplete_responses_fail(payload) -> None:
    async def run():
        model = AtlasCloudModel(
            settings(),
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
            ),
        )
        with pytest.raises(ModelError):
            await model.query([{"role": "user", "content": "hello"}])
        await model.close()

    asyncio.run(run())


def test_invalid_json_fails_without_retry() -> None:
    async def run():
        model = AtlasCloudModel(
            settings(),
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(200, text="not json"))
            ),
        )
        with pytest.raises(ModelError, match="valid JSON"):
            await model.query([{"role": "user", "content": "hello"}])
        await model.close()

    asyncio.run(run())


def test_summary_caps_output_and_omits_tools() -> None:
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response())

    async def run():
        model = AtlasCloudModel(
            settings(atlascloud_max_output_tokens=4096, atlascloud_context_window_tokens=50000),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )
        assert model.context_window_tokens == 50000
        for limit in (1000, 8192):
            result = await model.summarize_context(
                [{"role": "user", "content": "summarize"}], max_output_tokens=limit
            )
            assert set(result) == {"text", "token_usage", "cost"}
        await model.close()

    asyncio.run(run())
    assert [r["max_tokens"] for r in requests] == [1000, 4096]
    assert all("tools" not in r for r in requests)


def test_rejects_images_before_request() -> None:
    def handle(_):
        pytest.fail("image input must not send a request")

    async def run():
        model = AtlasCloudModel(
            settings(), client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        )
        assert not model.supports_images
        with pytest.raises(ModelError, match="not images"):
            await model.query(
                [{"role": "user", "content": [{"type": "input_image", "image_url": "test"}]}]
            )
        await model.close()

    asyncio.run(run())


def test_settings_factory_and_api_routing(monkeypatch) -> None:
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "environment-key")
    monkeypatch.setenv("ARTICRAFT_ATLASCLOUD_MODEL", "vendor/custom-model")
    config = Settings.model_validate({"provider": "atlascloud"})
    assert config.atlascloud_api_key == "environment-key"
    assert config.selected_model == "vendor/custom-model"
    assert isinstance(create_model(config), AtlasCloudModel)
    assert api._missing_provider_settings(config) == []
    selected = api._resolved_settings(config, provider="atlascloud", model="other/model")
    assert selected.selected_model == "other/model"
    assert selected.openai_model == config.openai_model
    assert selected.selected_reasoning_effort == ""
    assert Settings.model_validate({}).provider == "openai"
    assert api._missing_provider_settings(settings(atlascloud_api_key=" ")) == [
        "ATLASCLOUD_API_KEY"
    ]
    with pytest.raises(ModelError, match="ATLASCLOUD_API_KEY"):
        AtlasCloudModel(settings(atlascloud_api_key=" "))
    with pytest.raises(ModelError, match="model is required"):
        AtlasCloudModel(settings(atlascloud_model=" "))


@pytest.mark.parametrize("value", [-1, 1, 32768, 36383])
def test_invalid_context_windows_are_rejected(value) -> None:
    with pytest.raises(ValidationError):
        settings(atlascloud_context_window_tokens=value)


def test_cli_accepts_atlascloud_and_reports_its_missing_key(monkeypatch) -> None:
    monkeypatch.setattr("articraft.app.get_settings", lambda: settings(atlascloud_api_key=""))
    result = CliRunner().invoke(cli, ["generate", "--provider", "atlascloud", "test object"])
    assert result.exit_code != 0
    assert "ATLASCLOUD_API_KEY" in result.output
