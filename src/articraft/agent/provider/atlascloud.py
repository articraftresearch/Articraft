from __future__ import annotations

import json
from typing import Any

import httpx

from articraft.errors import ModelError
from articraft.settings import Settings, get_settings

_API_URL = "https://api.atlascloud.ai/v1/chat/completions"


class AtlasCloudModel:
    supports_images = False

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        self.config = settings or get_settings()
        if not (self.config.atlascloud_api_key or "").strip():
            raise ModelError("Atlas Cloud credentials are required. Set ATLASCLOUD_API_KEY.")
        if not self.config.atlascloud_model.strip():
            raise ModelError("Atlas Cloud model is required. Set ARTICRAFT_ATLASCLOUD_MODEL.")
        self._client = client

    @property
    def context_window_tokens(self) -> int:
        return self.config.atlascloud_context_window_tokens

    async def query(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await self._complete(messages, tools=tools)

    async def summarize_context(
        self,
        messages: list[dict[str, Any]],
        *,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        result = await self._complete(messages, max_output_tokens=max_output_tokens)
        if not result["text"]:
            raise ModelError("Atlas Cloud summary response did not contain text")
        return {key: result[key] for key in ("text", "token_usage", "cost")}

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.atlascloud_model,
            "messages": _messages(messages),
            "max_tokens": min(
                max_output_tokens or self.config.atlascloud_max_output_tokens,
                self.config.atlascloud_max_output_tokens,
            ),
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        key: tool[key]
                        for key in ("name", "description", "parameters", "strict")
                        if key in tool
                    },
                }
                for tool in tools
                if tool.get("type") == "function"
            ]
        if self._client is None:
            self._client = httpx.AsyncClient()
        # A transport failure can follow a billable generation; do not resubmit it.
        try:
            response = await self._client.post(
                _API_URL,
                headers={
                    "Authorization": f"Bearer {(self.config.atlascloud_api_key or '').strip()}"
                },
                json=request,
                timeout=self.config.atlascloud_request_timeout_seconds,
            )
        except httpx.TransportError as exc:
            raise ModelError(f"Atlas Cloud request failed: {type(exc).__name__}") from exc
        if response.is_error:
            raise ModelError(f"Atlas Cloud request failed (HTTP {response.status_code})")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelError("Atlas Cloud response was not valid JSON") from exc
        if not isinstance(payload, dict) or payload.get("error"):
            raise ModelError("Atlas Cloud returned an error or invalid response")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelError("Atlas Cloud response did not contain a valid choice")
        choice = choices[0]
        if choice.get("error") or choice.get("finish_reason") in {"error", "length"}:
            raise ModelError("Atlas Cloud completion failed or reached its output limit")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ModelError("Atlas Cloud response did not contain an assistant message")
        text = message.get("content") or ""
        calls = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ModelError("Atlas Cloud returned an invalid function call")
            function = call["function"]
            if not call.get("id") or not function.get("name"):
                raise ModelError("Atlas Cloud returned a function call without an id or name")
            calls.append(
                {
                    "id": call["id"],
                    "name": function["name"],
                    "arguments": function.get("arguments", "{}"),
                }
            )
        if not isinstance(text, str) or (not text and not calls):
            raise ModelError("Atlas Cloud response did not contain text or tool calls")
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            raise ModelError("Atlas Cloud returned invalid token usage")
        details = usage.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            raise ModelError("Atlas Cloud returned invalid cached token usage")
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        reasoning = message.get("reasoning_content")
        return {
            "text": text,
            "tool_calls": calls,
            "provider_content": (
                [{"type": "atlascloud_reasoning", "reasoning_content": reasoning}]
                if isinstance(reasoning, str)
                else []
            ),
            "token_usage": {
                "input_tokens": prompt_tokens,
                "cached_input_tokens": int(details.get("cached_tokens") or 0),
                "output_tokens": completion_tokens,
                "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
            },
            # Token counts are available, but this adapter does not maintain a price catalog.
            "cost": 0.0,
            "response": payload,
        }


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ModelError("Atlas Cloud message content must be text")
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "input_image":
            raise ModelError("Atlas Cloud adapter supports text and function calling, not images")
        if part.get("type") == "input_text":
            parts.append(str(part.get("text") or ""))
    return "\n".join(parts)


def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        if message.get("type") == "function_call_output":
            output = message.get("output")
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("call_id") or ""),
                    "content": _text(output)
                    if isinstance(output, (str, list))
                    else json.dumps(output),
                }
            )
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        converted: dict[str, Any] = {"role": role, "content": _text(message.get("content", ""))}
        if role == "assistant":
            calls = []
            for call in message.get("tool_calls") or []:
                arguments = call.get("arguments", "{}")
                calls.append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments),
                        },
                    }
                )
            if calls:
                converted["tool_calls"] = calls
                converted["content"] = converted["content"] or None
            for item in message.get("provider_content") or []:
                if item.get("type") == "atlascloud_reasoning":
                    converted["reasoning_content"] = item["reasoning_content"]
        result.append(converted)
    return result
