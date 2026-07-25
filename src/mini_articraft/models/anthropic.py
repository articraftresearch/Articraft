from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from mini_articraft.errors import ModelError
from mini_articraft.settings import DEFAULT_ANTHROPIC_MODEL, Settings, get_settings

_MAX_BASE64_IMAGE_BYTES = 10 * 1024 * 1024
_SONNET_5_STANDARD_PRICING_START = date(2026, 9, 1)
SUPPORTED_MODELS = (
    "claude-opus-5",
    "claude-sonnet-5",
)


@dataclass(frozen=True)
class _ModelSpec:
    context_window_tokens: int
    max_output_tokens: int
    input_price: float
    cache_write_5m_price: float
    cache_write_1h_price: float
    cache_read_price: float
    output_price: float


_OPUS_5_SPEC = _ModelSpec(
    context_window_tokens=1_000_000,
    max_output_tokens=128_000,
    input_price=5.00,
    cache_write_5m_price=6.25,
    cache_write_1h_price=10.00,
    cache_read_price=0.50,
    output_price=25.00,
)
_SONNET_5_INTRO_SPEC = _ModelSpec(
    context_window_tokens=1_000_000,
    max_output_tokens=128_000,
    input_price=2.00,
    cache_write_5m_price=2.50,
    cache_write_1h_price=4.00,
    cache_read_price=0.20,
    output_price=10.00,
)
_SONNET_5_STANDARD_SPEC = _ModelSpec(
    context_window_tokens=1_000_000,
    max_output_tokens=128_000,
    input_price=3.00,
    cache_write_5m_price=3.75,
    cache_write_1h_price=6.00,
    cache_read_price=0.30,
    output_price=15.00,
)


class AnthropicModel:
    def __init__(self, settings: Settings | None = None, *, client: Any | None = None):
        self.config = settings or get_settings()
        _raise_for_unsupported_model(self.config.anthropic_model)
        _raise_for_invalid_max_output_tokens(
            self.config.anthropic_model,
            self.config.anthropic_max_output_tokens,
        )
        self._client = client
        self._history: list[dict[str, Any]] = []
        self._last_message_count = 0

    @property
    def context_window_tokens(self) -> int:
        return context_window_tokens_for(self.config.anthropic_model) or 0

    async def query(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Query Anthropic and return the response shape used by the agent."""
        request_messages = self._request_messages(messages)
        try:
            response = await self._send(messages, request_messages, tools)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Anthropic request failed: {_format_exception(exc)}") from exc

        text = _response_text(response)
        tool_calls = _response_tool_calls(response)
        response_content = _response_blocks(response)
        if not text and not tool_calls and not _has_thinking(response_content):
            raise ModelError("Anthropic response did not contain text, thinking, or tool calls")

        token_usage = _response_token_usage(response)
        self._history = [
            *request_messages,
            {"role": "assistant", "content": response_content},
        ]
        self._last_message_count = len(messages)
        return {
            "text": text,
            "tool_calls": tool_calls,
            "token_usage": token_usage,
            "cost": _response_cost(self.config.anthropic_model, token_usage),
            "provider_content": [_record_block(block) for block in response_content],
            "response": response,
        }

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result

    async def _send(
        self,
        source_messages: list[dict[str, Any]],
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self.config.anthropic_model,
            "max_tokens": self.config.anthropic_max_output_tokens,
            "messages": request_messages,
            "cache_control": {"type": "ephemeral"},
        }
        system = _system(source_messages)
        if system:
            request["system"] = system
        converted_tools = _tools(tools or [])
        if converted_tools:
            request["tools"] = converted_tools

        return await self._client_or_create().messages.create(**request)

    def _request_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self._history:
            return _messages(messages)

        new_messages = _messages(
            messages[self._last_message_count :],
            include_assistant=False,
        )
        return [*self._history, *new_messages]

    def _client_or_create(self) -> Any:
        if self._client is None:
            api_key = anthropic_api_key_value(self.config)
            if not api_key:
                raise ModelError("Anthropic credentials are required. Set ANTHROPIC_API_KEY.")
            try:
                from anthropic import AsyncAnthropic  # type: ignore
            except Exception as exc:
                raise ModelError(
                    "Anthropic provider selected but the `anthropic` package is not installed."
                ) from exc
            self._client = AsyncAnthropic(
                api_key=api_key,
                max_retries=self.config.anthropic_max_attempts - 1,
                timeout=self.config.anthropic_request_timeout_seconds,
            )
        return self._client


def anthropic_api_key_value(settings: Settings) -> str:
    return (settings.anthropic_api_key or "").strip()


def _system(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        _message_text(message)
        for message in messages
        if "type" not in message and message.get("role") == "system"
    )


def _messages(
    messages: list[dict[str, Any]],
    *,
    include_assistant: bool = True,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []
    for raw in messages:
        if raw.get("type") == "function_call_output":
            pending_tool_results.append(_tool_result_block(raw))
            continue

        if pending_tool_results:
            converted.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

        message = _message(raw, include_assistant=include_assistant)
        if message is not None:
            converted.append(message)

    if pending_tool_results:
        converted.append({"role": "user", "content": pending_tool_results})
    return converted


def _message(
    message: dict[str, Any],
    *,
    include_assistant: bool,
) -> dict[str, Any] | None:
    if message.get("role") == "system":
        return None
    if message.get("role") == "assistant":
        return _assistant_message(message) if include_assistant else None
    if message.get("role") == "user":
        return {"role": "user", "content": _user_content(message)}
    return None


def _assistant_message(message: dict[str, Any]) -> dict[str, Any] | None:
    content: list[dict[str, Any]] = []
    text = _message_text(message)
    if text:
        content.append({"type": "text", "text": text})

    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        content.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or ""),
                "name": str(call.get("name") or ""),
                "input": _arguments(call),
            }
        )

    if not content:
        return None
    return {"role": "assistant", "content": content}


def _tool_result_block(message: dict[str, Any]) -> dict[str, Any]:
    output = message.get("output")
    block = {
        "type": "tool_result",
        "tool_use_id": str(message.get("call_id") or ""),
        "content": _tool_result_content(output),
    }
    if _tool_result_is_error(output):
        block["is_error"] = True
    return block


def _tool_result_content(output: Any) -> str | list[dict[str, Any]]:
    if isinstance(output, list):
        content = _content_items(output)
        if content:
            return content
        return json.dumps(output)
    if isinstance(output, str):
        return output
    return json.dumps(output)


def _tool_result_is_error(output: Any) -> bool:
    if isinstance(output, list):
        output = next(
            (
                item.get("text")
                for item in output
                if isinstance(item, dict) and item.get("type") == "input_text"
            ),
            None,
        )
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return False
    return isinstance(output, dict) and "error" in output


def _user_content(message: dict[str, Any]) -> str | list[dict[str, Any]]:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError("AnthropicModel message content must be a string or list")
    return _content_items(content)


def _content_items(items: list[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_text":
            converted.append({"type": "text", "text": str(item.get("text") or "")})
        elif item.get("type") == "input_image":
            converted.append(_image_content(item))
    return converted


def _image_content(item: dict[str, Any]) -> dict[str, Any]:
    image_url = str(item.get("image_url") or "")
    prefix = "data:"
    if image_url.startswith(("https://", "http://")):
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": image_url,
            },
        }
    if not image_url.startswith(prefix) or ";base64," not in image_url:
        raise ModelError("Anthropic image input must use a data URL or an HTTP URL")

    media_type, data = image_url[len(prefix) :].split(";base64,", 1)
    if media_type not in {"image/gif", "image/jpeg", "image/png", "image/webp"}:
        raise ModelError(f"Anthropic does not support image type: {media_type or 'missing'}")
    if not data:
        raise ModelError("Anthropic image input has no base64 data")
    try:
        data.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ModelError("Anthropic image input must contain ASCII base64 data") from exc
    if len(data) > _MAX_BASE64_IMAGE_BYTES:
        raise ModelError("Anthropic image input exceeds the 10 MB base64 limit")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments") or "{}"
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
    return {"_raw": raw}


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if not isinstance(content, str):
        raise TypeError("AnthropicModel messages must use string content")
    return content


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        converted_tool = {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "input_schema": tool.get("parameters"),
        }
        if isinstance(tool.get("strict"), bool):
            converted_tool["strict"] = tool["strict"]
        converted.append(converted_tool)
    return converted


def _response_text(response: Any) -> str:
    return "".join(
        str(_value(block, "text"))
        for block in _response_blocks(response)
        if _block_type(block) == "text" and _value(block, "text", None)
    )


def _response_tool_calls(response: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in _response_blocks(response):
        if _block_type(block) != "tool_use":
            continue
        tool_input = _value(block, "input", {})
        call_id = str(_value(block, "id", ""))
        if not call_id:
            raise ModelError("Anthropic tool call did not include an id")
        calls.append(
            {
                "id": call_id,
                "name": str(_value(block, "name", "") or ""),
                "arguments": json.dumps(tool_input)
                if isinstance(tool_input, (dict, list))
                else str(tool_input or "{}"),
            }
        )
    return calls


def _response_blocks(response: Any) -> list[Any]:
    content = _value(response, "content", [])
    if isinstance(content, list):
        return content
    try:
        return list(content)
    except TypeError:
        return [content]


def _record_block(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    model_dump = getattr(block, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")
        if isinstance(value, dict):
            return value
    raise ModelError(f"Anthropic returned an unsupported content block: {type(block).__name__}")


def _has_thinking(content: list[Any]) -> bool:
    return any(_block_type(block) in {"thinking", "redacted_thinking"} for block in content)


def _block_type(block: Any) -> str:
    return str(_value(block, "type", ""))


def _response_token_usage(response: Any) -> dict[str, int]:
    usage = _value(response, "usage", None)
    if usage is None:
        return {}

    input_tokens = _usage_int(usage, "input_tokens")
    output_tokens = _usage_int(usage, "output_tokens")
    cache_creation_input_tokens = _usage_int(usage, "cache_creation_input_tokens")
    cache_read_input_tokens = _usage_int(usage, "cache_read_input_tokens")
    cache_creation = _value(usage, "cache_creation", None)
    cache_creation_5m_input_tokens = _usage_int(
        cache_creation,
        "ephemeral_5m_input_tokens",
    )
    cache_creation_1h_input_tokens = _usage_int(
        cache_creation,
        "ephemeral_1h_input_tokens",
    )
    detailed_cache_creation_tokens = cache_creation_5m_input_tokens + cache_creation_1h_input_tokens
    if detailed_cache_creation_tokens:
        cache_creation_input_tokens = detailed_cache_creation_tokens
    else:
        cache_creation_5m_input_tokens = cache_creation_input_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_creation_5m_input_tokens": cache_creation_5m_input_tokens,
        "cache_creation_1h_input_tokens": cache_creation_1h_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cached_tokens": cache_read_input_tokens,
        "cached_input_tokens": cache_read_input_tokens,
        "total_tokens": (
            input_tokens + output_tokens + cache_creation_input_tokens + cache_read_input_tokens
        ),
    }


def _usage_int(usage: Any, name: str) -> int:
    value = _value(usage, name, None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _response_cost(model: str, usage: dict[str, int], *, today: date | None = None) -> float:
    spec = _model_spec(model, today=today)
    if spec is None or not usage:
        return 0.0

    cache_creation_5m_input_tokens = usage.get("cache_creation_5m_input_tokens", 0)
    cache_creation_1h_input_tokens = usage.get("cache_creation_1h_input_tokens", 0)
    if not cache_creation_5m_input_tokens and not cache_creation_1h_input_tokens:
        cache_creation_5m_input_tokens = usage.get("cache_creation_input_tokens", 0)

    return round(
        (
            usage.get("input_tokens", 0) * spec.input_price
            + cache_creation_5m_input_tokens * spec.cache_write_5m_price
            + cache_creation_1h_input_tokens * spec.cache_write_1h_price
            + usage.get("cache_read_input_tokens", 0) * spec.cache_read_price
            + usage.get("output_tokens", 0) * spec.output_price
        )
        / 1_000_000,
        8,
    )


def _model_spec(model: str, *, today: date | None = None) -> _ModelSpec | None:
    if model == "claude-opus-5":
        return _OPUS_5_SPEC
    if model == "claude-sonnet-5":
        current_date = today or date.today()
        if current_date >= _SONNET_5_STANDARD_PRICING_START:
            return _SONNET_5_STANDARD_SPEC
        return _SONNET_5_INTRO_SPEC
    return None


def context_window_tokens_for(model: str) -> int | None:
    spec = _model_spec(model)
    return spec.context_window_tokens if spec is not None else None


def max_output_tokens_for(model: str) -> int | None:
    spec = _model_spec(model)
    return spec.max_output_tokens if spec is not None else None


def _raise_for_unsupported_model(model: str) -> None:
    if model not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ModelError(f"Unsupported Anthropic model: {model}. Supported models: {supported}")


def _raise_for_invalid_max_output_tokens(model: str, max_output_tokens: int) -> None:
    supported_max = max_output_tokens_for(model)
    if supported_max is not None and max_output_tokens > supported_max:
        raise ModelError(f"Anthropic max output tokens for {model} cannot exceed {supported_max}")


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _format_exception(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    message = str(exc).strip()
    summary = type(exc).__name__
    if isinstance(status, int):
        summary += f" (HTTP {status})"
    return f"{summary}: {message or repr(exc)}"


__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "SUPPORTED_MODELS",
    "AnthropicModel",
    "anthropic_api_key_value",
    "context_window_tokens_for",
    "max_output_tokens_for",
]
