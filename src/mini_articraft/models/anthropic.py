from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

from mini_articraft.errors import ModelError
from mini_articraft.settings import DEFAULT_ANTHROPIC_MODEL, Settings, get_settings

logger = logging.getLogger(__name__)

_RETRY_BASE_SECONDS = 0.5
_RETRY_MAX_SECONDS = 20.0
_SONNET_5_STANDARD_PRICING_START = date(2026, 9, 1)
_SUPPORTED_MODELS = {
    "claude-opus-4-8",
    "claude-sonnet-5",
}


@dataclass(frozen=True)
class _ModelSpec:
    context_window_tokens: int
    max_output_tokens: int
    input_price: float
    cache_write_5m_price: float
    cache_write_1h_price: float
    cache_read_price: float
    output_price: float


_OPUS_4_8_SPEC = _ModelSpec(
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
        self._client = client

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
        response = await self._send_with_retries(messages, tools)
        text = _response_text(response)
        tool_calls = _response_tool_calls(response)
        if not text and not tool_calls:
            raise ModelError("Anthropic response did not contain text or tool calls")

        token_usage = _response_token_usage(response)
        return {
            "text": text,
            "tool_calls": tool_calls,
            "token_usage": token_usage,
            "cost": _response_cost(self.config.anthropic_model, token_usage),
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

    async def _send_with_retries(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        for attempt in range(1, self.config.anthropic_max_attempts + 1):
            try:
                return await self._send(messages, tools)
            except Exception as exc:
                if attempt >= self.config.anthropic_max_attempts or not _should_retry(exc):
                    if isinstance(exc, ModelError):
                        raise
                    raise ModelError(f"Anthropic request failed: {_format_exception(exc)}") from exc
                delay = random.random() * min(
                    _RETRY_MAX_SECONDS,
                    _RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
                )
                logger.warning(
                    "Anthropic request failed (attempt %s/%s), retrying in %.2fs: %s",
                    attempt,
                    self.config.anthropic_max_attempts,
                    delay,
                    _format_exception(exc),
                )
                await asyncio.sleep(delay)
        raise AssertionError("retry loop did not return or raise")

    async def _send(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Any:
        request: dict[str, Any] = {
            "model": self.config.anthropic_model,
            "max_tokens": self.config.anthropic_max_output_tokens,
            "messages": _messages(messages),
        }
        system = _system(messages)
        if system:
            request["system"] = system
        converted_tools = _tools(tools or [])
        if converted_tools:
            request["tools"] = converted_tools

        async def send() -> Any:
            return await self._client_or_create().messages.create(**request)

        return await asyncio.wait_for(send(), timeout=self.config.anthropic_request_timeout_seconds)

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
            self._client = AsyncAnthropic(api_key=api_key)
        return self._client


def anthropic_api_key_value(settings: Settings) -> str:
    return (settings.anthropic_api_key or "").strip()


def _system(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        _message_text(message)
        for message in messages
        if "type" not in message and message.get("role") == "system"
    )


def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []
    for raw in messages:
        if raw.get("type") == "function_call_output":
            pending_tool_results.append(_tool_result_block(raw))
            continue

        if pending_tool_results:
            converted.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

        message = _message(raw)
        if message is not None:
            converted.append(message)

    if pending_tool_results:
        converted.append({"role": "user", "content": pending_tool_results})
    return converted


def _message(message: dict[str, Any]) -> dict[str, Any] | None:
    if message.get("role") == "system":
        return None
    if message.get("role") == "assistant":
        return _assistant_message(message)
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
    return {
        "type": "tool_result",
        "tool_use_id": str(message.get("call_id") or ""),
        "content": _tool_result_content(message.get("output")),
    }


def _tool_result_content(output: Any) -> str | list[dict[str, Any]]:
    if isinstance(output, list):
        content: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "input_text":
                content.append({"type": "text", "text": str(item.get("text") or "")})
            elif item.get("type") == "input_image":
                image = _image_content(item)
                if image is not None:
                    content.append(image)
        if content:
            return content
        return json.dumps(output)
    if isinstance(output, str):
        return output
    return json.dumps(output)


def _user_content(message: dict[str, Any]) -> str | list[dict[str, Any]]:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError("AnthropicModel message content must be a string or list")

    converted: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_text":
            converted.append({"type": "text", "text": str(item.get("text") or "")})
        elif item.get("type") == "input_image":
            image = _image_content(item)
            if image is not None:
                converted.append(image)
    return converted


def _image_content(item: dict[str, Any]) -> dict[str, Any] | None:
    image_url = str(item.get("image_url") or "")
    prefix = "data:"
    if not image_url.startswith(prefix) or ";base64," not in image_url:
        return None
    media_type, data = image_url[len(prefix) :].split(";base64,", 1)
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
        converted.append(
            {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "input_schema": _strip_unsupported_schema_keys(tool.get("parameters")),
            }
        )
    return converted


def _strip_unsupported_schema_keys(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            key: _strip_unsupported_schema_keys(value)
            for key, value in schema.items()
            if key != "strict"
        }
    if isinstance(schema, list):
        return [_strip_unsupported_schema_keys(item) for item in schema]
    return schema


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
        calls.append(
            {
                "id": str(_value(block, "id", "") or f"call_{uuid.uuid4().hex}"),
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
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cached_tokens": cache_read_input_tokens,
        "cached_input_tokens": cache_read_input_tokens,
        "total_tokens": (
            input_tokens
            + output_tokens
            + cache_creation_input_tokens
            + cache_read_input_tokens
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

    return round(
        (
            usage.get("input_tokens", 0) * spec.input_price
            + usage.get("cache_creation_input_tokens", 0) * spec.cache_write_5m_price
            + usage.get("cache_read_input_tokens", 0) * spec.cache_read_price
            + usage.get("output_tokens", 0) * spec.output_price
        )
        / 1_000_000,
        8,
    )


def _model_spec(model: str, *, today: date | None = None) -> _ModelSpec | None:
    if model == "claude-opus-4-8":
        return _OPUS_4_8_SPEC
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
    if model not in _SUPPORTED_MODELS:
        supported = ", ".join(sorted(_SUPPORTED_MODELS))
        raise ModelError(f"Unsupported Anthropic model: {model}. Supported models: {supported}")


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _http_status(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status", "status", "code"):
        value = getattr(exc, attr, None)
        try:
            if callable(value):
                value = value()
        except Exception:
            value = None
        if isinstance(value, int) and 100 <= value <= 599:
            return value

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) or getattr(response, "status", None)
    return status if isinstance(status, int) and 100 <= status <= 599 else None


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True

    status = _http_status(exc)
    if status is not None:
        if status in {408, 409, 425, 429} or status >= 500:
            return True
        if 400 <= status < 500:
            return False

    message = str(exc).lower()
    if any(
        token in message
        for token in (
            "overloaded",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
            "timed out",
            "rate limit",
            "too many requests",
            "connection reset",
            "connection aborted",
            "connection refused",
            "internal error",
        )
    ):
        return True
    if any(
        token in message
        for token in (
            "api key",
            "unauthorized",
            "permission denied",
            "forbidden",
            "invalid request",
            "not found",
        )
    ):
        return False
    return False


def _format_exception(exc: BaseException) -> str:
    status = _http_status(exc)
    message = str(exc).strip()
    summary = type(exc).__name__
    if status is not None:
        summary += f" (HTTP {status})"
    return f"{summary}: {message or repr(exc)}"


__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "AnthropicModel",
    "anthropic_api_key_value",
    "context_window_tokens_for",
    "max_output_tokens_for",
]
