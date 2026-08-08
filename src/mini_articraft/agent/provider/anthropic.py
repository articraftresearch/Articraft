from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from mini_articraft.errors import ModelError
from mini_articraft.settings import DEFAULT_ANTHROPIC_MODEL, Settings, get_settings

# Claude 5 has a 1M API window; use the same conservative working budget as Codex.
_AGENT_CONTEXT_WINDOW_TOKENS = 272_000
_MAX_OUTPUT_TOKENS = 128_000
_MAX_IMAGE_DATA_LENGTH = 10 * 1024 * 1024
_COMPACTION_BETA = "compact-2026-01-12"
_COMPACTION_TRIGGER_TOKENS = 50_000
_CACHE_WRITE_5M_MULTIPLIER = 1.25
_CACHE_WRITE_1H_MULTIPLIER = 2.0
_CACHE_READ_MULTIPLIER = 0.1
_SONNET_5_STANDARD_PRICING_START = date(2026, 9, 1)


@dataclass(frozen=True)
class _Prices:
    input_price: float
    output_price: float


_MODEL_PRICES = {
    "claude-fable-5": _Prices(input_price=10.00, output_price=50.00),
    "claude-mythos-5": _Prices(input_price=10.00, output_price=50.00),
    "claude-opus-5": _Prices(input_price=5.00, output_price=25.00),
    "claude-sonnet-5": _Prices(input_price=2.00, output_price=10.00),
}
_SONNET_5_STANDARD_PRICES = _Prices(
    input_price=3.00,
    output_price=15.00,
)
KNOWN_MODELS = tuple(_MODEL_PRICES)


class AnthropicModel:
    def __init__(self, settings: Settings | None = None, *, client: Any | None = None):
        self.config = settings or get_settings()
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
        try:
            response = await self._send(messages, tools)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Anthropic request failed: {_format_exception(exc)}") from exc

        response_content = list(_value(response, "content", []) or [])
        text = _response_text(response_content)
        tool_calls = _response_tool_calls(response_content)
        if not text and not tool_calls and not _has_thinking(response_content):
            raise ModelError("Anthropic response did not contain text, thinking, or tool calls")

        token_usage = _response_token_usage(response)
        return {
            "text": text,
            "tool_calls": tool_calls,
            "token_usage": token_usage,
            "cost": _response_cost(self.config.anthropic_model, token_usage),
            "provider_content": [_record_block(block) for block in response_content],
            "response": response,
        }

    async def summarize_context(
        self,
        messages: list[dict[str, Any]],
        *,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        """Create one plain checkpoint with Anthropic's native compaction."""
        system = _system(messages)
        edit: dict[str, Any] = {
            "type": "compact_20260112",
            "trigger": {
                "type": "input_tokens",
                "value": _COMPACTION_TRIGGER_TOKENS,
            },
            "pause_after_compaction": True,
        }
        if system:
            # Native compaction uses its own instructions instead of the normal
            # system prompt. Keep the system prompt too so short inputs that do
            # not reach the 50k minimum still produce the same plain summary.
            edit["instructions"] = system
        request: dict[str, Any] = {
            "model": self.config.anthropic_model,
            "max_tokens": max_output_tokens,
            "messages": _messages(messages),
            "betas": [_COMPACTION_BETA],
            "context_management": {"edits": [edit]},
        }
        if system:
            request["system"] = system
        try:
            response = await self._client_or_create().beta.messages.create(**request)
        except Exception as exc:
            raise ModelError(f"Anthropic summary request failed: {_format_exception(exc)}") from exc

        response_content = list(_value(response, "content", []) or [])
        text = _compaction_text(response_content) or _response_text(response_content)
        if not text:
            raise ModelError("Anthropic summary response did not contain a checkpoint")
        token_usage = _response_token_usage(response)
        return {
            "text": text,
            "token_usage": token_usage,
            "cost": _response_cost(self.config.anthropic_model, token_usage),
        }

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    async def _send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self.config.anthropic_model,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "messages": _messages(messages),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
        system = _system(messages)
        if system:
            request["system"] = system
        converted_tools = _tools(tools or [])
        if converted_tools:
            request["tools"] = converted_tools

        return await self._client_or_create().messages.create(**request)

    def _client_or_create(self) -> Any:
        if self._client is None:
            api_key = anthropic_api_key_value(self.config)
            if not api_key:
                raise ModelError("Anthropic credentials are required. Set ANTHROPIC_API_KEY.")
            from anthropic import AsyncAnthropic

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


def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if tool_results:
            converted.append({"role": "user", "content": list(tool_results)})
            tool_results.clear()

    for message in messages:
        if message.get("type") == "function_call_output":
            tool_results.append(_tool_result_block(message))
            continue

        flush_tool_results()
        if message.get("role") == "user":
            converted.append({"role": "user", "content": _user_content(message)})
        elif message.get("role") == "assistant":
            converted.append({"role": "assistant", "content": _assistant_content(message)})
    flush_tool_results()
    return converted


def _assistant_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    provider_content = message.get("provider_content")
    if not isinstance(provider_content, list):
        raise TypeError("Anthropic assistant messages require provider_content")
    return [dict(block) for block in provider_content if isinstance(block, dict)]


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
        return content or json.dumps(output)
    return str(output or "")


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
        if item.get("type") == "input_text":
            converted.append({"type": "text", "text": str(item.get("text") or "")})
        elif item.get("type") == "input_image":
            converted.append(_image_content(item))
    return converted


def _image_content(item: dict[str, Any]) -> dict[str, Any]:
    image_url = str(item.get("image_url") or "")
    prefix = "data:"
    if not image_url.startswith(prefix) or ";base64," not in image_url:
        raise ModelError("Anthropic image input must use a base64 data URL")

    media_type, data = image_url[len(prefix) :].split(";base64,", 1)
    if media_type not in {"image/gif", "image/jpeg", "image/png", "image/webp"}:
        raise ModelError(f"Anthropic does not support image type: {media_type or 'missing'}")
    if not data:
        raise ModelError("Anthropic image input has no base64 data")
    try:
        data.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ModelError("Anthropic image input must contain ASCII base64 data") from exc
    if len(data) > _MAX_IMAGE_DATA_LENGTH:
        raise ModelError("Anthropic image input exceeds the 10 MB base64 limit")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if not isinstance(content, str):
        raise TypeError("AnthropicModel messages must use string content")
    return content


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "input_schema": tool.get("parameters"),
            "strict": bool(tool.get("strict", False)),
        }
        for tool in tools
        if tool.get("type") == "function"
    ]


def _response_text(content: list[Any]) -> str:
    return "".join(
        str(_value(block, "text"))
        for block in content
        if _block_type(block) == "text" and _value(block, "text", None)
    )


def _compaction_text(content: list[Any]) -> str:
    for block in reversed(content):
        block_content = _value(block, "content", None)
        if _block_type(block) == "compaction" and block_content:
            return str(block_content)
    return ""


def _response_tool_calls(content: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in content:
        if _block_type(block) != "tool_use":
            continue
        tool_input = _value(block, "input", {})
        call_id = str(_value(block, "id", ""))
        name = str(_value(block, "name", ""))
        if not call_id or not name:
            raise ModelError("Anthropic tool call did not include an id or name")
        calls.append(
            {
                "id": call_id,
                "name": name,
                "arguments": json.dumps(tool_input),
            }
        )
    return calls


def _record_block(block: Any) -> dict[str, Any]:
    return (
        dict(block) if isinstance(block, dict) else block.model_dump(mode="json", exclude_none=True)
    )


def _has_thinking(content: list[Any]) -> bool:
    return any(_block_type(block) in {"thinking", "redacted_thinking"} for block in content)


def _block_type(block: Any) -> str:
    return str(_value(block, "type", ""))


def _response_token_usage(response: Any) -> dict[str, int]:
    usage = _value(response, "usage", None)
    if usage is None:
        return {}

    iterations = _value(usage, "iterations", None)
    rows = iterations if isinstance(iterations, list) and iterations else [usage]
    input_tokens = 0
    output_tokens = 0
    cache_creation_input_tokens = 0
    cache_creation_5m_input_tokens = 0
    cache_creation_1h_input_tokens = 0
    cache_read_input_tokens = 0
    for row in rows:
        input_tokens += _usage_int(row, "input_tokens")
        output_tokens += _usage_int(row, "output_tokens")
        row_cache_creation_input_tokens = _usage_int(row, "cache_creation_input_tokens")
        cache_creation = _value(row, "cache_creation", None)
        row_cache_creation_5m_input_tokens = _usage_int(
            cache_creation,
            "ephemeral_5m_input_tokens",
        )
        row_cache_creation_1h_input_tokens = _usage_int(
            cache_creation,
            "ephemeral_1h_input_tokens",
        )
        detailed_cache_creation_tokens = (
            row_cache_creation_5m_input_tokens + row_cache_creation_1h_input_tokens
        )
        if detailed_cache_creation_tokens:
            row_cache_creation_input_tokens = detailed_cache_creation_tokens
        else:
            row_cache_creation_5m_input_tokens = row_cache_creation_input_tokens
        cache_creation_input_tokens += row_cache_creation_input_tokens
        cache_creation_5m_input_tokens += row_cache_creation_5m_input_tokens
        cache_creation_1h_input_tokens += row_cache_creation_1h_input_tokens
        cache_read_input_tokens += _usage_int(row, "cache_read_input_tokens")

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_creation_5m_input_tokens": cache_creation_5m_input_tokens,
        "cache_creation_1h_input_tokens": cache_creation_1h_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
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
    prices = _prices_for(model, today=today)
    if prices is None or not usage:
        return 0.0

    cache_creation_5m_input_tokens = usage.get("cache_creation_5m_input_tokens", 0)
    cache_creation_1h_input_tokens = usage.get("cache_creation_1h_input_tokens", 0)
    if not cache_creation_5m_input_tokens and not cache_creation_1h_input_tokens:
        cache_creation_5m_input_tokens = usage.get("cache_creation_input_tokens", 0)

    return round(
        (
            usage.get("input_tokens", 0) * prices.input_price
            + cache_creation_5m_input_tokens * prices.input_price * _CACHE_WRITE_5M_MULTIPLIER
            + cache_creation_1h_input_tokens * prices.input_price * _CACHE_WRITE_1H_MULTIPLIER
            + usage.get("cache_read_input_tokens", 0) * prices.input_price * _CACHE_READ_MULTIPLIER
            + usage.get("output_tokens", 0) * prices.output_price
        )
        / 1_000_000,
        8,
    )


def _prices_for(model: str, *, today: date | None = None) -> _Prices | None:
    if model == "claude-sonnet-5" and (today or date.today()) >= _SONNET_5_STANDARD_PRICING_START:
        return _SONNET_5_STANDARD_PRICES
    return _MODEL_PRICES.get(model)


def context_window_tokens_for(model: str) -> int | None:
    return _AGENT_CONTEXT_WINDOW_TOKENS if model in _MODEL_PRICES else None


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _format_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message or repr(exc)}"


__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "KNOWN_MODELS",
    "AnthropicModel",
    "anthropic_api_key_value",
    "context_window_tokens_for",
]
