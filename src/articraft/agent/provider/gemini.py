from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from articraft.errors import ModelError
from articraft.settings import DEFAULT_GEMINI_MODEL, Settings, get_settings

_LONG_CONTEXT_THRESHOLD_TOKENS = 200_000
# These models have a 1M API window; use the same conservative working budget as Codex.
_AGENT_CONTEXT_WINDOW_TOKENS = 272_000


@dataclass(frozen=True)
class _ModelSpec:
    context_window_tokens: int
    input_price: float
    cached_input_price: float
    output_price: float
    long_context_prices: tuple[float, float, float] | None = None


# Prices are USD per million tokens for the Gemini Developer API Standard tier.
_MODELS = {
    "gemini-3.6-flash": _ModelSpec(_AGENT_CONTEXT_WINDOW_TOKENS, 1.50, 0.15, 7.50),
    "gemini-3.5-flash-lite": _ModelSpec(_AGENT_CONTEXT_WINDOW_TOKENS, 0.30, 0.03, 2.50),
}
KNOWN_MODELS = tuple(sorted(_MODELS))


class GeminiModel:
    def __init__(self, settings: Settings | None = None, *, client: Any | None = None):
        self.config = settings or get_settings()
        self._client = client

    @property
    def context_window_tokens(self) -> int:
        return context_window_tokens_for(self.config.gemini_model) or 0

    async def query(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Query Gemini and return the response shape used by the agent."""
        try:
            response = await self._client_or_create().aio.interactions.create(
                model=self.config.gemini_model,
                input=_input_steps(messages),
                system_instruction=_instructions(messages) or None,
                tools=_tools(tools or []) or None,
                store=False,
            )
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Gemini request failed: {_format_exception(exc)}") from exc

        response_steps = _response_steps(response)
        text = _response_text(response_steps)
        tool_calls = _response_tool_calls(response_steps)
        _raise_for_bad_status(response, tool_calls)
        if not text and not tool_calls and not _response_has_thoughts(response_steps):
            raise ModelError("Gemini response did not contain text, thoughts, or tool calls")

        token_usage = _response_token_usage(response)
        return {
            "text": text,
            "tool_calls": tool_calls,
            "token_usage": token_usage,
            "cost": _response_cost(self.config.gemini_model, token_usage),
            "provider_content": response_steps,
            "response": response,
        }

    async def summarize_context(
        self,
        messages: list[dict[str, Any]],
        *,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        """Create one context summary."""
        try:
            response = await self._client_or_create().aio.interactions.create(
                model=self.config.gemini_model,
                input=_input_steps(messages),
                system_instruction=_instructions(messages) or None,
                tools=None,
                generation_config={"max_output_tokens": max_output_tokens},
                store=False,
            )
        except Exception as exc:
            raise ModelError(f"Gemini summary request failed: {_format_exception(exc)}") from exc

        response_steps = _response_steps(response)
        text = _response_text(response_steps)
        _raise_for_bad_status(response, [])
        if not text:
            raise ModelError("Gemini summary response did not contain text")
        token_usage = _response_token_usage(response)
        return {
            "text": text,
            "token_usage": token_usage,
            "cost": _response_cost(self.config.gemini_model, token_usage),
        }

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            close = getattr(getattr(client, "aio", None), "aclose", None)
            if close is not None:
                await close()

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = (self.config.gemini_api_key or "").strip()
        if not api_key:
            raise ModelError("Gemini credentials are required. Set GEMINI_API_KEY.")

        from google import genai  # type: ignore

        self._client = genai.Client(
            api_key=api_key,
            http_options={
                "timeout": max(1, round(self.config.gemini_request_timeout_seconds * 1_000)),
                "retry_options": {"attempts": self.config.gemini_max_attempts},
            },
        )
        return self._client


def _instructions(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        _message_text(message)
        for message in messages
        if "type" not in message and message.get("role") == "system"
    )


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if not isinstance(content, str):
        raise TypeError("GeminiModel messages must use string content")
    return content


def _input_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        if message.get("type") == "function_call_output":
            call_id = str(message.get("call_id") or "")
            name = tool_names.get(call_id)
            if not name:
                raise ModelError(f"Gemini tool result has unknown call id: {call_id}")
            output = message.get("output")
            result = {
                "type": "function_result",
                "name": name,
                "call_id": call_id,
                "result": _function_result_content(output),
            }
            if _function_result_is_error(output):
                result["is_error"] = True
            steps.append(result)
        elif message.get("role") == "user":
            steps.append({"type": "user_input", "content": _user_content(message)})
        elif message.get("role") == "assistant":
            assistant_steps = _assistant_steps(message)
            steps.extend(assistant_steps)
            for step in assistant_steps:
                if step.get("type") == "function_call":
                    tool_names[str(step.get("id") or "")] = str(step.get("name") or "")
    return steps


def _assistant_steps(message: dict[str, Any]) -> list[dict[str, Any]]:
    provider_content = message.get("provider_content")
    if not isinstance(provider_content, list):
        raise TypeError("Gemini assistant messages require provider_content")
    return [dict(step) for step in provider_content if isinstance(step, dict)]


def _user_content(message: dict[str, Any]) -> list[dict[str, str]]:
    content = message.get("content", "")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise TypeError("GeminiModel message content must be a string or list")

    converted: list[dict[str, str]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_text":
            converted.append({"type": "text", "text": str(item.get("text") or "")})
        elif item.get("type") == "input_image":
            image = _image_content(
                str(item.get("image_url") or ""),
                str(item.get("detail") or "high"),
            )
            if image is not None:
                converted.append(image)
    return converted


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": _strip_unsupported_schema_keys(tool.get("parameters") or {}),
        }
        for tool in tools
        if tool.get("type") == "function"
    ]


def _strip_unsupported_schema_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_unsupported_schema_keys(child)
            for key, child in value.items()
            if key not in {"additionalProperties", "strict"}
        }
    if isinstance(value, list):
        return [_strip_unsupported_schema_keys(child) for child in value]
    return value


def _function_result_content(output: Any) -> list[dict[str, str]]:
    if not isinstance(output, list):
        return [{"type": "text", "text": _result_text(output)}]

    content: list[dict[str, str]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_text":
            content.append({"type": "text", "text": str(item.get("text") or "")})
        elif item.get("type") == "input_image":
            image = _image_content(
                str(item.get("image_url") or ""),
                str(item.get("detail") or "high"),
            )
            if image is not None:
                content.append(image)
    return content or [{"type": "text", "text": json.dumps(output)}]


def _result_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output)


def _function_result_is_error(output: Any) -> bool:
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


def _image_content(image_url: str, detail: str) -> dict[str, str] | None:
    header, separator, data = image_url.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        return None
    return {
        "type": "image",
        "mime_type": header.removeprefix("data:").removesuffix(";base64"),
        "data": data,
        "resolution": "ultra_high" if detail == "original" else "high",
    }


def _response_steps(response: Any) -> list[dict[str, Any]]:
    steps = getattr(response, "steps", None)
    if not steps:
        return []
    return [_record_step(step) for step in steps]


def _record_step(step: Any) -> dict[str, Any]:
    if isinstance(step, dict):
        return dict(step)
    if getattr(step, "is_unknown", False) and isinstance(getattr(step, "raw", None), dict):
        return dict(step.raw)
    return step.model_dump(mode="json", by_alias=True, exclude_none=True)


def _response_text(steps: list[Any]) -> str:
    parts: list[str] = []
    for step in steps:
        if _value(step, "type") != "model_output":
            continue
        for content in _value(step, "content") or []:
            text = _value(content, "text")
            if _value(content, "type") == "text" and isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _response_tool_calls(steps: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for step in steps:
        if _value(step, "type") != "function_call":
            continue
        call_id = str(_value(step, "id") or "")
        name = str(_value(step, "name") or "")
        if not call_id or not name:
            raise ModelError("Gemini returned a function call without an id or name")
        arguments = _value(step, "arguments")
        calls.append(
            {
                "id": call_id,
                "name": name,
                "arguments": json.dumps(arguments if isinstance(arguments, dict) else {}),
            }
        )
    return calls


def _response_has_thoughts(steps: list[Any]) -> bool:
    return any(_value(step, "type") == "thought" for step in steps)


def _response_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    input_tokens = _usage_int(usage, "total_input_tokens")
    cached_tokens = _usage_int(usage, "total_cached_tokens")
    output_tokens = _usage_int(usage, "total_output_tokens")
    thought_tokens = _usage_int(usage, "total_thought_tokens")
    total_tokens = (
        _usage_int(usage, "total_tokens") or input_tokens + output_tokens + thought_tokens
    )
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "thought_tokens": thought_tokens,
        "total_tokens": total_tokens,
    }


def _usage_int(usage: Any, name: str) -> int:
    value = _value(usage, name)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _value(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _response_cost(model: str, usage: dict[str, int]) -> float:
    spec = _MODELS.get(model)
    if spec is None or not usage:
        return 0.0

    prices = (spec.input_price, spec.cached_input_price, spec.output_price)
    input_tokens = usage.get("input_tokens", 0)
    if input_tokens > _LONG_CONTEXT_THRESHOLD_TOKENS and spec.long_context_prices is not None:
        prices = spec.long_context_prices

    input_price, cached_input_price, output_price = prices
    cached_tokens = usage.get("cached_input_tokens", 0)
    uncached_tokens = max(0, input_tokens - cached_tokens)
    billed_output_tokens = usage.get("output_tokens", 0) + usage.get("thought_tokens", 0)
    return round(
        (
            uncached_tokens * input_price
            + cached_tokens * cached_input_price
            + billed_output_tokens * output_price
        )
        / 1_000_000,
        8,
    )


def context_window_tokens_for(model: str) -> int | None:
    spec = _MODELS.get(model)
    return spec.context_window_tokens if spec is not None else None


def _raise_for_bad_status(response: Any, tool_calls: list[dict[str, Any]]) -> None:
    status = _value(response, "status")
    if status in {None, "completed"} or (status == "requires_action" and tool_calls):
        return
    raise ModelError(f"Gemini interaction ended with status {status}")


def _format_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message or repr(exc)}"


__all__ = ["DEFAULT_GEMINI_MODEL", "KNOWN_MODELS", "GeminiModel", "context_window_tokens_for"]
