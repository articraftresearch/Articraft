from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from articraft.errors import ModelError
from articraft.settings import Settings, get_settings

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRY_BASE_SECONDS = 0.5
_RETRY_MAX_SECONDS = 20.0
logger = logging.getLogger(__name__)


class OpenRouterModel:
    supports_images = False

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        self.config = settings or get_settings()
        if not (self.config.openrouter_api_key or "").strip():
            raise ModelError("OpenRouter credentials are required. Set OPENROUTER_API_KEY.")
        if not self.config.openrouter_model.strip():
            raise ModelError(
                "OpenRouter model is required. Pass --model or set ARTICRAFT_OPENROUTER_MODEL."
            )
        self._client = client

    @property
    def context_window_tokens(self) -> int:
        """Return unknown because OpenRouter accepts models without a local catalog."""
        return 0

    async def query(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Query OpenRouter and return the response shape used by the agent."""
        request: dict[str, Any] = {
            "model": self.config.openrouter_model,
            "messages": _messages(messages),
        }
        converted_tools = _tools(tools or [])
        if converted_tools:
            request["tools"] = converted_tools

        response = await self._send_with_retries(request)
        payload = _response_payload(response)
        _raise_for_provider_error(response.status_code, payload)
        text, tool_calls, provider_content = _assistant_output(payload)
        if not text and not tool_calls:
            raise ModelError("OpenRouter response did not contain text or tool calls")

        return {
            "text": text,
            "tool_calls": tool_calls,
            "token_usage": _response_token_usage(payload),
            "cost": _response_cost(payload),
            "provider_content": provider_content,
            "response": payload,
        }

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def _send_with_retries(self, request: dict[str, Any]) -> httpx.Response:
        for attempt in range(1, self.config.openrouter_max_attempts + 1):
            response: httpx.Response | None = None
            try:
                response = await self._client_or_create().post(
                    _API_URL,
                    headers=self._headers(),
                    json=request,
                    timeout=self.config.openrouter_request_timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.config.openrouter_max_attempts:
                    raise ModelError(
                        f"OpenRouter request failed: {_format_exception(exc)}"
                    ) from exc
                delay = _retry_delay(attempt)
                logger.warning(
                    "OpenRouter request failed (attempt %s/%s), retrying in %.2fs: %s",
                    attempt,
                    self.config.openrouter_max_attempts,
                    delay,
                    _format_exception(exc),
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code < 400:
                payload = _response_payload(response)
                provider_status = _embedded_provider_status(payload)
                if (
                    provider_status is not None
                    and _retryable_status(provider_status)
                    and attempt < self.config.openrouter_max_attempts
                ):
                    delay = _retry_delay(attempt, response.headers.get("Retry-After"))
                    logger.warning(
                        "OpenRouter provider failed (attempt %s/%s), retrying in %.2fs: "
                        "provider code %s",
                        attempt,
                        self.config.openrouter_max_attempts,
                        delay,
                        provider_status,
                    )
                    await asyncio.sleep(delay)
                    continue
                return response

            if _retryable_status(response.status_code) and (
                attempt < self.config.openrouter_max_attempts
            ):
                delay = _retry_delay(attempt, response.headers.get("Retry-After"))
                logger.warning(
                    "OpenRouter request failed (attempt %s/%s), retrying in %.2fs: HTTP %s",
                    attempt,
                    self.config.openrouter_max_attempts,
                    delay,
                    response.status_code,
                )
                await asyncio.sleep(delay)
                continue

            payload = _error_payload(response)
            raise ModelError(_provider_error(response.status_code, payload))

        raise AssertionError("retry loop did not return or raise")

    def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {(self.config.openrouter_api_key or '').strip()}",
            "Content-Type": "application/json",
        }
        referer = (self.config.openrouter_http_referer or "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        title = (self.config.openrouter_app_title or "").strip()
        if title:
            headers["X-OpenRouter-Title"] = title
        return headers


def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.get("type") == "function_call_output":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("call_id") or ""),
                    "content": _tool_result_content(message.get("output")),
                }
            )
            continue

        role = message.get("role")
        if role in {"system", "user"}:
            converted.append({"role": role, "content": _text_content(message.get("content"))})
        elif role == "assistant":
            converted.append(_assistant_message(message))
    return converted


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    text = _text_content(message.get("content"))
    converted: dict[str, Any] = {
        "role": "assistant",
        "content": text or None,
    }
    tool_calls = [
        {
            "id": str(call.get("id") or ""),
            "type": "function",
            "function": {
                "name": str(call.get("name") or ""),
                "arguments": _arguments_text(call.get("arguments")),
            },
        }
        for call in message.get("tool_calls") or []
        if isinstance(call, dict)
    ]
    if tool_calls:
        converted["tool_calls"] = tool_calls
    for item in message.get("provider_content") or []:
        if not isinstance(item, dict) or item.get("type") != "openrouter_reasoning":
            continue
        reasoning = item.get("reasoning")
        if isinstance(reasoning, str):
            converted["reasoning"] = reasoning
        reasoning_details = item.get("reasoning_details")
        if isinstance(reasoning_details, list):
            converted["reasoning_details"] = reasoning_details
    return converted


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError("OpenRouterModel message content must be a string or list")

    text: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_image":
            raise ModelError("OpenRouterModel supports text input and function calling, not images")
        if item.get("type") == "input_text":
            text.append(str(item.get("text") or ""))
    return "\n".join(text)


def _tool_result_content(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return _text_content(output)
    return json.dumps(output)


def _arguments_text(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments)


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "parameters": tool.get("parameters") or {},
                "strict": bool(tool.get("strict", False)),
            },
        }
        for tool in tools
        if tool.get("type") == "function"
    ]


def _response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ModelError(
            f"OpenRouter response was not valid JSON (HTTP {response.status_code})"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelError(f"OpenRouter response was not an object (HTTP {response.status_code})")
    return payload


def _error_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {"error": {"message": response.text.strip() or "request failed"}}
    if isinstance(payload, dict):
        return payload
    return {"error": {"message": response.text.strip() or "request failed"}}


def _raise_for_provider_error(status: int, payload: dict[str, Any]) -> None:
    error = payload.get("error")
    if isinstance(error, dict):
        raise ModelError(_provider_error(status, payload))

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelError(f"OpenRouter response did not contain choices (HTTP {status})")
    first = choices[0]
    if not isinstance(first, dict):
        raise ModelError(f"OpenRouter response contained an invalid choice (HTTP {status})")
    if isinstance(first.get("error"), dict) or first.get("finish_reason") == "error":
        raise ModelError(_provider_error(status, first))


def _embedded_provider_status(payload: dict[str, Any]) -> int | None:
    error = payload.get("error")
    if isinstance(error, dict):
        return _status_code(error.get("code"))

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    choice_error = choices[0].get("error")
    if isinstance(choice_error, dict):
        return _status_code(choice_error.get("code"))
    return None


def _status_code(value: Any) -> int | None:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    return status if status > 0 else None


def _provider_error(status: int, payload: dict[str, Any]) -> str:
    error = payload.get("error")
    message = ""
    code: Any = None
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        code = error.get("code")
    elif isinstance(error, str):
        message = error.strip()

    detail = message or "request failed"
    code_detail = f", provider code {code}" if code not in {None, status} else ""
    return f"OpenRouter request failed (HTTP {status}{code_detail}): {detail}"


def _assistant_output(
    payload: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ModelError("OpenRouter response did not contain a valid choice")
    first = choices[0]
    message = first.get("message")
    if not isinstance(message, dict):
        raise ModelError("OpenRouter response choice did not contain an assistant message")

    content = message.get("content")
    text = content if isinstance(content, str) else ""
    calls: list[dict[str, Any]] = []
    for raw_call in message.get("tool_calls") or []:
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        call_id = str(raw_call.get("id") or "")
        name = str(function.get("name") or "")
        if not call_id or not name:
            raise ModelError("OpenRouter returned a function call without an id or name")
        calls.append(
            {
                "id": call_id,
                "name": name,
                "arguments": _arguments_text(function.get("arguments")),
            }
        )
    provider_content: list[dict[str, Any]] = []
    reasoning: dict[str, Any] = {"type": "openrouter_reasoning"}
    raw_reasoning = message.get("reasoning")
    if isinstance(raw_reasoning, str):
        reasoning["reasoning"] = raw_reasoning
    reasoning_details = message.get("reasoning_details")
    if isinstance(reasoning_details, list):
        reasoning["reasoning_details"] = reasoning_details
    if len(reasoning) > 1:
        provider_content.append(reasoning)
    return text, calls, provider_content


def _response_token_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}

    input_tokens = _int(usage.get("prompt_tokens"))
    output_tokens = _int(usage.get("completion_tokens"))
    total_tokens = _int(usage.get("total_tokens")) or input_tokens + output_tokens
    details = usage.get("prompt_tokens_details")
    cached_input_tokens = _int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _response_cost(payload: dict[str, Any]) -> float:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0.0
    try:
        return float(usage.get("cost") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _retryable_status(status: int) -> bool:
    return status in {408, 429} or status >= 500


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                now = datetime.now(retry_at.tzinfo or UTC)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                delay = (retry_at - now).total_seconds()
                return max(0.0, delay)
            except (OverflowError, TypeError, ValueError):
                pass
    return min(_RETRY_MAX_SECONDS, _RETRY_BASE_SECONDS * (2 ** (attempt - 1)))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message or repr(exc)}"


__all__ = ["OpenRouterModel"]
