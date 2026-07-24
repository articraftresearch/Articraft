from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any

import websockets
from websockets.exceptions import WebSocketException

from mini_articraft.errors import ModelError
from mini_articraft.settings import Settings, get_settings

_WEBSOCKET_URL = "wss://api.openai.com/v1/responses"
_MAX_OUTPUT_TOKENS = 128_000
_RETRY_BASE_SECONDS = 0.5
_RETRY_MAX_SECONDS = 20.0
_WEBSOCKET_OPEN_TIMEOUT_SECONDS = 20.0
# Codex caps the 1.05M API window at 272K to avoid much higher cost and lower quality.
_CODEX_CONTEXT_WINDOW_TOKENS = 272_000
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ModelSpec:
    context_window_tokens: int
    input_price: float
    cached_input_price: float
    output_price: float
    cache_write_price: float | None = None


# Prices are USD per million tokens: input, cached input, output, cache write.
_MODELS = {
    "gpt-5.6-sol": _ModelSpec(_CODEX_CONTEXT_WINDOW_TOKENS, 5.0, 0.5, 30.0, 6.25),
    "gpt-5.5-pro": _ModelSpec(_CODEX_CONTEXT_WINDOW_TOKENS, 30.0, 30.0, 180.0),
    "gpt-5.5": _ModelSpec(_CODEX_CONTEXT_WINDOW_TOKENS, 5.0, 0.5, 30.0),
    "gpt-5.4-pro": _ModelSpec(1_050_000, 30.0, 30.0, 180.0),
    "gpt-5.4-mini": _ModelSpec(400_000, 0.75, 0.075, 4.5),
    "gpt-5.4-nano": _ModelSpec(400_000, 0.2, 0.02, 1.25),
    "gpt-5.4": _ModelSpec(1_050_000, 2.5, 0.25, 15.0),
}
_MODEL_ALIASES = {"gpt-5.6": "gpt-5.6-sol"}


class OpenAIModel:
    def __init__(self, settings: Settings | None = None):
        self.config = settings or get_settings()
        if not self.config.openai_api_key:
            raise ModelError("OpenAI credentials are required. Set OPENAI_API_KEY.")
        self._websocket: Any = None
        self._input_items: list[dict[str, Any]] = []
        self._last_message_count = 0
        self._previous_response_id: str | None = None

    @property
    def context_window_tokens(self) -> int:
        return context_window_tokens_for(self.config.openai_model) or 0

    async def query(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Query the OpenAI Responses API and return the completed text response."""
        new_items = self._new_input_items(messages)
        previous_response_id = self._previous_response_id
        if previous_response_id is None:
            input_items = [*self._input_items, *new_items]
        else:
            input_items = new_items

        request = self._request(messages, input_items, previous_response_id, tools)
        fallback_request = self._request(
            messages,
            [*self._input_items, *new_items],
            None,
            tools,
        )

        response = await self._send_with_retries(request, fallback_request)
        text = _response_text(response)
        tool_calls = _response_tool_calls(response)
        _raise_for_bad_status(response, text)
        if not text and not tool_calls and not _response_has_reasoning(response):
            raise ModelError("OpenAI response did not contain output_text")

        self._input_items.extend(new_items)
        self._input_items.extend(_response_output(response))
        self._previous_response_id = _response_id(response)
        self._last_message_count = len(messages)
        return {
            "text": text,
            "tool_calls": tool_calls,
            "token_usage": _response_token_usage(response),
            "cost": _response_cost(response),
            "response": response,
        }

    async def close(self) -> None:
        await self._close_websocket()

    def _request(
        self,
        messages: list[dict[str, Any]],
        input_items: list[dict[str, Any]],
        previous_response_id: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.openai_model,
            "input": input_items,
            "reasoning": {"effort": self.config.openai_reasoning_effort},
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": _MAX_OUTPUT_TOKENS,
            "store": False,
        }
        if previous_response_id is not None:
            request["previous_response_id"] = previous_response_id
        if tools:
            request["tools"] = tools
            request["parallel_tool_calls"] = True
        instructions = _instructions(messages)
        if instructions:
            request["instructions"] = instructions
        return request

    async def _send_with_retries(
        self,
        request: dict[str, Any],
        fallback_request: dict[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(1, self.config.openai_max_attempts + 1):
            try:
                return await asyncio.wait_for(
                    self._send_with_fallback(request, fallback_request),
                    timeout=self.config.openai_request_timeout_seconds,
                )
            except Exception as exc:
                if attempt >= self.config.openai_max_attempts or not _should_retry(exc):
                    raise
                await self._close_websocket()
                delay = random.random() * min(
                    _RETRY_MAX_SECONDS,
                    _RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
                )
                logger.warning(
                    "OpenAI request failed (attempt %s/%s), retrying in %.2fs: %s",
                    attempt,
                    self.config.openai_max_attempts,
                    delay,
                    _format_exception(exc),
                )
                await asyncio.sleep(delay)
        raise AssertionError("retry loop did not return or raise")

    async def _send_with_fallback(
        self,
        request: dict[str, Any],
        fallback_request: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return await self._send_websocket(request)
        except _OpenAIWebSocketError as exc:
            if exc.code != "previous_response_not_found" or "previous_response_id" not in request:
                raise
            logger.warning("OpenAI lost the previous response; resending the full conversation")
            return await self._send_websocket(fallback_request, force_reconnect=True)

    async def _send_websocket(
        self,
        request: dict[str, Any],
        *,
        force_reconnect: bool = False,
    ) -> dict[str, Any]:
        websocket = await self._ensure_websocket(force_reconnect=force_reconnect)
        await websocket.send(json.dumps({"type": "response.create", **request}))
        return await _receive_websocket_response(websocket)

    async def _ensure_websocket(self, *, force_reconnect: bool = False) -> Any:
        if (
            not force_reconnect
            and self._websocket is not None
            and not getattr(self._websocket, "closed", False)
        ):
            return self._websocket

        await self._close_websocket()
        self._websocket = await websockets.connect(
            _WEBSOCKET_URL,
            additional_headers={"Authorization": f"Bearer {self.config.openai_api_key}"},
            open_timeout=_WEBSOCKET_OPEN_TIMEOUT_SECONDS,
            max_size=None,
        )
        return self._websocket

    async def _close_websocket(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is not None:
            try:
                await websocket.close()
            except Exception:
                logger.debug("OpenAI websocket close failed", exc_info=True)

    def _new_input_items(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        new_items: list[dict[str, Any]] = []
        for message in messages[self._last_message_count :]:
            if "type" in message:
                new_items.append(message)
                continue
            if message["role"] == "system":
                continue
            if self._previous_response_id is not None and message["role"] == "assistant":
                continue
            new_items.append(_input_message(message))
        return new_items


async def _receive_websocket_response(websocket: Any) -> dict[str, Any]:
    while True:
        raw = await websocket.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
        event_type = event.get("type")

        if event_type == "error":
            raise _OpenAIWebSocketError.from_event(event)
        if event_type in {"response.completed", "response.incomplete"}:
            response = event.get("response")
            if not isinstance(response, dict):
                raise _OpenAIWebSocketError(
                    code="missing_response",
                    message=f"{event_type} did not include a response object",
                )
            return response
        if event_type == "response.failed":
            raise _OpenAIWebSocketError.from_event(event)


class _OpenAIWebSocketError(ModelError):
    def __init__(self, *, code: str | None, message: str, status: int | None = None):
        self.code = code
        self.status = status
        prefix = f"{code}: " if code else ""
        super().__init__(f"{prefix}{message}")

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> _OpenAIWebSocketError:
        error: Any = event.get("error")
        if event.get("type") == "response.failed" and isinstance(event.get("response"), dict):
            error = event["response"].get("error") or error
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message") or json.dumps(error, sort_keys=True)
        else:
            code = None
            message = str(error or "OpenAI websocket error")
        status = event.get("status")
        return cls(
            code=str(code) if code is not None else None,
            message=str(message),
            status=status if isinstance(status, int) else None,
        )


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, OSError, WebSocketException)):
        status = _http_status(exc)
        return status is None or _is_transient_status(status)

    status = _http_status(exc)
    if status is not None:
        return _is_transient_status(status)

    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return True
    if isinstance(exc, _OpenAIWebSocketError):
        if exc.code == "previous_response_not_found":
            return False
        return exc.code in {
            "internal_error",
            "missing_response",
            "overloaded",
            "rate_limit_exceeded",
            "response_failed",
            "server_error",
            "temporarily_unavailable",
            "websocket_connection_limit_reached",
        }
    return False


def _http_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _is_transient_status(status: int) -> bool:
    return status in {408, 409, 425, 429} or status >= 500


def _format_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message or repr(exc)}"


def _instructions(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        _message_text(message)
        for message in messages
        if "type" not in message and message["role"] == "system"
    )


def _input_message(message: dict[str, Any]) -> dict[str, Any]:
    return {"role": message["role"], "content": _message_text(message)}


def _message_text(message: dict[str, Any]) -> str:
    content = message["content"]
    if not isinstance(content, str):
        raise TypeError("OpenAIModel messages must use string content")
    return content


def _response_id(response: dict[str, Any]) -> str | None:
    response_id = response.get("id")
    return response_id if isinstance(response_id, str) else None


def _response_output(response: dict[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        return []
    return [item for item in output if isinstance(item, dict)]


def _response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    for item in _response_output(response):
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _response_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _response_output(response):
        if item.get("type") != "function_call":
            continue
        calls.append(
            {
                "id": item["call_id"],
                "name": item["name"],
                "arguments": item.get("arguments") or "{}",
            }
        )
    return calls


def _response_has_reasoning(response: dict[str, Any]) -> bool:
    return any(item.get("type") == "reasoning" for item in _response_output(response))


def _response_cost(response: dict[str, Any]) -> float:
    spec = _model_spec(str(response.get("model") or ""))
    usage = _response_token_usage(response)
    if not usage or spec is None:
        return 0.0

    uncached_tokens = max(
        0,
        usage["input_tokens"] - usage["cached_input_tokens"] - usage["cache_write_tokens"],
    )
    cache_write_price = spec.cache_write_price or spec.input_price
    return round(
        (
            uncached_tokens * spec.input_price
            + usage["cached_input_tokens"] * spec.cached_input_price
            + usage["cache_write_tokens"] * cache_write_price
            + usage["output_tokens"] * spec.output_price
        )
        / 1_000_000,
        8,
    )


def _response_token_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}

    details = usage.get("input_tokens_details")
    cached_tokens = _int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    cache_write_tokens = _int(details.get("cache_write_tokens")) if isinstance(details, dict) else 0
    input_tokens = _int(usage.get("input_tokens"))
    output_tokens = _int(usage.get("output_tokens"))
    total_tokens = _int(usage.get("total_tokens")) or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _model_spec(model: str) -> _ModelSpec | None:
    model = _MODEL_ALIASES.get(model, model)
    for name, spec in sorted(_MODELS.items(), key=lambda item: -len(item[0])):
        if model == name or model.startswith(f"{name}-"):
            return spec
    return None


def context_window_tokens_for(model: str) -> int | None:
    spec = _model_spec(model)
    return spec.context_window_tokens if spec is not None else None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _raise_for_bad_status(response: dict[str, Any], text: str) -> None:
    status = response["status"]
    if status == "completed":
        return
    if status == "incomplete":
        details = response.get("incomplete_details") or {}
        reason = details.get("reason", "unknown")
        if text:
            raise ModelError(f"OpenAI response incomplete ({reason}); partial output returned")
        raise ModelError(f"OpenAI response incomplete ({reason}); no visible output")
    raise ModelError(f"OpenAI response ended with status {status}")
