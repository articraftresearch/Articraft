from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import mini_articraft.agent.tools as tools
from mini_articraft import Environment, Model, package_dir
from mini_articraft.agent import events
from mini_articraft.agent.tools import ToolContext
from mini_articraft.record import Record, append_conversation
from mini_articraft.settings import DEFAULT_MAX_TURNS

PROMPT_SLUG_MAX_LENGTH = 48
MAX_CONSECUTIVE_EMPTY_RESPONSES = 3


class AgentConfig(BaseModel):
    max_turns: int = DEFAULT_MAX_TURNS
    output_path: Path | None = None


class Agent:
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        on_event: Callable[[events.Event], None] | None = None,
        **kwargs: Any,
    ):
        self.config = AgentConfig(**kwargs)
        self.model = model
        self.env = env
        self.messages: list[dict[str, Any]] = []
        self._on_event = on_event

    def _emit(self, event: events.Event) -> None:
        if self._on_event is not None:
            self._on_event(event)

    async def run(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        run_id = str(kwargs.get("run_id") or _run_id_for_prompt(prompt))
        run_dir = self.env.create_run(run_id)
        context = ToolContext(self.env, run_dir, run_dir / "workspace")
        conversation_path = run_dir / "conversation.jsonl"
        record_path = run_dir / "record.json"
        record = Record.load(record_path)
        record.status = "running"
        record.error = ""
        record.result = ""
        record.save(record_path)

        self.messages = [
            {
                "role": "system",
                "content": _read_prompt("system.md"),
            },
            {
                "role": "user",
                "content": _read_sdk_quickstart(),
            },
            {"role": "user", "content": _read_prompt("task.md").replace("{{ prompt }}", prompt)},
        ]
        for message in self.messages:
            append_conversation(conversation_path, message)

        model_config = getattr(self.model, "config", None)
        model_name = getattr(
            model_config,
            "selected_model",
            getattr(model_config, "openai_model", ""),
        )
        reasoning_effort = getattr(
            model_config,
            "selected_reasoning_effort",
            getattr(model_config, "openai_reasoning_effort", ""),
        )
        context_window_tokens = int(getattr(self.model, "context_window_tokens", 0) or 0)
        self._emit(
            events.RunStarted(
                run_id,
                model_name,
                prompt,
                reasoning_effort,
                context_window_tokens,
            )
        )

        started = time.perf_counter()
        final_text = ""
        cost = 0.0
        token_usage: dict[str, int] = {}
        turn = 0
        hit_max_turns = False
        termination_error = ""
        consecutive_empty_responses = 0
        try:
            for turn in range(1, self.config.max_turns + 1):
                self._emit(events.TurnStarted(turn))
                try:
                    response = await self.model.query(self.messages, tools=tools.schemas())
                except Exception as exc:
                    termination_error = f"model query failed: {type(exc).__name__}: {exc}"
                    break
                cost += _cost(response)
                response_usage = _token_usage(response)
                token_usage = _add_token_usage(token_usage, response_usage)
                _save_cost(run_dir, cost, token_usage)
                text = str(response.get("text") or "")
                tool_calls = list(response.get("tool_calls") or [])
                assistant = {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls,
                    "token_usage": response_usage,
                }
                self.messages.append(assistant)
                append_conversation(conversation_path, assistant)
                self._emit(events.AssistantMessage(turn, text, tool_calls, response_usage))

                if not tool_calls:
                    if text.strip():
                        consecutive_empty_responses = 0
                    else:
                        consecutive_empty_responses += 1

                    workspace_is_compiled = _latest_workspace_is_compiled(context)
                    if text.strip() and workspace_is_compiled:
                        final_text = text
                        break

                    if consecutive_empty_responses >= MAX_CONSECUTIVE_EMPTY_RESPONSES:
                        termination_error = (
                            "agent returned three consecutive responses with no visible text or "
                            "tool calls"
                        )
                        break
                    if consecutive_empty_responses == 2:
                        _append_reminder(
                            self.messages,
                            conversation_path,
                            _empty_response_reminder(
                                context,
                                workspace_is_compiled=workspace_is_compiled,
                            ),
                        )
                    elif workspace_is_compiled:
                        _append_reminder(
                            self.messages,
                            conversation_path,
                            _final_response_required_reminder(),
                        )
                    else:
                        _append_reminder(
                            self.messages,
                            conversation_path,
                            _compile_required_reminder(context),
                        )
                    continue

                consecutive_empty_responses = 0
                await self._run_tool_calls(context, tool_calls, conversation_path)
            else:
                hit_max_turns = True
        finally:
            await context.exec_sessions.aclose()
            # The agent owns the model for the run's duration, so a finished
            # run never leaves a live websocket behind. Close failures are
            # inconsequential teardown noise and must not change the outcome.
            with contextlib.suppress(Exception):
                await self.model.close()

        workspace_is_compiled = _latest_workspace_is_compiled(context)
        record = Record.load(record_path)
        if termination_error:
            record.status = "error"
            record.error = termination_error
            record.result = ""
        elif hit_max_turns:
            record.status = "error"
            record.error = "agent hit max turns limit"
            record.result = ""
        elif final_text and workspace_is_compiled:
            try:
                result_path = _result_path(run_dir, context.successful_compile_result)
            except ValueError as exc:
                record.status = "error"
                record.error = str(exc)
                record.result = ""
            else:
                if result_path and run_dir.joinpath(result_path).is_file():
                    record.status = "success"
                    record.error = ""
                    record.result = result_path
                else:
                    record.status = "error"
                    record.error = "fresh compile did not produce a USDZ result"
                    record.result = ""
        else:
            record.status = "error"
            record.error = "agent stopped without a visible final response after a fresh compile"
            record.result = ""
        record.save(record_path)

        data = record.to_dict()
        data["message"] = final_text
        data["run"] = str(run_dir)
        compile_result = (
            context.successful_compile_result if workspace_is_compiled else context.compile_result
        )
        if compile_result and isinstance(
            compile_result.get("compile_report"),
            dict,
        ):
            data["compile_report"] = compile_result["compile_report"]
        if self.config.output_path:
            record.save(self.config.output_path)
        self._emit(
            events.RunFinished(
                status=str(data.get("status") or ""),
                run=str(run_dir),
                result=str(data.get("result") or ""),
                error=str(data.get("error") or ""),
                turns=turn,
                duration=round(time.perf_counter() - started, 4),
                cost=float(data.get("cost") or 0.0),
                token_usage=dict(data.get("token_usage") or {}),
            )
        )
        return data

    async def _run_tool_calls(
        self,
        context: ToolContext,
        tool_calls: list[dict[str, Any]],
        conversation_path: Path,
    ) -> None:
        batch: list[dict[str, Any]] = []
        for call in tool_calls:
            if _supports_parallel(call):
                batch.append(call)
                continue

            await self._run_tool_batch(context, batch, conversation_path)
            batch = []
            await self._run_tool_batch(context, [call], conversation_path)

        await self._run_tool_batch(context, batch, conversation_path)

    async def _run_tool_batch(
        self,
        context: ToolContext,
        tool_calls: list[dict[str, Any]],
        conversation_path: Path,
    ) -> None:
        if not tool_calls:
            return

        started = []
        for call in tool_calls:
            self._emit(
                events.ToolStarted(
                    str(call["id"]), str(call["name"]), str(call.get("arguments") or "{}")
                )
            )
            started.append(time.perf_counter())

        completed = await asyncio.gather(*(self._run_tool(context, call) for call in tool_calls))
        for call, (item, payload), tool_started in zip(
            tool_calls,
            completed,
            started,
            strict=True,
        ):
            self.messages.append(item)
            append_conversation(
                conversation_path,
                tools.result_item(str(call["id"]), payload),
            )
            self._emit(
                events.ToolFinished(
                    str(call["id"]),
                    str(call["name"]),
                    _display_payload(context, str(call["name"]), payload),
                    round(time.perf_counter() - tool_started, 4),
                )
            )

    async def _run_tool(
        self,
        context: ToolContext,
        call: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        name = str(call["name"])
        call_id = str(call["id"])
        try:
            live_sessions = context.exec_sessions.live_ids()
            if live_sessions and name != "write_stdin":
                raise ValueError(
                    "finish the running exec_command with write_stdin before calling "
                    f"{name} (session_id={live_sessions[0]})"
                )
            tool = tools.get(name)
            result = await tool.run(context, _arguments(call))
            if isinstance(result, tools.ToolResult):
                payload = {"result": result.output}
                item = tools.result_item(
                    call_id,
                    payload,
                    content_items=result.content_items,
                )
                return item, payload
            payload = {"result": result}
        except Exception as exc:
            payload = {"error": str(exc)}
        return tools.result_item(call_id, payload), payload


def _display_payload(
    context: ToolContext,
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Give the TUI full compile details without adding them to model context."""
    if name != "compile" or "result" not in payload:
        return payload
    compact_result = payload["result"]
    if not isinstance(compact_result, dict):
        return payload
    full_result = context.compile_result
    if compact_result.get("status") == "success":
        full_result = context.successful_compile_result or full_result
    if not isinstance(full_result, dict):
        return payload
    return {**payload, "result": full_result}


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments") or "{}"
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("tool arguments must be a JSON object")
    return payload


def _supports_parallel(call: dict[str, Any]) -> bool:
    try:
        name = str(call["name"])
        return name == "read" and tools.get(name).supports_parallel
    except (KeyError, ValueError):
        return False


def _latest_workspace_is_compiled(context: ToolContext) -> bool:
    if context.exec_sessions.live_ids():
        return False
    return context.refresh_compile_freshness()


def _has_successful_compile(context: ToolContext) -> bool:
    return context.successful_compile_result is not None


def _compile_required_reminder(context: ToolContext) -> str:
    if live_sessions := context.exec_sessions.live_ids():
        return (
            "<compile_required>\n"
            f"exec_command session {live_sessions[0]} is still running.\n"
            "Use `write_stdin` until it exits, then run `compile` before concluding.\n"
            "</compile_required>"
        )
    reason = (
        "The workspace has changed since the last successful compile."
        if _has_successful_compile(context)
        else "No successful compile has completed yet."
    )
    return f"<compile_required>\n{reason}\nRun `compile` before concluding.\n</compile_required>"


def _final_response_required_reminder() -> str:
    return (
        "<final_response_required>\n"
        "The latest workspace has already compiled successfully.\n"
        "Return a visible final response, or call a tool if further work is needed.\n"
        "</final_response_required>"
    )


def _empty_response_reminder(
    context: ToolContext,
    *,
    workspace_is_compiled: bool,
) -> str:
    if workspace_is_compiled:
        tag = "final_response_required"
        state = "The latest workspace has already compiled successfully."
        action = "Return a visible final response now, or call a tool if further work is needed."
    elif live_sessions := context.exec_sessions.live_ids():
        tag = "compile_required"
        state = f"exec_command session {live_sessions[0]} is still running."
        action = "Use `write_stdin` until it exits, then run `compile` before concluding."
    else:
        tag = "compile_required"
        state = (
            "The workspace has changed since the last successful compile."
            if _has_successful_compile(context)
            else "No successful compile has completed yet."
        )
        action = "Complete the workspace changes and run `compile` before concluding."
    return (
        f"<{tag}>\n"
        "Your previous response produced no visible text and no tool calls.\n"
        "Do not continue with reasoning-only output.\n"
        f"{action}\n"
        f"{state}\n"
        f"</{tag}>"
    )


def _append_reminder(messages: list[dict[str, Any]], path: Path, content: str) -> None:
    reminder = {"role": "user", "content": content}
    messages.append(reminder)
    append_conversation(path, reminder)


def _result_path(run_dir: Path, compile_result: dict[str, Any] | None) -> str:
    raw = compile_result.get("usdz") if compile_result else None
    if not raw:
        return ""
    path = Path(str(raw))
    path = path if path.is_absolute() else run_dir / path
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("compiled USDZ path must stay inside the run directory") from exc


def _save_cost(run_dir: Path, cost: float, token_usage: dict[str, int]) -> None:
    record = Record.load(run_dir / "record.json")
    record.cost = round(cost, 8)
    record.token_usage = dict(token_usage)
    record.save(run_dir / "record.json")


def _cost(response: dict[str, Any]) -> float:
    try:
        return float(response.get("cost") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _token_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("token_usage")
    if not isinstance(usage, dict):
        return {}
    return {str(key): int(value) for key, value in usage.items()}


def _add_token_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    keys = left.keys() | right.keys()
    return {key: left.get(key, 0) + right.get(key, 0) for key in keys}


def _read_prompt(name: str) -> str:
    return (package_dir / "prompts" / name).read_text(encoding="utf-8")


def _read_sdk_quickstart() -> str:
    quickstart = (package_dir / "sdk" / "docs" / "common" / "00_quickstart.md").read_text(
        encoding="utf-8"
    )
    return (
        "<sdk_quickstart>\n"
        "This SDK quickstart is preloaded for the run. Use it as the first "
        "reference. Before writing code, inspect the current script and survey "
        "the relevant SDK pages across plausible build123d and mesh approaches. "
        "Use parallel reads when comparing independent references. Do not stop "
        "at the first workable API.\n\n"
        f"{quickstart.rstrip()}\n"
        "</sdk_quickstart>"
    )


def _run_id_for_prompt(prompt: str, *, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{_prompt_slug(prompt)}"


def _prompt_slug(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    slug = slug[:PROMPT_SLUG_MAX_LENGTH].strip("-")
    return slug or "prompt"
