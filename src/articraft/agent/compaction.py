from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from articraft import package_dir

RESERVE_TOKENS = 16_384
KEEP_RECENT_TOKENS = 20_000
SUMMARY_MAX_OUTPUT_TOKENS = 8_192
STATIC_MESSAGE_COUNT = 3
TOOL_RESULT_MAX_CHARS = 2_000
ESTIMATED_IMAGE_TOKENS = 1_200


@dataclass(frozen=True)
class Compaction:
    tokens_before: int
    recent_messages: list[dict[str, Any]]
    summary_messages: list[dict[str, Any]]
    modified_files: list[str]
    inspected_images: list[str]

    def apply(self, messages: list[dict[str, Any]], summary: str) -> list[dict[str, Any]]:
        checkpoint = {
            "role": "user",
            "content": _checkpoint(summary, self.modified_files, self.inspected_images),
            "compaction": {
                "summary": summary,
                "modified_files": self.modified_files,
                "inspected_images": self.inspected_images,
            },
        }
        return [*messages[:STATIC_MESSAGE_COUNT], checkpoint, *self.recent_messages]

    def record(self, summary: str, usage: dict[str, int]) -> dict[str, Any]:
        return {
            "type": "compaction",
            "summary": summary,
            "tokens_before": self.tokens_before,
            "modified_files": self.modified_files,
            "inspected_images": self.inspected_images,
            "token_usage": usage,
        }


def prepare_compaction(
    messages: list[dict[str, Any]],
    context_window_tokens: int,
) -> Compaction | None:
    if len(messages) <= STATIC_MESSAGE_COUNT or context_window_tokens <= 0:
        return None

    tokens_before = _context_tokens(messages)
    if tokens_before <= context_window_tokens - RESERVE_TOKENS:
        return None

    cut = _recent_cut(messages)
    if cut <= STATIC_MESSAGE_COUNT:
        return None

    old_messages = messages[STATIC_MESSAGE_COUNT:cut]
    previous = ""
    for message in old_messages:
        metadata = message.get("compaction")
        if isinstance(metadata, dict) and isinstance(metadata.get("summary"), str):
            previous = metadata["summary"]
            break
    new_messages = [
        message for message in old_messages if not isinstance(message.get("compaction"), dict)
    ]
    if not new_messages and not previous:
        return None

    modified_files, inspected_images = _tracked_paths(messages)
    task = _text(messages[2].get("content"))
    summary_parts = [f"<task>\n{task}\n</task>"]
    if previous:
        summary_parts.append(f"<previous-checkpoint>\n{previous}\n</previous-checkpoint>")
    summary_parts.append(f"<old-work>\n{_summary_history(new_messages)}\n</old-work>")

    return Compaction(
        tokens_before=tokens_before,
        recent_messages=messages[cut:],
        summary_messages=[
            {
                "role": "system",
                "content": (package_dir / "prompts" / "compaction.md").read_text(encoding="utf-8"),
            },
            {"role": "user", "content": "\n\n".join(summary_parts)},
        ],
        modified_files=modified_files,
        inspected_images=inspected_images,
    )


def _context_tokens(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        usage = messages[index].get("token_usage")
        if messages[index].get("role") != "assistant" or not isinstance(usage, dict):
            continue
        try:
            total = int(usage.get("total_tokens") or 0)
        except (TypeError, ValueError):
            total = 0
        if total > 0:
            trailing = sum(_message_tokens(message) for message in messages[index + 1 :])
            return total + trailing
    return sum(_message_tokens(message) for message in messages)


def _summary_history(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.get("type") == "function_call_output":
            parts.append(f"[Tool result]\n{_truncate(_tool_output(message.get('output')))}")
            continue

        role = message.get("role")
        text = _text(message.get("content")).strip()
        if text and role in {"user", "assistant"}:
            parts.append(f"[{str(role).title()}]\n{text}")
        if role == "assistant":
            calls = [
                _tool_call(call)
                for call in message.get("tool_calls") or []
                if isinstance(call, dict)
            ]
            if calls:
                parts.append("[Assistant tool calls]\n" + "\n".join(calls))
    return "\n\n".join(parts) or "(no new text)"


def _recent_cut(messages: list[dict[str, Any]]) -> int:
    tokens = 0
    for index in range(len(messages) - 1, STATIC_MESSAGE_COUNT - 1, -1):
        tokens += _message_tokens(messages[index])
        if tokens >= KEEP_RECENT_TOKENS and messages[index].get("role") == "assistant":
            return index
    return STATIC_MESSAGE_COUNT


def _message_tokens(message: dict[str, Any]) -> int:
    chars = _content_chars(message.get("content")) + _content_chars(message.get("output"))
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            chars += len(str(call.get("name") or ""))
            chars += len(str(call.get("arguments") or ""))
    return (chars + 3) // 4


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    return sum(
        ESTIMATED_IMAGE_TOKENS * 4
        if isinstance(item, dict) and item.get("type") in {"input_image", "image"}
        else len(str(item.get("text") or ""))
        for item in content
        if isinstance(item, dict)
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(item.get("text") or "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "input_text"
    )


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments") or {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _tool_call(call: dict[str, Any]) -> str:
    name = str(call.get("name") or "")
    arguments = _arguments(call)
    if name == "write":
        arguments["content"] = "[file content omitted]"
    elif name == "edit":
        arguments.pop("old_text", None)
        arguments.pop("new_text", None)
        arguments["text"] = "[edit text omitted]"
    return f"{name}({_truncate(json.dumps(arguments, sort_keys=True))})"


def _tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        return json.dumps(output)
    text = _text(output)
    images = sum(isinstance(item, dict) and item.get("type") == "input_image" for item in output)
    return "\n".join(
        part for part in [f"[{images} image payload omitted]" if images else "", text] if part
    )


def _tracked_paths(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    modified: set[str] = set()
    inspected: set[str] = set()
    for message in messages:
        metadata = message.get("compaction")
        if isinstance(metadata, dict):
            modified.update(
                path for path in metadata.get("modified_files", []) if isinstance(path, str)
            )
            inspected.update(
                path for path in metadata.get("inspected_images", []) if isinstance(path, str)
            )
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            path = _arguments(call).get("path")
            if not isinstance(path, str) or not path:
                continue
            if call.get("name") in {"write", "edit"}:
                modified.add(path)
            elif call.get("name") == "view_image":
                inspected.add(path)
    return sorted(modified), sorted(inspected)


def _checkpoint(summary: str, modified: list[str], inspected: list[str]) -> str:
    parts = [summary.strip()]
    if modified:
        parts.append("<modified-files>\n" + "\n".join(modified) + "\n</modified-files>")
    if inspected:
        parts.append("<inspected-images>\n" + "\n".join(inspected) + "\n</inspected-images>")
    return "<compaction-checkpoint>\n" + "\n\n".join(parts) + "\n</compaction-checkpoint>"


def _truncate(text: str) -> str:
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return text
    omitted = len(text) - TOOL_RESULT_MAX_CHARS
    return f"{text[:TOOL_RESULT_MAX_CHARS]}\n[... {omitted} more characters omitted]"
