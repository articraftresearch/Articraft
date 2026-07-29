from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from mini_articraft import package_dir

RESERVE_TOKENS = 16_384
KEEP_RECENT_TOKENS = 20_000
SUMMARY_MAX_OUTPUT_TOKENS = 8_192
STATIC_MESSAGE_COUNT = 3
TOOL_RESULT_MAX_CHARS = 2_000
ESTIMATED_IMAGE_TOKENS = 1_200


@dataclass(frozen=True)
class CompactionPlan:
    tokens_before: int
    messages_to_summarize: list[dict[str, Any]]
    recent_messages: list[dict[str, Any]]
    summary_messages: list[dict[str, Any]]
    previous_summary: str
    modified_files: list[str]
    inspected_images: list[str]


def prepare_compaction(
    messages: list[dict[str, Any]],
    context_window_tokens: int,
) -> CompactionPlan | None:
    if context_window_tokens <= 0 or len(messages) <= STATIC_MESSAGE_COUNT:
        return None

    tokens_before = estimate_context_tokens(messages)
    if tokens_before <= context_window_tokens - RESERVE_TOKENS:
        return None

    cut_index = _recent_cut_index(messages)
    if cut_index <= STATIC_MESSAGE_COUNT:
        return None

    prefix = messages[STATIC_MESSAGE_COUNT:cut_index]
    recent = messages[cut_index:]
    previous_summary = _previous_summary(prefix)
    new_messages = [message for message in prefix if not _is_compaction_message(message)]
    if not new_messages and not previous_summary:
        return None

    modified_files, inspected_images = _tracked_paths(messages)
    summary_messages = [
        {
            "role": "system",
            "content": (package_dir / "prompts" / "compaction.md").read_text(encoding="utf-8"),
        },
        {
            "role": "user",
            "content": _summary_input(
                _task_text(messages),
                previous_summary,
                serialize_messages(new_messages),
            ),
        },
    ]
    return CompactionPlan(
        tokens_before=tokens_before,
        messages_to_summarize=prefix,
        recent_messages=recent,
        summary_messages=summary_messages,
        previous_summary=previous_summary,
        modified_files=modified_files,
        inspected_images=inspected_images,
    )


def compacted_messages(
    messages: list[dict[str, Any]],
    plan: CompactionPlan,
    summary: str,
) -> list[dict[str, Any]]:
    checkpoint = {
        "role": "user",
        "content": _checkpoint_content(
            summary,
            plan.modified_files,
            plan.inspected_images,
        ),
        "compaction": {
            "summary": summary,
            "modified_files": plan.modified_files,
            "inspected_images": plan.inspected_images,
        },
    }
    return [*messages[:STATIC_MESSAGE_COUNT], checkpoint, *plan.recent_messages]


def compaction_record(plan: CompactionPlan, summary: str, usage: dict[str, int]) -> dict[str, Any]:
    return {
        "type": "compaction",
        "summary": summary,
        "tokens_before": plan.tokens_before,
        "messages_summarized": len(plan.messages_to_summarize),
        "messages_kept": len(plan.recent_messages),
        "modified_files": plan.modified_files,
        "inspected_images": plan.inspected_images,
        "token_usage": usage,
    }


def estimate_context_tokens(messages: list[dict[str, Any]]) -> int:
    last_usage_index: int | None = None
    usage_tokens = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        usage = message.get("token_usage")
        if not isinstance(usage, dict):
            continue
        try:
            total = int(usage.get("total_tokens") or 0)
        except (TypeError, ValueError):
            total = 0
        if total > 0:
            last_usage_index = index
            usage_tokens = total
            break

    if last_usage_index is None:
        return sum(estimate_message_tokens(message) for message in messages)
    trailing = sum(estimate_message_tokens(message) for message in messages[last_usage_index + 1 :])
    return usage_tokens + trailing


def estimate_message_tokens(message: dict[str, Any]) -> int:
    chars = 0
    content = message.get("content")
    if isinstance(content, str):
        chars += len(content)
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"input_image", "image"}:
                chars += ESTIMATED_IMAGE_TOKENS * 4
            else:
                chars += len(str(item.get("text") or ""))

    output = message.get("output")
    if isinstance(output, str):
        chars += len(output)
    elif isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "input_image":
                chars += ESTIMATED_IMAGE_TOKENS * 4
            else:
                chars += len(str(item.get("text") or ""))

    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        chars += len(str(call.get("name") or ""))
        chars += len(str(call.get("arguments") or ""))
    return math.ceil(chars / 4)


def serialize_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.get("type") == "function_call_output":
            output = _tool_output_text(message.get("output"))
            parts.append(f"[Tool result]\n{_truncate(output, TOOL_RESULT_MAX_CHARS)}")
            continue

        role = message.get("role")
        if role == "user":
            text = _content_text(message.get("content"))
            if text:
                parts.append(f"[User]\n{text}")
        elif role == "assistant":
            text = str(message.get("content") or "").strip()
            if text:
                parts.append(f"[Assistant]\n{text}")
            calls = [
                _serialized_tool_call(call)
                for call in message.get("tool_calls") or []
                if isinstance(call, dict)
            ]
            if calls:
                parts.append(f"[Assistant tool calls]\n{chr(10).join(calls)}")
    return "\n\n".join(parts) or "(no new text)"


def _recent_cut_index(messages: list[dict[str, Any]]) -> int:
    recent_tokens = 0
    reached_target = False
    for index in range(len(messages) - 1, STATIC_MESSAGE_COUNT - 1, -1):
        recent_tokens += estimate_message_tokens(messages[index])
        if recent_tokens >= KEEP_RECENT_TOKENS:
            reached_target = True
        if reached_target and messages[index].get("role") == "assistant":
            return index
    return STATIC_MESSAGE_COUNT


def _previous_summary(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        metadata = message.get("compaction")
        if isinstance(metadata, dict) and isinstance(metadata.get("summary"), str):
            return str(metadata["summary"])
    return ""


def _is_compaction_message(message: dict[str, Any]) -> bool:
    return isinstance(message.get("compaction"), dict)


def _tracked_paths(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    modified: set[str] = set()
    inspected: set[str] = set()
    for message in messages:
        metadata = message.get("compaction")
        if isinstance(metadata, dict):
            modified.update(_string_list(metadata.get("modified_files")))
            inspected.update(_string_list(metadata.get("inspected_images")))
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            arguments = _tool_arguments(call)
            path = arguments.get("path")
            if not isinstance(path, str) or not path:
                continue
            if name in {"write", "edit"}:
                modified.add(path)
            elif name == "view_image":
                inspected.add(path)
    return sorted(modified), sorted(inspected)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _serialized_tool_call(call: dict[str, Any]) -> str:
    name = str(call.get("name") or "")
    arguments = _tool_arguments(call)
    if name == "write":
        arguments.pop("content", None)
        arguments["content"] = "[file content omitted]"
    elif name == "edit":
        arguments.pop("old_text", None)
        arguments.pop("new_text", None)
        arguments["text"] = "[edit text omitted]"
    raw = json.dumps(arguments, sort_keys=True)
    return f"{name}({_truncate(raw, TOOL_RESULT_MAX_CHARS)})"


def _tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments") or "{}"
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _tool_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        texts = [
            str(item.get("text") or "")
            for item in output
            if isinstance(item, dict) and item.get("type") == "input_text"
        ]
        image_count = sum(
            1 for item in output if isinstance(item, dict) and item.get("type") == "input_image"
        )
        if image_count:
            texts.insert(0, f"[{image_count} image payload omitted]")
        return "\n".join(texts)
    return json.dumps(output)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = [
        str(item.get("text") or "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "input_text"
    ]
    return "\n".join(texts)


def _task_text(messages: list[dict[str, Any]]) -> str:
    if len(messages) <= 2:
        return ""
    return _content_text(messages[2].get("content"))


def _summary_input(task: str, previous_summary: str, conversation: str) -> str:
    parts = [f"<task>\n{task}\n</task>"]
    if previous_summary:
        parts.append(f"<previous-checkpoint>\n{previous_summary}\n</previous-checkpoint>")
    parts.append(f"<old-work>\n{conversation}\n</old-work>")
    return "\n\n".join(parts)


def _checkpoint_content(
    summary: str,
    modified_files: list[str],
    inspected_images: list[str],
) -> str:
    sections = [summary.strip()]
    if modified_files:
        sections.append(f"<modified-files>\n{chr(10).join(modified_files)}\n</modified-files>")
    if inspected_images:
        sections.append(
            f"<inspected-images>\n{chr(10).join(inspected_images)}\n</inspected-images>"
        )
    body = "\n\n".join(section for section in sections if section)
    return f"<compaction-checkpoint>\n{body}\n</compaction-checkpoint>"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[... {omitted} more characters omitted]"
