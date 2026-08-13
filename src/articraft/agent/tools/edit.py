from __future__ import annotations

from itertools import pairwise
from typing import Any

from articraft.agent.tools._core import Tool, ToolContext, display_path, schema, workspace_path


def _replacement_args(args: dict[str, Any]) -> list[tuple[str, str]]:
    raw = args.get("edits")
    if (
        raw is None
        and isinstance(args.get("old_text"), str)
        and isinstance(args.get("new_text"), str)
    ):
        raw = [{"old_text": args["old_text"], "new_text": args["new_text"]}]
    if not isinstance(raw, list) or not raw:
        raise ValueError("edits must contain at least one replacement")
    replacements: list[tuple[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"edits[{index}] must be an object")
        old_text = item.get("old_text")
        new_text = item.get("new_text")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ValueError(f"edits[{index}] must contain string old_text and new_text values")
        if not old_text:
            raise ValueError(f"edits[{index}].old_text must not be empty")
        if old_text == new_text:
            raise ValueError(f"edits[{index}] would not change the file")
        replacements.append((old_text, new_text))
    return replacements


async def run(context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path = workspace_path(context.workspace, str(args["path"]))
    text = path.read_text(encoding="utf-8")
    matches: list[tuple[int, int, int, str]] = []
    replacements = _replacement_args(args)
    for index, (old_text, new_text) in enumerate(replacements):
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"edits[{index}].old_text matched {count} times")
        start = text.index(old_text)
        matches.append((start, start + len(old_text), index, new_text))
    matches.sort()
    for previous, current in pairwise(matches):
        if previous[1] > current[0]:
            raise ValueError(
                f"edits[{previous[2]}] and edits[{current[2]}] overlap; merge them into one edit"
            )
    updated = text
    for start, end, _index, new_text in reversed(matches):
        updated = updated[:start] + new_text + updated[end:]
    path.write_text(updated, encoding="utf-8")
    return {"path": display_path(context.workspace, path), "replaced": len(replacements)}


TOOL = Tool(
    "edit",
    schema(
        "edit",
        "Edit one workspace file with one or more exact replacements. Every old text must appear exactly once in the original file. The whole call fails without writing if any replacement is invalid.",
        {
            "path": {
                "type": "string",
                "description": "Path to the file inside the run workspace. Relative paths are resolved against the workspace.",
            },
            "edits": {
                "type": "array",
                "minItems": 1,
                "description": "Targeted replacements matched against the original file. Use one call for disjoint changes. Merge nearby or overlapping changes into one replacement.",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_text": {
                            "type": "string",
                            "description": "Exact text to replace. It must appear exactly once in the original file.",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text to write in place of old_text.",
                        },
                    },
                    "required": ["old_text", "new_text"],
                    "additionalProperties": False,
                },
            },
        },
        ["path", "edits"],
    ),
    run,
)
