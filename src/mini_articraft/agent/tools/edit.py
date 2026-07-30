from __future__ import annotations

from typing import Any

from mini_articraft.agent.tools._core import Tool, ToolContext, display_path, schema, workspace_path


async def run(context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path = workspace_path(context.workspace, str(args["path"]))
    old_text = str(args["old_text"])
    new_text = str(args["new_text"])
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text matched {count} times")
    path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return {"path": display_path(context.workspace, path), "replaced": 1}


TOOL = Tool(
    "edit",
    schema(
        "edit",
        "Edit one workspace file by replacing an exact text block. The old text must appear exactly once; use read first if you need current context.",
        {
            "path": {
                "type": "string",
                "description": "Path to the file inside the run workspace. Relative paths are resolved against the workspace.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to replace. The call fails unless this text appears exactly once.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text to write in place of old_text.",
            },
        },
        ["path", "old_text", "new_text"],
    ),
    run,
)
