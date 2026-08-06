from __future__ import annotations

from typing import Any

from mini_articraft.agent.tools._core import Tool, ToolContext, display_path, schema, workspace_path


async def run(context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path = workspace_path(context.workspace, str(args["path"]))
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "path": display_path(context.workspace, path),
        "bytes": len(content.encode("utf-8")),
    }


TOOL = Tool(
    "write",
    schema(
        "write",
        "Create or overwrite a file in the run workspace. Parent directories are created automatically.",
        {
            "path": {
                "type": "string",
                "description": "Path to create or overwrite inside the run workspace. Relative paths are resolved against the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Complete UTF-8 file contents to write.",
            },
        },
        ["path", "content"],
    ),
    run,
)
