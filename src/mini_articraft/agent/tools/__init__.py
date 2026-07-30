from __future__ import annotations

from typing import Any, Literal, overload

from mini_articraft.agent.tools._core import Tool, ToolContext, ToolResult, result_item
from mini_articraft.agent.tools.compile import TOOL as compile_tool
from mini_articraft.agent.tools.edit import TOOL as edit_tool
from mini_articraft.agent.tools.exec_command import TOOL as exec_command_tool
from mini_articraft.agent.tools.read import TOOL as read_tool
from mini_articraft.agent.tools.view_image import TOOL as view_image_tool
from mini_articraft.agent.tools.write import TOOL as write_tool
from mini_articraft.agent.tools.write_stdin import TOOL as write_stdin_tool

TOOLS: dict[str, Tool[Any]] = {
    tool.name: tool
    for tool in (
        read_tool,
        view_image_tool,
        edit_tool,
        write_tool,
        exec_command_tool,
        write_stdin_tool,
        compile_tool,
    )
}


def schemas(*, include_images: bool = True) -> list[dict[str, Any]]:
    return [tool.schema for tool in TOOLS.values() if include_images or tool.name != "view_image"]


@overload
def get(name: Literal["view_image"]) -> Tool[ToolResult]: ...


@overload
def get(
    name: Literal["read", "edit", "write", "exec_command", "write_stdin", "compile"],
) -> Tool[dict[str, Any]]: ...


@overload
def get(name: str) -> Tool[Any]: ...


def get(name: str) -> Tool[Any]:
    try:
        return TOOLS[name]
    except KeyError as exc:
        raise ValueError(f"unknown tool: {name}") from exc


__all__ = ["Tool", "ToolContext", "ToolResult", "get", "result_item", "schemas"]
