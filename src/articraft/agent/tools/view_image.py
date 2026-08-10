from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from articraft.agent.images import LIMITS, ImageDetail, prepare_image
from articraft.agent.tools._core import (
    Tool,
    ToolContext,
    ToolResult,
    display_path,
    readable_path,
    schema,
)


async def run(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    detail = cast(ImageDetail, str(args.get("detail") or "original"))
    if detail not in LIMITS:
        raise ValueError("detail must be high or original")

    requested_path = readable_path(context.workspace, str(args["path"]))
    raster_path = _raster_path(requested_path)
    image = prepare_image(raster_path, detail=detail)
    requested_label = display_path(context.workspace, requested_path)
    raster_label = display_path(context.workspace, raster_path)
    result = image.metadata(requested_label)
    if raster_path != requested_path:
        result["raster_path"] = raster_label
    return ToolResult(result, [image.content_item()])


def _raster_path(path: Path) -> Path:
    if path.suffix.lower() != ".svg":
        return path
    companion = path.with_suffix(".svg.webp")
    if not companion.is_file():
        raise ValueError(f"SVG preview is missing: {companion.name}")
    return companion


TOOL = Tool(
    "view_image",
    schema(
        "view_image",
        "View a local raster image or a vendored SDK SVG preview as image input. Defaults to bounded original detail.",
        {
            "path": {
                "type": "string",
                "description": "Path inside the run workspace or read-only docs/sdk tree.",
            },
            "detail": {
                "type": "string",
                "enum": ["high", "original"],
                "description": "Image detail level. Defaults to original.",
            },
        },
        ["path"],
    ),
    run,
)
