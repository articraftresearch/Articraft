from __future__ import annotations

import asyncio
from typing import Any

from mini_articraft.agent.tools._core import Tool, ToolContext, schema, workspace_digest
from mini_articraft.compiler.feedback import compile_failure_signature, render_compile_report


async def run(context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if args:
        raise ValueError("compile does not take arguments")
    if context.refresh_compile_freshness() and context.successful_compile_result is not None:
        context.last_compile_failure_signature = None
        context.consecutive_compile_failures = 0
        return _compact_result(context.successful_compile_result)

    result = await asyncio.to_thread(context.env.compile_path, context.run_dir)
    internal_result = _internal_result(context, result)
    context.compile_result = internal_result
    if result["status"] == "success":
        context.successful_compile_result = internal_result
        context.successful_compile_digest = workspace_digest(context.workspace)
    return _compact_result(internal_result)


def _internal_result(context: ToolContext, result: dict[str, Any]) -> dict[str, Any]:
    compile_report = result.get("compile_report")
    if isinstance(compile_report, dict):
        signature = compile_failure_signature(compile_report)
        if signature is None:
            context.last_compile_failure_signature = None
            context.consecutive_compile_failures = 0
            result["compile_report"] = render_compile_report(compile_report)
        else:
            repeated = signature == context.last_compile_failure_signature
            context.last_compile_failure_signature = signature
            context.consecutive_compile_failures += 1
            result["compile_report"] = render_compile_report(
                compile_report,
                repeated=repeated,
                failure_streak=context.consecutive_compile_failures,
            )
    return {
        key: result[key]
        for key in (
            "status",
            "manifest",
            "usdz",
            "error",
            "stdout",
            "stderr",
            "traceback",
            "returncode",
            "compile_stats",
            "test_report",
            "compile_report",
        )
        if key in result
    }


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    report = result.get("compile_report")
    signals = report.get("signals_text") if isinstance(report, dict) else None
    if not isinstance(signals, str):
        raise ValueError("compile result is missing rendered compile signals")
    return {"status": result["status"], "compile_signals": signals}


TOOL = Tool(
    "compile",
    schema(
        "compile",
        "Compile the current workspace. The result contains only status and one structured "
        "<compile_signals> block. It does not render, copy, or attach images. Registered visual "
        "signals contain workspace paths that must be opened with view_image. Run this after "
        "changing files and before the final response.",
        {},
        [],
    ),
    run,
)
