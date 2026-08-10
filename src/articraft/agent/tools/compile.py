from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

from articraft.agent.tools._core import Tool, ToolContext, schema, workspace_digest
from articraft.compiler.feedback import compile_failure_signature, render_compile_report
from articraft.compiler.result import CompilePayload

_AGENT_FACING_KEYS = frozenset(
    {
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
    }
)


async def run(context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if args:
        raise ValueError("compile does not take arguments")
    if context.refresh_compile_freshness() and context.successful_compile_result is not None:
        context.last_compile_failure_signature = None
        context.consecutive_compile_failures = 0
        return _compact_result(context.successful_compile_result)

    result = await _compile_path(context)
    internal_result = _internal_result(context, result)
    context.compile_result = internal_result
    if result["status"] == "success":
        context.successful_compile_result = internal_result
        context.successful_compile_digest = workspace_digest(context.workspace)
    return _compact_result(internal_result)


async def _compile_path(context: ToolContext) -> CompilePayload:
    """Finish a started compile before propagating task cancellation.

    ``asyncio.to_thread`` cannot stop its worker. Shielding and joining it keeps
    a cancelled generation from leaving a compiler mutating the run after the
    task has finished.
    """
    task = asyncio.create_task(asyncio.to_thread(context.compiler.compile_path, context.run_dir))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except BaseException:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result


def _internal_result(context: ToolContext, result: CompilePayload) -> CompilePayload:
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
    # Iterate the payload rather than index it: dynamic keys cannot be spelled
    # as a TypedDict literal, and this way no optional key is accessed.
    return cast(CompilePayload, {k: v for k, v in result.items() if k in _AGENT_FACING_KEYS})


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
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
