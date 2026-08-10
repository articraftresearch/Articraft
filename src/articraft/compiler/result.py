from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

CompileStatus = Literal["success", "error"]


class _CompilePayloadBase(TypedDict):
    """Written by every compile, whatever the outcome.

    ``CompileResult.to_payload`` always sets these, so reading them needs no
    guard. Everything below is conditional and does.
    """

    status: CompileStatus
    manifest: str
    usdz: str
    test_report: Any
    stdout: str
    stderr: str
    error: str
    traceback: str


class CompilePayload(_CompilePayloadBase, total=False):
    """What one compile attempt looks like on the wire.

    The compiler produces the required fields above. The workspace attaches
    process details and the report after it collects the subprocess.
    """

    compile_stats: dict[str, Any]

    # Attached by the workspace once the subprocess is collected.
    returncode: int | None
    # A dict on the wire and in the record; the agent's compile tool replaces it
    # with the rendered text before the model sees it. Any, because this type
    # exists to check key names -- the nested reports are their own shapes.
    compile_report: Any


@dataclass
class CompileResult:
    """One compile attempt before transport-specific fields are attached."""

    status: CompileStatus = "error"
    manifest: str = ""
    usdz: str = ""
    test_report: object = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    traceback: str = ""
    compile_stats: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CompileResult:
        compile_stats = payload.get("compile_stats")
        return cls(
            status="success" if payload.get("status") == "success" else "error",
            manifest=str(payload.get("manifest") or ""),
            usdz=str(payload.get("usdz") or ""),
            test_report=payload.get("test_report"),
            stdout=str(payload.get("stdout") or ""),
            stderr=str(payload.get("stderr") or ""),
            error=str(payload.get("error") or ""),
            traceback=str(payload.get("traceback") or ""),
            compile_stats=dict(compile_stats) if isinstance(compile_stats, dict) else None,
        )

    def to_payload(
        self,
        *,
        include_returncode: bool = False,
        returncode: int | None = None,
    ) -> CompilePayload:
        payload: CompilePayload = {
            "status": self.status,
            "manifest": self.manifest,
            "usdz": self.usdz,
            "test_report": self.test_report,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "traceback": self.traceback,
        }
        if self.compile_stats is not None:
            payload["compile_stats"] = self.compile_stats
        if include_returncode:
            payload["returncode"] = returncode
        return payload
