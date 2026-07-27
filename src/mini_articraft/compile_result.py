from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal

CompileStatus = Literal["success", "error"]


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
        include_report: bool = False,
        include_returncode: bool = False,
        returncode: int | None = None,
    ) -> dict[str, Any]:
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        if include_returncode:
            payload["returncode"] = returncode
        if self.compile_stats is None:
            payload.pop("compile_stats")
        if include_report:
            from mini_articraft.compile_feedback import build_compile_report_from_payload

            payload["compile_report"] = build_compile_report_from_payload(payload)
        return payload
