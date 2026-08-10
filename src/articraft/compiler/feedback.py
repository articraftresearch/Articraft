from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from articraft.compiler.result import CompilePayload
from articraft.sdk import FailureKind, TestReport

Severity = Literal["failure", "warning", "note"]
Status = Literal["success", "failure"]
SignalGroup = Literal["build", "qc", "design", "hygiene"]


@dataclass(frozen=True)
class CompileSignal:
    severity: Severity
    kind: str
    code: str
    summary: str
    details: str = ""
    blocking: bool = False
    source: str = "compiler"
    group: SignalGroup = "qc"
    check_name: str | None = None
    dedupe_key: str = ""

    def __post_init__(self) -> None:
        if self.dedupe_key:
            return
        raw = "\n".join(
            [
                self.severity,
                self.kind,
                self.code,
                self.summary,
                self.details,
                str(self.blocking),
                self.source,
                self.group,
                self.check_name or "",
            ]
        )
        object.__setattr__(self, "dedupe_key", hashlib.sha1(raw.encode()).hexdigest())


@dataclass(frozen=True)
class CompileSignalBundle:
    status: Status
    summary: str
    signals: tuple[CompileSignal, ...] = ()


def with_compile_report(payload: CompilePayload) -> CompilePayload:
    """Attach the agent-facing report to a compile payload.

    Callers add this rather than ``CompileResult`` building its own report: the
    result is data, this is presentation, and the dependency only runs one way.
    """

    payload["compile_report"] = build_compile_report_from_payload(payload)
    return payload


def build_compile_report_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    status: Status = "success" if payload["status"] == "success" else "failure"
    return build_compile_report(
        status=status,
        error=str(payload.get("error") or ""),
        traceback_text=str(payload.get("traceback") or ""),
        stdout=str(payload.get("stdout") or ""),
        stderr=str(payload.get("stderr") or ""),
        returncode=payload.get("returncode"),
        test_report=payload.get("test_report"),
    )


def build_compile_report(
    *,
    status: Status,
    error: str = "",
    traceback_text: str = "",
    stdout: str = "",
    stderr: str = "",
    returncode: object = None,
    test_report: object = None,
) -> dict[str, Any]:
    report = _report_dict(test_report)
    bundle = _bundle(status, error, report, traceback_text=traceback_text)
    return {
        "status": status,
        "error": error,
        "traceback": traceback_text,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "counts": _counts(bundle.signals),
        "test_report": report,
        "signal_bundle": asdict(bundle),
        "signals_text": render_compile_signals(bundle),
    }


def render_compile_report(
    report: dict[str, Any],
    *,
    repeated: bool = False,
    failure_streak: int = 0,
) -> dict[str, Any]:
    rendered = dict(report)
    rendered["signals_text"] = render_compile_signals(
        _bundle_from_dict(report["signal_bundle"]),
        repeated=repeated,
        failure_streak=failure_streak,
    )
    return rendered


def compile_failure_signature(report: dict[str, Any]) -> str | None:
    bundle = report["signal_bundle"]
    failures = [signal for signal in bundle["signals"] if signal["severity"] == "failure"]
    if not failures:
        return None
    failures.sort(
        key=lambda signal: (
            signal.get("kind", ""),
            signal.get("source", ""),
            signal.get("check_name") or "",
            signal.get("summary", ""),
            signal.get("details", ""),
        )
    )
    raw = json.dumps(failures, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(raw).hexdigest()


def render_compile_signals(
    bundle: CompileSignalBundle,
    *,
    repeated: bool = False,
    failure_streak: int = 0,
) -> str:
    failures = _failures(bundle.signals)
    warnings = _ordered_signals(bundle.signals, "warning")
    notes = _ordered_signals(bundle.signals, "note")
    summary = bundle.summary
    if repeated and failures:
        summary += "\nThis failure matches the previous compile attempt."
    if failure_streak >= 3 and failures:
        summary += f"\nThis is compile failure {failure_streak} in a row."

    parts = ["<compile_signals>", "<summary>", summary, "</summary>"]
    _add_section(parts, "failures", "Failures (blocking):", failures)
    _add_section(parts, "warnings", "Warnings (non-blocking):", warnings)
    if notes:
        _add_section(parts, "notes", "Notes (informational):", notes)
    rules = _rules(
        failures,
        has_warnings=bool(warnings),
        has_visuals=any(signal.kind == "qa_image" for signal in bundle.signals),
        repeated=repeated,
        failure_streak=failure_streak,
    )
    if rules:
        parts += [
            "",
            "<response_rules>",
            "Suggested next steps:\n" + "\n".join(rules),
            "</response_rules>",
        ]
    parts.append("</compile_signals>")
    return "\n".join(parts)


def _bundle(
    status: Status,
    error: str,
    report: dict[str, Any] | None,
    *,
    traceback_text: str = "",
) -> CompileSignalBundle:
    signals = _signals_from_report(report) if report else []
    if status == "failure" and not _failures(signals):
        signals.append(_runtime_signal(error, traceback_text=traceback_text))
    signals = sorted(
        {signal.dedupe_key: signal for signal in signals}.values(),
        key=_signal_sort_key,
    )
    return CompileSignalBundle(status, _summary(status, signals), tuple(signals))


def _signals_from_report(report: dict[str, Any]) -> list[CompileSignal]:
    signals = [
        _warning_signal(str(item)) for item in report.get("warnings", []) if str(item).strip()
    ]
    signals += [_allowance_signal(str(a)) for a in report.get("allowances", []) if str(a).strip()]
    signals += [_failure_signal(f) for f in report.get("failures", [])]
    signals += [
        _artifact_signal(artifact)
        for artifact in report.get("artifacts", [])
        if isinstance(artifact, dict)
        and artifact.get("kind") == "image"
        and _safe_workspace_artifact_path(str(artifact.get("path") or ""))
    ]
    return signals


def _safe_workspace_artifact_path(value: str) -> bool:
    value = value.strip()
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def _artifact_signal(artifact: dict[str, Any]) -> CompileSignal:
    name = str(artifact.get("name") or "visual evidence")
    path = str(artifact.get("path") or "")
    caption = str(artifact.get("caption") or "")
    details = f"workspace_path={path}"
    if caption:
        details += f"\n{caption}"
    return CompileSignal(
        "note",
        "qa_image",
        "QA_IMAGE",
        f"Visual evidence is ready: {name}",
        details,
        source="tests",
        group="design",
    )


def _warning_signal(text: str) -> CompileSignal:
    stripped = text.strip()
    lower = stripped.lower()
    headline = stripped.splitlines()[0]
    if any(
        marker in lower
        for marker in (
            "non-finite or absurd geometry dimensions",
            "geometry outlier dimensions",
            "geometry scale",
            "scale warning",
        )
    ):
        return CompileSignal(
            "warning",
            "geometry_scale",
            "WARN_GEOMETRY_SCALE",
            headline,
            stripped,
            source="compiler",
            group="design",
        )
    if lower.startswith("overlaps detected but allowed by justification"):
        return CompileSignal(
            "note",
            "allowed_overlap",
            "NOTE_ALLOWED_OVERLAP",
            "Overlap findings were allowed by justification.",
            stripped,
            source="tests",
            group="qc",
        )
    if lower.startswith("isolated parts detected but allowed by justification"):
        return CompileSignal(
            "note",
            "allowed_isolated_part",
            "NOTE_ALLOWED_ISOLATED_PART",
            "Isolated-part findings were allowed by justification.",
            stripped,
            source="tests",
            group="design",
        )
    if "disconnected geometry" in lower or lower.startswith(
        "warn_if_part_contains_disconnected_geometry_islands("
    ):
        return CompileSignal(
            "warning",
            "disconnected_geometry",
            "WARN_DISCONNECTED_GEOMETRY",
            "A part contains disconnected geometry. Move or extend the disconnected piece "
            "so its OWN body overlaps the nearest piece by a few mm -- overlap within a "
            "part is free and counts as connected. Do not add a separate piece to bridge "
            "the gap. To then smooth or trim that overlap, `weld(...)` blends it and "
            "boolean_difference trims it; if a protrusion fragmented into slivers, it was "
            "a boolean_difference against a thin hollow shell, so subtract a solid form of "
            "the body's outer surface instead.",
            stripped,
            source="tests",
            group="qc",
        )
    return CompileSignal(
        "warning",
        "test_warning",
        "TEST_WARNING",
        headline,
        stripped,
        source="tests",
        group="qc",
    )


@dataclass(frozen=True)
class _FailureSpec:
    signal_kind: str
    code_suffix: str
    summary: str
    compiler_summary: str | None = None
    compiler_group: SignalGroup = "qc"


# Failure semantics have one owner. Provenance only selects the QC/TEST code
# prefix and the few compiler-specific presentation fields.
_FAILURE_SPECS: dict[FailureKind, _FailureSpec] = {
    FailureKind.MODEL_VALIDITY: _FailureSpec(
        "model_validity",
        "MODEL_VALIDITY",
        "Model validation failed.",
        compiler_summary="Compiler model validation failed.",
        compiler_group="build",
    ),
    FailureKind.SINGLE_ROOT: _FailureSpec(
        "single_root_policy",
        "SINGLE_ROOT_POLICY",
        "The model must have exactly one root part.",
        compiler_group="build",
    ),
    FailureKind.MESH_HEALTH: _FailureSpec(
        "mesh_health",
        "MESH_HEALTH",
        "Unhealthy mesh geometry was found.",
        compiler_summary="Compiler mesh health validation failed.",
    ),
    FailureKind.MISSING_MASS: _FailureSpec(
        "missing_mass",
        "MISSING_MASS",
        "A part has no mass properties while physics is enabled.",
        compiler_group="build",
    ),
    FailureKind.ISOLATED_PART: _FailureSpec(
        "isolated_part",
        "ISOLATED_PART",
        "Floating or disconnected parts were found.",
    ),
    FailureKind.DISCONNECTED_GEOMETRY: _FailureSpec(
        "disconnected_geometry",
        "DISCONNECTED_GEOMETRY",
        "A part contains disconnected geometry islands.",
    ),
    FailureKind.OVERLAP: _FailureSpec(
        "real_overlap",
        "REAL_OVERLAP",
        "A mesh check found overlapping parts.",
    ),
    FailureKind.CONTACT: _FailureSpec(
        "exact_contact_gap",
        "EXACT_CONTACT_GAP",
        "A contact check found a gap where contact was expected.",
    ),
    FailureKind.ARTICULATION_SEPARATION: _FailureSpec(
        "articulation_separation",
        "ARTICULATION_SEPARATION",
        "A hinge or pivot pulls its child away from its parent during motion.",
    ),
    FailureKind.AUTHORED: _FailureSpec(
        "test_failure",
        "FAILURE",
        "",
    ),
}


def _failure_signal(failure: dict[str, Any]) -> CompileSignal:
    name = str(failure["name"])
    details = str(failure.get("details") or "")
    try:
        kind = FailureKind(str(failure.get("kind") or FailureKind.AUTHORED))
    except ValueError:
        kind = FailureKind.AUTHORED
    source = str(failure.get("source") or "tests")
    if source not in {"compiler", "tests"}:
        source = "tests"
    spec = _FAILURE_SPECS[kind]
    compiler_owned = source == "compiler"
    prefix = "QC" if compiler_owned else "TEST"
    if kind is FailureKind.AUTHORED:
        owner = "Compiler check" if compiler_owned else "Authored test"
        summary = f"{owner} failed: {name}"
    else:
        summary = (
            spec.compiler_summary if compiler_owned and spec.compiler_summary else spec.summary
        )
    group = spec.compiler_group if compiler_owned else "qc"
    return _failure(
        spec.signal_kind,
        f"{prefix}_{spec.code_suffix}",
        summary,
        name,
        details,
        source=source,
        group=group,
    )


def _failure(
    kind: str,
    code: str,
    summary: str,
    check_name: str,
    details: str,
    *,
    source: str = "tests",
    group: SignalGroup = "qc",
) -> CompileSignal:
    return CompileSignal(
        "failure",
        kind,
        code,
        summary,
        details,
        blocking=True,
        source=source,
        group=group,
        check_name=check_name,
    )


def _allowance_signal(text: str) -> CompileSignal:
    if text.startswith("allow_mesh_issues("):
        kind, code, summary = (
            "allowed_mesh_issues",
            "NOTE_ALLOWED_MESH_ISSUES",
            "Mesh health allowance declared.",
        )
    elif text.startswith("allow_overlap("):
        kind, code, summary = (
            "allowed_overlap",
            "NOTE_ALLOWED_OVERLAP",
            "Overlap allowance declared.",
        )
    elif text.startswith("allow_isolated_part("):
        kind, code, summary = (
            "allowed_isolated_part",
            "NOTE_ALLOWED_ISOLATED_PART",
            "Isolated-part allowance declared.",
        )
    else:
        kind, code, summary = "allowance", "ALLOWANCE", text
    group: SignalGroup = "design" if kind == "allowed_isolated_part" else "qc"
    return CompileSignal(
        "note",
        kind,
        code,
        summary,
        text,
        source="tests",
        group=group,
    )


def _runtime_signal(error: str, *, traceback_text: str = "") -> CompileSignal:
    text = error.strip() or "Compile execution failed."
    details = _runtime_details(text, traceback_text)
    lower = text.lower()
    if "compile timed out after" in lower:
        return CompileSignal(
            "failure",
            "compile_timeout",
            "COMPILE_TIMEOUT",
            text,
            "The compile worker and its child processes were stopped.",
            blocking=True,
            group="build",
        )
    if "unhealthy mesh geometry" in lower:
        return CompileSignal(
            "failure",
            "mesh_health",
            "COMPILE_MESH_HEALTH",
            "A mesh authoring operation produced unhealthy geometry.",
            details,
            blocking=True,
            group="qc",
        )
    if any(
        marker in lower
        for marker in (
            "unknown shape",
            "missing named geometry",
            "missing exact geometry",
        )
    ):
        return CompileSignal(
            "failure",
            "missing_exact_geometry",
            "TEST_MISSING_EXACT_GEOMETRY",
            "A check references named geometry that is not present.",
            details,
            blocking=True,
            source="tests",
            group="qc",
        )
    if "missing required `run_tests()`" in lower:
        return CompileSignal(
            "failure",
            "missing_run_tests",
            "COMPILE_MISSING_RUN_TESTS",
            "main.py must define run_tests().",
            details,
            blocking=True,
            group="build",
        )
    if "run_tests() must return testreport" in lower:
        return CompileSignal(
            "failure",
            "invalid_run_tests_report",
            "COMPILE_INVALID_RUN_TESTS_REPORT",
            "run_tests() must return TestReport.",
            details,
            blocking=True,
            group="build",
        )
    location = _user_traceback_location(traceback_text)
    return CompileSignal(
        "failure",
        "compile_runtime",
        "COMPILE_RUNTIME_FAILURE",
        text,
        f"location={location}" if location is not None else "",
        blocking=True,
        group="build",
    )


def _runtime_details(error: str, traceback_text: str) -> str:
    location = _user_traceback_location(traceback_text)
    return error if location is None else f"{error}\nlocation={location}"


def _user_traceback_location(traceback_text: str) -> str | None:
    found: str | None = None
    pattern = re.compile(r'^\s*File "([^"]+)", line (\d+)(?:, in ([^\n]+))?', re.MULTILINE)
    for match in pattern.finditer(traceback_text):
        raw_path, line, function = match.groups()
        normalized = raw_path.replace("\\", "/")
        marker = "/workspace/"
        if marker in normalized:
            path = normalized.rsplit(marker, 1)[1]
        elif normalized == "main.py" or normalized.endswith("/main.py"):
            path = "main.py"
        else:
            continue
        found = f"{path}:{line}"
        if function:
            found += f" in {function.strip()}"
    return found


def _summary(status: Status, signals: list[CompileSignal]) -> str:
    counts = _counts(signals)
    header = f"status={status} failures={counts['failures']} warnings={counts['warnings']} notes={counts['notes']}"
    if first := next(iter(_failures(signals)), None):
        return f"{header}\nPrimary issue: {_primary_issue(first)}"
    if counts["warnings"]:
        return f"{header}\nPrimary issue: compile passed with warnings."
    return f"{header}\nCompile passed cleanly."


def _primary_issue(signal: CompileSignal) -> str:
    issues = {
        "compile_timeout": "the compile exceeded its time limit.",
        "missing_run_tests": "generated script is missing required run_tests().",
        "invalid_run_tests_report": "run_tests() returned the wrong type.",
        "single_root_policy": "compiler-owned root policy failed.",
        "model_validity": "compiler-owned model validation failed.",
        "mesh_health": "compiler-owned mesh health validation failed.",
        "isolated_part": (
            "compiler-owned connectivity checks found isolated parts."
            if signal.source == "compiler"
            else "an authored test found isolated parts."
        ),
        "real_overlap": "mesh checks found overlapping parts that need classification.",
        "missing_exact_geometry": "a check references missing named geometry.",
        "exact_contact_gap": "a contact check found separation where contact was expected.",
        "disconnected_geometry": "a part contains disconnected geometry islands.",
        "articulation_separation": "an articulation separates its child part during motion.",
        "test_failure": "a required authored test failed.",
    }
    return issues.get(signal.kind, signal.summary)


_RUNTIME_RULES = (
    "- Fix the compile or runtime error first. Geometry checks are blocked until the script runs.",
)
_STRUCTURE_RULES = ("- Fix the model structure first. Local geometry tuning comes after that.",)
_RULES_BY_KIND = {
    "missing_mass": (
        "- Say what each shape is made of: `material=Material.STEEL` on `part.add()`.",
        "- If the library has nothing close, build one: "
        '`Material(name="ceramic", density=2400.0)`.',
    ),
    "compile_timeout": (
        "- Inspect the phase named in the timeout before editing.",
        "- If model construction timed out, simplify the slow operation or use a coarser "
        "tolerance. If a compiler check timed out, reduce unnecessary shape count while "
        "preserving visible geometry.",
    ),
    **dict.fromkeys(
        (
            "compile_runtime",
            "missing_run_tests",
            "invalid_run_tests_report",
        ),
        _RUNTIME_RULES,
    ),
    **dict.fromkeys(("single_root_policy", "model_validity"), _STRUCTURE_RULES),
    "mesh_health": (
        "- Repair the named mesh issue at its source and compile again.",
        "- Use `allow_mesh_issues(...)` only when the exact issue on the exact named shape "
        "is intentional.",
    ),
    "isolated_part": (
        "- Repair the reported support path, or add a precise allowance when the separation "
        "is intentional.",
    ),
    "real_overlap": (
        "- Decide whether the reported overlap is intentional embedding or an unintended "
        "collision.",
        "- Preserve prompt-critical visible geometry while repairing or allowing the exact "
        "reported pair.",
    ),
    "missing_exact_geometry": (
        "- Restore the named geometry or update the dependent check in the same edit. This "
        "does not call for a broad geometry rewrite.",
    ),
    "exact_contact_gap": (
        "- This is a gap, not an overlap. Verify the tested pair, then repair its geometry "
        "or placement.",
    ),
    "disconnected_geometry": (
        "- Move or extend the disconnected piece so its own body overlaps the nearest piece "
        "by a few mm -- overlap within a part is free and counts as connected. Do not add a "
        "separate bridging piece.",
    ),
    "articulation_separation": (
        "- Place the articulation origin and axis so the child stays connected to the parent "
        "throughout the motion range.",
    ),
}
_DEFAULT_FAILURE_RULES = ("- Fix the named failing check before adding more geometry or tests.",)


def _rules(
    failures: list[CompileSignal],
    *,
    has_warnings: bool,
    has_visuals: bool,
    repeated: bool,
    failure_streak: int,
) -> list[str]:
    if not failures:
        rules: list[str] = []
        if has_visuals:
            rules.append(
                "- Open every relevant workspace image with `view_image` before you finish."
            )
        if has_warnings:
            rules.append(
                "- Warnings are design evidence. Do not remove or simplify required geometry "
                "only to clear them."
            )
        return rules
    kind = failures[0].kind
    if kind == "mesh_health" and failures[0].code == "COMPILE_MESH_HEALTH":
        rules = [
            "- Repair the source operation named in the error. Use the reported bounds to "
            "inspect the bad region.",
            "- Change the inputs or use a finer surface tolerance instead of accepting the "
            "broken result.",
        ]
    else:
        rules = list(_RULES_BY_KIND.get(kind, _DEFAULT_FAILURE_RULES))
    if failure_streak >= 3:
        rules.append(f"- The repair loop has continued. {_inspection_advice(kind)}")
    elif repeated:
        rules.append(f"- This failure repeated. {_inspection_advice(kind)}")
    if has_warnings:
        rules.append(
            "- Warnings are design evidence. Do not remove or simplify prompt-critical geometry just to clear them."
        )
    return rules


def _inspection_advice(kind: str) -> str:
    if kind == "compile_timeout":
        return "Inspect the named compile phase and the densest operation in that phase before editing."
    if kind in {"compile_runtime", "missing_run_tests", "invalid_run_tests_report"}:
        return "Use one short `exec_command` inspection of the exception location or API before editing."
    if kind in {"single_root_policy", "model_validity"}:
        return "Use one short `exec_command` inspection of the model structure before editing."
    return "Use one short `exec_command` inspection of the named geometry or support path before editing."


def _add_section(parts: list[str], tag: str, title: str, signals: list[CompileSignal]) -> None:
    if signals:
        parts += [
            "",
            f"<{tag}>",
            title,
            *[_signal_line(s) for s in signals],
            f"</{tag}>",
        ]


def _signal_line(signal: CompileSignal) -> str:
    line = f"- {signal.severity.upper()} [{signal.kind}] {signal.summary}"
    if signal.details:
        line += "\n" + "\n".join(
            f"  {part}" if part else "" for part in signal.details.splitlines()
        )
    return line


def _failures(
    signals: list[CompileSignal] | tuple[CompileSignal, ...],
) -> list[CompileSignal]:
    return _ordered_signals(signals, "failure")


def _ordered_signals(
    signals: list[CompileSignal] | tuple[CompileSignal, ...],
    severity: Severity,
) -> list[CompileSignal]:
    return sorted(
        (signal for signal in signals if signal.severity == severity),
        key=_signal_sort_key,
    )


def _signal_sort_key(signal: CompileSignal) -> tuple[int, int, str, str, str, str]:
    severity_priority = {"failure": 0, "warning": 1, "note": 2}
    failure_priority = {
        "compile_timeout": 0,
        "compile_runtime": 0,
        "missing_run_tests": 0,
        "invalid_run_tests_report": 0,
        "model_validity": 1,
        "single_root_policy": 1,
        "mesh_health": 2,
        "isolated_part": 2 if signal.source == "compiler" else 6,
        "disconnected_geometry": 2 if signal.source == "compiler" else 6,
        "articulation_separation": 2 if signal.source == "compiler" else 6,
        "real_overlap": 2 if signal.source == "compiler" else 5,
        "missing_exact_geometry": 3,
        "exact_contact_gap": 4,
        "test_failure": 7,
    }
    kind_priority = failure_priority.get(signal.kind, 20)
    return (
        severity_priority[signal.severity],
        kind_priority,
        signal.kind,
        signal.check_name or "",
        signal.summary,
        signal.details,
    )


def _counts(signals: list[CompileSignal] | tuple[CompileSignal, ...]) -> dict[str, int]:
    return {
        name: sum(1 for s in signals if s.severity == severity)
        for name, severity in (
            ("failures", "failure"),
            ("warnings", "warning"),
            ("notes", "note"),
        )
    }


def _bundle_from_dict(value: dict[str, Any]) -> CompileSignalBundle:
    return CompileSignalBundle(
        status=value["status"],
        summary=value["summary"],
        signals=tuple(CompileSignal(**signal) for signal in value["signals"]),
    )


def _report_dict(value: object) -> dict[str, Any] | None:
    if isinstance(value, TestReport):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return None
