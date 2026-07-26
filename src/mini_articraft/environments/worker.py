from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import sys
import time
import traceback
from collections.abc import Hashable, Iterable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TypeVar

from mini_articraft.compile_result import CompileResult
from mini_articraft.sdk import ArticulatedObject, TestContext, TestReport
from mini_articraft.sdk.export import export_object

T = TypeVar("T", bound=Hashable)
_COMPILE_PROGRESS_FILE = ".compile-progress.json"


class _CompileTracker:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()
        self.current_phase = "starting the compile worker"
        self.phase_started = self.started
        self.phases: dict[str, float] = {}
        self.model: dict[str, int] = {}
        self._write()

    @contextlib.contextmanager
    def phase(self, name: str):
        self.current_phase = name
        self.phase_started = time.perf_counter()
        self._write()
        try:
            yield
        finally:
            self.phases[name] = round(time.perf_counter() - self.phase_started, 4)
            self._write()

    def set_model(self, obj: ArticulatedObject) -> None:
        self.model = {
            "parts": len(obj.parts),
            "shapes": sum(1 for part in obj.parts for _shape in part._iter_shapes()),
            "articulations": len(obj.articulations),
        }
        self._write()

    def finish(self) -> dict[str, Any]:
        self.current_phase = "finishing the compile"
        self.phase_started = time.perf_counter()
        self._write()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        now = time.perf_counter()
        return {
            "total_seconds": round(now - self.started, 4),
            "current_phase": self.current_phase,
            "current_phase_seconds": round(now - self.phase_started, 4),
            "phases": dict(self.phases),
            "model": dict(self.model),
        }

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.snapshot()), encoding="utf-8")


def compile_run(run_dir: Path, *, include_report: bool = True) -> dict[str, Any]:
    workspace = run_dir / "workspace"
    result_dir = run_dir / "result"
    tracker = _CompileTracker(result_dir / _COMPILE_PROGRESS_FILE)
    result = _compile_workspace(workspace, result_dir, tracker=tracker)
    result.compile_stats = tracker.finish()
    tracker.remove()
    return result.to_payload(include_report=include_report)


def _compile_workspace(
    workspace: Path,
    export_dir: Path,
    *,
    tracker: _CompileTracker,
) -> CompileResult:
    export_dir.mkdir(parents=True, exist_ok=True)

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    result = CompileResult()

    previous_cwd = Path.cwd()
    sys.path.insert(0, str(workspace))
    try:
        os.chdir(workspace)
        with (
            contextlib.redirect_stdout(captured_stdout),
            contextlib.redirect_stderr(captured_stderr),
        ):
            with tracker.phase("loading main.py and building the model"):
                globals_dict = runpy.run_path(str(workspace / "main.py"), run_name="__main__")
            object_model = globals_dict.get("object_model")
            if not isinstance(object_model, ArticulatedObject):
                raise TypeError("main.py must define object_model as an ArticulatedObject")
            tracker.set_model(object_model)
            with tracker.phase("running authored tests"):
                authored_report = _run_required_tests(globals_dict)
            baseline_report = _run_baseline_tests(object_model, authored_report, tracker=tracker)
            test_report = _merge_test_reports(authored_report, baseline_report)
            result.test_report = _serialize_test_report(
                test_report,
                compiler_failure_names={failure.name for failure in baseline_report.failures},
            )
            with tracker.phase("exporting and validating the USDZ file"):
                export_result = export_object(object_model, export_dir)
            result.manifest = str(export_result.manifest)
            result.usdz = str(export_result.usdz)
            _raise_for_failed_test_report(test_report)

        result.status = "success"
    except BaseException as exc:
        test_report = exc.test_report if isinstance(exc, TestReportError) else None
        serialized_test_report = result.test_report
        if serialized_test_report is None and isinstance(test_report, TestReport):
            serialized_test_report = _serialize_test_report(test_report)
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        result.traceback = traceback.format_exc()
        result.test_report = serialized_test_report
    finally:
        os.chdir(previous_cwd)
        if sys.path and sys.path[0] == str(workspace):
            sys.path.pop(0)

    result.stdout = captured_stdout.getvalue()
    result.stderr = captured_stderr.getvalue()
    return result


def _run_required_tests(globals_dict: dict[str, Any]) -> TestReport:
    run_tests = globals_dict.get("run_tests")
    if not callable(run_tests):
        raise ValueError(
            "Missing required `run_tests()` in main.py. "
            "Add a top-level `def run_tests() -> TestReport:` and return `ctx.report()`."
        )

    report = run_tests()
    if not isinstance(report, TestReport):
        raise ValueError(f"run_tests() must return TestReport (got {type(report).__name__})")
    return report


def _run_baseline_tests(
    obj: ArticulatedObject,
    authored_report: TestReport,
    *,
    tracker: _CompileTracker,
) -> TestReport:
    ctx = TestContext(obj)
    for part_name in authored_report.allowed_isolated_parts:
        ctx.allow_isolated_part(
            part_name,
            reason="carried over from authored run_tests() allowance",
        )
    for overlap in authored_report.allowed_overlaps:
        ctx.allow_overlap(
            overlap.part_a,
            overlap.part_b,
            reason=overlap.reason,
            shape_a=overlap.shape_a,
            shape_b=overlap.shape_b,
        )

    with tracker.phase("checking the model structure"):
        ctx.check_model_valid()
        ctx.check_single_root_part()
    preliminary = ctx.report()
    if not preliminary.passed:
        return _without_allowance_notes(preliminary)

    with tracker.phase("checking for isolated parts"):
        ctx.fail_if_isolated_parts()
    with tracker.phase("checking for disconnected geometry"):
        ctx.warn_if_part_contains_disconnected_geometry_islands()
    with tracker.phase("checking the model scale"):
        ctx.warn_if_absurd_dimensions()
    with tracker.phase("checking for part overlaps"):
        ctx.fail_if_parts_overlap_in_current_pose()
    with tracker.phase("checking articulation motion"):
        ctx.fail_if_articulation_separates_child()
    return _without_allowance_notes(ctx.report())


def _without_allowance_notes(report: TestReport) -> TestReport:
    """Strip baseline-only allowance bookkeeping before reports are merged."""
    return replace(
        report,
        allowances=(),
        allowed_isolated_parts=(),
        allowed_overlaps=(),
    )


def _merge_test_reports(authored_report: TestReport, baseline_report: TestReport) -> TestReport:
    checks = _ordered_unique([*authored_report.checks, *baseline_report.checks])
    baseline_failure_names = {failure.name for failure in baseline_report.failures}
    failures = _ordered_unique(
        [
            *(
                failure
                for failure in authored_report.failures
                if failure.name not in baseline_failure_names
            ),
            *baseline_report.failures,
        ]
    )
    warnings = _ordered_unique([*authored_report.warnings, *baseline_report.warnings])
    allowances = _ordered_unique(authored_report.allowances)

    return TestReport(
        passed=not failures,
        checks_run=len(checks),
        checks=checks,
        failures=failures,
        warnings=warnings,
        allowances=allowances,
        allowed_isolated_parts=authored_report.allowed_isolated_parts,
        allowed_overlaps=authored_report.allowed_overlaps,
    )


def _ordered_unique(items: Iterable[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(items))


class TestReportError(ValueError):
    """Raised when SDK checks fail; carries the failed report for the payload."""

    def __init__(self, report: TestReport) -> None:
        self.test_report = report
        lines = ["SDK tests failed:"]
        lines.extend(f"- {failure.name}: {failure.details}" for failure in report.failures[:10])
        if len(report.failures) > 10:
            lines.append(f"... ({len(report.failures) - 10} more)")
        super().__init__("\n".join(lines))


def _raise_for_failed_test_report(report: TestReport) -> None:
    if not report.passed:
        raise TestReportError(report)


def _serialize_test_report(
    report: TestReport | None,
    *,
    compiler_failure_names: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    if report is None:
        return None
    serialized = asdict(report)
    for failure, serialized_failure in zip(
        report.failures,
        serialized["failures"],
        strict=True,
    ):
        serialized_failure["source"] = (
            "compiler" if failure.name in compiler_failure_names else "tests"
        )
    return serialized


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    raw = bool(args and args[0] == "--raw")
    if raw:
        args.pop(0)
    if len(args) != 1:
        payload = CompileResult(error="Usage: mini-articraft-compile-run <run_dir>").to_payload(
            include_report=not raw
        )
        print(json.dumps(payload))
        return 2

    payload = compile_run(Path(args[0]).resolve(), include_report=not raw)
    print(json.dumps(payload))
    return 0 if payload["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
