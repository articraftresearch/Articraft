from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mini_articraft import package_dir
from mini_articraft._child_process import child_environment
from mini_articraft.compile_result import CompileResult
from mini_articraft.record import Record
from mini_articraft.settings import DEFAULT_COMPILE_TIMEOUT_SECONDS, DEFAULT_OUTPUT_DIR

_COMPILE_PROGRESS_FILE = ".compile-progress.json"


class LocalEnvironmentConfig(BaseModel):
    output_dir: Path = DEFAULT_OUTPUT_DIR
    timeout_seconds: float = Field(default=DEFAULT_COMPILE_TIMEOUT_SECONDS, gt=0.0)
    physics_enabled: bool = False


DEFAULT_MAIN_PY = """from build123d import Box

from mini_articraft.sdk import ArticulatedObject, Material, TestContext, TestReport


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("object")
    base = model.part("base")
    base.add(Box(0.2, 0.2, 0.1), name="body", material=Material.plastic((0.7, 0.7, 0.72)))
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    return ctx.report()
"""


class LocalEnvironment:
    def __init__(self, **kwargs: Any):
        self.config = LocalEnvironmentConfig(**kwargs)

    def create_run(self, run_id: str) -> Path:
        run_id = _validate_run_id(run_id)
        run_dir = self.config.output_dir / run_id
        if run_dir.exists():
            raise FileExistsError(f"run already exists: {run_id}")

        (run_dir / "workspace").mkdir(parents=True)
        (run_dir / "result").mkdir()
        _link_sdk_docs(run_dir / "workspace")
        run_dir.joinpath("workspace", "main.py").write_text(DEFAULT_MAIN_PY, encoding="utf-8")
        (run_dir / "conversation.jsonl").touch()
        Record(run_id=run_id).save(run_dir / "record.json")
        return run_dir

    def compile_path(self, run_dir: Path | str) -> dict[str, Any]:
        run_dir = Path(run_dir)
        workspace = run_dir / "workspace"

        if not (workspace / "main.py").is_file():
            result = _error_result(run_dir, error="workspace/main.py is required")
            self._record_compile(run_dir)
            return result

        result = self._run_worker(run_dir)
        self._record_compile(run_dir)
        return result

    def _run_worker(self, run_dir: Path) -> dict[str, Any]:
        args = [
            sys.executable,
            "-m",
            "mini_articraft.environments.worker",
            "--raw",
            str(run_dir.resolve()),
        ]
        if self.config.physics_enabled:
            args.append("--physics")
        completed = _run_isolated_process(
            args,
            cwd=run_dir.resolve(),
            timeout_seconds=self.config.timeout_seconds,
        )
        compile_stats = _read_compile_progress(run_dir)
        if completed.timed_out:
            if not compile_stats:
                compile_stats = {
                    "total_seconds": self.config.timeout_seconds,
                    "current_phase": "starting the compile worker",
                    "current_phase_seconds": self.config.timeout_seconds,
                    "phases": {},
                    "model": {},
                }
            return _error_result(
                run_dir,
                error=_timeout_error(self.config.timeout_seconds, compile_stats),
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
                compile_stats=compile_stats,
            )

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return _error_result(
                run_dir,
                error="compile worker did not return JSON",
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
            )

        if not isinstance(payload, dict):
            return _error_result(
                run_dir,
                error="compile worker returned invalid JSON",
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
            )

        return _finalize_payload(
            run_dir,
            payload,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

    def _record_compile(self, run_dir: Path) -> None:
        record = Record.load(run_dir / "record.json")
        record.run_id = run_dir.name
        record.attempts += 1
        record.save(run_dir / "record.json")


def _validate_run_id(run_id: str) -> str:
    value = str(run_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("run_id must be a simple folder name")
    return value


def _link_sdk_docs(workspace: Path) -> None:
    docs_dir = workspace / "docs"
    docs_dir.mkdir()
    docs_dir.joinpath("sdk").symlink_to(package_dir / "sdk" / "docs", target_is_directory=True)


@dataclass(frozen=True)
class _ProcessResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False


def _run_isolated_process(
    args: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> _ProcessResult:
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        env=child_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, stderr = _stop_timed_out_worker(proc)
        return _ProcessResult(stdout, stderr, proc.returncode, timed_out=True)
    return _ProcessResult(stdout, stderr, proc.returncode)


def _stop_timed_out_worker(proc: subprocess.Popen[str]) -> tuple[str, str]:
    _signal_worker_group(proc, signal.SIGTERM)
    try:
        return proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        _signal_worker_group(proc, signal.SIGKILL)
        return proc.communicate()


def _signal_worker_group(proc: subprocess.Popen[str], sig: signal.Signals) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return


def _with_paths(run_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    workspace = run_dir / "workspace"
    result_dir = run_dir / "result"
    result["run_id"] = run_dir.name
    result["run"] = str(run_dir)
    result["workspace"] = str(workspace)
    result["entrypoint"] = str(workspace / "main.py")
    result["result"] = str(result_dir)
    return result


def _finalize_payload(
    run_dir: Path,
    payload: dict[str, Any],
    *,
    stderr: str,
    returncode: int | None,
) -> dict[str, Any]:
    """Assemble the environment-level compile result from a worker payload.

    Single owner of result assembly for any worker transport: captured worker
    stderr is followed by process-level stderr, missing keys are defaulted,
    and the compile report and run paths are attached here.
    """
    result = CompileResult.from_payload(payload)
    result.stderr += stderr
    return _finalize_result(run_dir, result, returncode)


def _finalize_result(
    run_dir: Path,
    result: CompileResult,
    returncode: int | None,
) -> dict[str, Any]:
    payload = result.to_payload(
        include_report=True,
        include_returncode=True,
        returncode=returncode,
    )
    return _with_paths(run_dir, payload)


def _read_compile_progress(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "result" / _COMPILE_PROGRESS_FILE
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    finally:
        path.unlink(missing_ok=True)


def _timeout_error(timeout_seconds: float, compile_stats: dict[str, Any]) -> str:
    phase = str(compile_stats.get("current_phase") or "starting the compile worker")
    phase_seconds = compile_stats.get("current_phase_seconds")
    phase_detail = (
        f" The compiler spent {float(phase_seconds):.1f}s in that phase before it was stopped."
        if isinstance(phase_seconds, int | float)
        else ""
    )
    return (
        f"Compile timed out after {timeout_seconds:g}s while {phase}."
        f"{phase_detail} The worker and its child processes were stopped. "
        "Simplify the operation named above, increase weld tolerance when a dense mesh operation "
        "is the cause, or raise MINI_ARTICRAFT_COMPILE_TIMEOUT_SECONDS."
    )


def _error_result(
    run_dir: Path,
    *,
    error: str,
    stdout: str = "",
    stderr: str = "",
    returncode: int | None = None,
    compile_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _finalize_result(
        run_dir,
        CompileResult(
            error=error,
            stdout=stdout,
            stderr=stderr,
            compile_stats=compile_stats,
        ),
        returncode,
    )
