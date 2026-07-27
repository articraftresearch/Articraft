from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

import mini_articraft.environments.local as local_module
from mini_articraft.environments.local import (
    DEFAULT_MAIN_PY,
    LocalEnvironment,
    _run_isolated_process,
)
from mini_articraft.environments.worker import _merge_test_reports, _serialize_test_report
from mini_articraft.record import Record, read_conversation
from mini_articraft.sdk import TestFailure, TestReport


def write_main(run_dir, code: str) -> None:
    run_dir.joinpath("workspace", "main.py").write_text(code, encoding="utf-8")


def _assert_process_exited(pid: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    pytest.fail(f"process {pid} was still running after compile timeout")


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_create_run_seeds_an_editable_scaffold_and_docs(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("seeded")

    assert run_dir.joinpath("workspace", "main.py").read_text() == DEFAULT_MAIN_PY
    assert run_dir.joinpath("workspace", "docs", "sdk").is_symlink()
    assert Record.load(run_dir / "record.json") == Record(run_id="seeded")


def test_compile_path_exports_usdz_but_only_updates_attempt_data(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("box")

    result = env.compile_path(run_dir)

    assert result["status"] == "success"
    assert result["usdz"] == str(run_dir / "result" / "usdz" / "0000.usdz")
    assert result["compile_report"]["status"] == "success"
    assert result["compile_report"]["counts"]["failures"] == 0
    assert "<compile_signals>" in result["compile_report"]["signals_text"]
    assert result["compile_stats"]["total_seconds"] > 0.0
    assert result["compile_stats"]["model"] == {
        "parts": 1,
        "shapes": 1,
        "articulations": 0,
    }
    assert "loading main.py and building the model" in result["compile_stats"]["phases"]
    assert not run_dir.joinpath("result", ".compile-progress.json").exists()
    manifest = json.loads(run_dir.joinpath("result", "model.json").read_text())
    assert manifest["name"] == "object"
    assert "qa_report" not in result
    assert not run_dir.joinpath("workspace", "qa").exists()
    assert not run_dir.joinpath("result", "qa").exists()
    assert not list(run_dir.joinpath("result").rglob("*.png"))

    assert Record.load(run_dir / "record.json") == Record(
        run_id="box",
        status="created",
        attempts=1,
    )
    assert read_conversation(run_dir / "conversation.jsonl") == []


def test_failed_checks_save_a_usdz_without_publishing_it(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("failure")
    write_main(
        run_dir,
        """from build123d import Box
from mini_articraft.sdk import ArticulatedObject, TestContext, TestReport

object_model = ArticulatedObject("failure")
base = object_model.part("base")
base.add(Box(1, 1, 1), name="body")

def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.fail("prompt feature", "missing")
    return ctx.report()
""",
    )

    first = env.compile_path(run_dir)
    second = env.compile_path(run_dir)

    assert first["status"] == second["status"] == "error"
    assert first["usdz"] == str(run_dir / "result" / "usdz" / "0000.usdz")
    assert second["usdz"] == str(run_dir / "result" / "usdz" / "0001.usdz")
    assert Path(first["usdz"]).is_file()
    assert Path(second["usdz"]).is_file()
    assert run_dir.joinpath("result", "model.json").is_file()
    assert not run_dir.joinpath("result", "qa").exists()
    assert first["test_report"]["passed"] is False
    record = Record.load(run_dir / "record.json")
    assert record.status == "created"
    assert record.attempts == 2
    assert record.result == ""


def test_a_fatal_attempt_does_not_consume_the_next_usdz_number(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("versions")
    first = env.compile_path(run_dir)
    write_main(run_dir, "raise RuntimeError('bad edit')\n")
    failed = env.compile_path(run_dir)
    write_main(run_dir, DEFAULT_MAIN_PY)
    second = env.compile_path(run_dir)

    assert first["usdz"].endswith("0000.usdz")
    assert failed["status"] == "error"
    assert not failed.get("usdz")
    assert second["usdz"].endswith("0001.usdz")


def test_compile_path_supports_workspace_modules(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("modules")
    parts = run_dir / "workspace" / "parts"
    parts.mkdir()
    parts.joinpath("body.py").write_text(
        """from build123d import Box
from mini_articraft.sdk import ArticulatedObject

def build():
    model = ArticulatedObject("module")
    base = model.part("base")
    base.add(Box(0.2, 0.2, 0.2), name="body")
    return model
""",
        encoding="utf-8",
    )
    write_main(
        run_dir,
        """from mini_articraft.sdk import TestContext, TestReport
from parts.body import build

object_model = build()

def run_tests() -> TestReport:
    return TestContext(object_model).report()
""",
    )

    assert env.compile_path(run_dir)["status"] == "success"


def test_compile_worker_runs_from_run_dir_without_api_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("EXAMPLE_API_KEY", "example-secret")
    monkeypatch.setenv("VISIBLE_SETTING", "visible")
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("worker-boundary")
    write_main(
        run_dir,
        """import json
import os
from pathlib import Path
from build123d import Box
from mini_articraft.sdk import ArticulatedObject, TestContext, TestReport

Path("worker-boundary.json").write_text(json.dumps({
    "cwd": str(Path.cwd()),
    "openai": os.environ.get("OPENAI_API_KEY"),
    "example": os.environ.get("EXAMPLE_API_KEY"),
    "visible": os.environ.get("VISIBLE_SETTING"),
}), encoding="utf-8")
object_model = ArticulatedObject("boundary")
object_model.part("body").add(Box(0.1, 0.1, 0.1), name="shell")

def run_tests() -> TestReport:
    return TestContext(object_model).report()
""",
    )

    result = env.compile_path(run_dir)
    assert result["status"] == "success", result
    child = json.loads(
        run_dir.joinpath("workspace", "worker-boundary.json").read_text(encoding="utf-8")
    )

    assert child == {
        "cwd": str(run_dir.joinpath("workspace").resolve()),
        "openai": None,
        "example": None,
        "visible": "visible",
    }


def test_local_environment_starts_worker_from_run_dir(monkeypatch, tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("worker-cwd")
    captured: dict[str, Path] = {}

    def run_worker(args, *, cwd, timeout_seconds):
        captured["cwd"] = cwd
        return local_module._ProcessResult(
            stdout=json.dumps({"status": "success"}),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(local_module, "_run_isolated_process", run_worker)

    assert env._run_worker(run_dir)["status"] == "success"
    assert captured["cwd"] == run_dir.resolve()


def test_create_run_requires_a_new_simple_run_id(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    env.create_run("one")
    with pytest.raises(FileExistsError):
        env.create_run("one")
    with pytest.raises(ValueError):
        env.create_run("../escape")


def test_compile_path_reports_missing_required_entrypoint_values(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)

    missing_main = env.create_run("missing_main")
    missing_main.joinpath("workspace", "main.py").unlink()
    assert env.compile_path(missing_main)["error"] == "workspace/main.py is required"

    missing_model = env.create_run("missing_model")
    write_main(missing_model, "value = 1\n")
    assert "object_model" in env.compile_path(missing_model)["error"]

    missing_tests = env.create_run("missing_tests")
    write_main(
        missing_tests,
        """from build123d import Box
from mini_articraft.sdk import ArticulatedObject
object_model = ArticulatedObject("box")
base = object_model.part("base")
base.add(Box(1, 1, 1), name="body")
""",
    )
    assert "run_tests" in env.compile_path(missing_tests)["error"]

    bad_report = env.create_run("bad_report")
    write_main(
        bad_report,
        """from build123d import Box
from mini_articraft.sdk import ArticulatedObject
object_model = ArticulatedObject("box")
base = object_model.part("base")
base.add(Box(1, 1, 1), name="body")
def run_tests():
    return None
""",
    )
    assert "must return TestReport" in env.compile_path(bad_report)["error"]


def test_baseline_overlap_is_a_nonblocking_compiler_diagnostic(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("overlap")
    write_main(
        run_dir,
        """from build123d import Box
from mini_articraft.sdk import ArticulatedObject, ArticulationType, Origin, TestContext, TestReport

object_model = ArticulatedObject("overlap")
base = object_model.part("base"); base.add(Box(1, 1, 1), name="body")
child = object_model.part("child"); child.add(Box(1, 1, 1), name="body")
object_model.articulation("mount", ArticulationType.FIXED, base, child, origin=Origin())

def run_tests() -> TestReport:
    return TestContext(object_model).report()
""",
    )

    result = env.compile_path(run_dir)

    assert result["status"] == "success"
    assert any(
        "fail_if_parts_overlap_in_current_pose" in warning
        for warning in result["test_report"]["warnings"]
    )
    assert Path(result["usdz"]).is_file()


def test_compile_reports_custom_visual_artifacts_without_copying_files(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("custom-evidence")
    write_main(
        run_dir,
        """import json
from pathlib import Path
from build123d import Box
from mini_articraft.sdk import ArticulatedObject, TestContext, TestReport

object_model = ArticulatedObject("evidence")
object_model.part("body").add(Box(1, 1, 1), name="box")

def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    Path("qa").mkdir(exist_ok=True)
    Path("qa/measurements.json").write_text(
        json.dumps({"width": 1.0}), encoding="utf-8"
    )
    ctx.attach_artifact("qa/measurements.json", name="measurements")
    ctx.attach_artifact("qa/custom_view.png", name="custom view")
    return ctx.report()
""",
    )
    qa_dir = run_dir / "workspace" / "qa"
    qa_dir.mkdir()
    Image.new("RGB", (8, 8), color="white").save(qa_dir / "custom_view.png")
    Image.new("RGB", (8, 8), color="white").save(qa_dir / "working_preview.png")

    result = env.compile_path(run_dir)

    assert result["status"] == "success", result
    assert {item["name"] for item in result["test_report"]["artifacts"]} == {
        "measurements",
        "custom view",
    }
    assert "workspace_path=qa/custom_view.png" in result["compile_report"]["signals_text"]
    assert run_dir.joinpath("workspace", "qa", "working_preview.png").is_file()
    assert not run_dir.joinpath("result", "qa").exists()
    assert not run_dir.joinpath("result", "artifacts").exists()
    assert not list(run_dir.joinpath("result").rglob("*.png"))


def test_disconnected_geometry_is_a_compiler_warning(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("disconnected")
    write_main(
        run_dir,
        """from build123d import Box, Pos
from mini_articraft.sdk import ArticulatedObject, TestContext, TestReport

object_model = ArticulatedObject("disconnected")
base = object_model.part("base")
base.add(Pos(X=-1) * Box(0.5, 0.5, 0.5), name="left")
base.add(Pos(X=1) * Box(0.5, 0.5, 0.5), name="right")

def run_tests() -> TestReport:
    return TestContext(object_model).report()
""",
    )

    result = env.compile_path(run_dir)

    assert result["status"] == "success"
    assert any("Disconnected geometry" in warning for warning in result["test_report"]["warnings"])
    assert result["compile_report"]["counts"]["warnings"] >= 1


def test_compile_honors_an_exact_shape_overlap_allowance(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("allowance")
    write_main(
        run_dir,
        """from build123d import Box
from mini_articraft.sdk import ArticulatedObject, ArticulationType, Origin, TestContext, TestReport

object_model = ArticulatedObject("allowance")
shaft = object_model.part("shaft"); shaft.add(Box(1, 1, 1), name="steel")
hub = object_model.part("hub"); hub.add(Box(1, 1, 1), name="liner")
object_model.articulation("mount", ArticulationType.FIXED, shaft, hub, origin=Origin())

def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.allow_overlap(
        "shaft", "hub", shape_a="steel", shape_b="liner", reason="captured shaft"
    )
    return ctx.report()
""",
    )

    result = env.compile_path(run_dir)

    assert result["status"] == "success"
    assert result["test_report"]["allowed_overlaps"][0]["shape_a"] == "liner"
    assert result["test_report"]["allowed_overlaps"][0]["shape_b"] == "steel"


def test_compile_path_reports_timeout(tmp_path) -> None:
    env = LocalEnvironment(output_dir=tmp_path, timeout_seconds=0.2)
    run_dir = env.create_run("slow")
    write_main(run_dir, "import time\ntime.sleep(60)\n")

    result = env.compile_path(run_dir)

    assert result["status"] == "error"
    assert "timed out after 0.2s" in result["error"].lower()
    assert result["compile_stats"]["current_phase"] in result["error"]
    assert "MINI_ARTICRAFT_COMPILE_TIMEOUT_SECONDS" in result["error"]
    assert result["compile_report"]["status"] == "failure"
    signal = result["compile_report"]["signal_bundle"]["signals"][0]
    assert signal["kind"] == "compile_timeout"
    assert signal["code"] == "COMPILE_TIMEOUT"
    assert not run_dir.joinpath("result", ".compile-progress.json").exists()


def test_isolated_process_timeout_cleans_descendants(tmp_path) -> None:
    worker_pid_file = tmp_path / "worker.pid"
    child_pid_file = tmp_path / "child.pid"
    child_code = (
        "from pathlib import Path; import os, time; "
        f"Path({str(child_pid_file)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
        "time.sleep(60)"
    )
    worker_code = f"""
import os
import subprocess
import sys
import time
from pathlib import Path

child_pid_file = Path({str(child_pid_file)!r})
subprocess.Popen([sys.executable, "-c", {child_code!r}])
for _ in range(100):
    if child_pid_file.exists():
        break
    time.sleep(0.01)
Path({str(worker_pid_file)!r}).write_text(str(os.getpid()), encoding="utf-8")
time.sleep(60)
"""
    result = _run_isolated_process(
        [sys.executable, "-c", worker_code], cwd=tmp_path, timeout_seconds=1
    )

    assert result.timed_out is True
    _assert_process_exited(int(worker_pid_file.read_text()))
    _assert_process_exited(int(child_pid_file.read_text()))


def test_merge_test_reports_deduplicates_compiler_owned_checks() -> None:
    authored = TestReport(
        passed=False,
        checks_run=2,
        checks=("custom", "check_model_valid"),
        failures=(TestFailure("check_model_valid", "authored duplicate"),),
        warnings=("one",),
        allowances=("allow", "allow"),
    )
    baseline = TestReport(
        passed=False,
        checks_run=2,
        checks=("check_model_valid", "check_single_root_part"),
        failures=(TestFailure("check_model_valid", "compiler result"),),
        warnings=("one", "two"),
    )

    merged = _merge_test_reports(authored, baseline)

    assert merged.checks == ("custom", "check_model_valid", "check_single_root_part")
    assert merged.failures == (TestFailure("check_model_valid", "compiler result"),)
    assert merged.warnings == ("one", "two")
    assert merged.allowances == ("allow",)


def test_merge_test_reports_keeps_authored_failure_when_baseline_check_passes() -> None:
    authored = TestReport(
        passed=False,
        checks_run=1,
        checks=("check_model_valid",),
        failures=(TestFailure("check_model_valid", "authored constraint"),),
    )
    baseline = TestReport(
        passed=True,
        checks_run=1,
        checks=("check_model_valid",),
        failures=(),
    )

    merged = _merge_test_reports(authored, baseline)

    assert merged.failures == (TestFailure("check_model_valid", "authored constraint"),)
    assert not merged.passed


def test_merge_test_reports_keeps_distinct_same_named_authored_failures() -> None:
    authored = TestReport(
        passed=False,
        checks_run=2,
        checks=("prompt_geometry",),
        failures=(
            TestFailure("prompt_geometry", "missing handle"),
            TestFailure("prompt_geometry", "missing controls"),
        ),
    )
    baseline = TestReport(True, 0, (), ())

    merged = _merge_test_reports(authored, baseline)

    assert merged.failures == authored.failures


def test_serialize_test_report_assigns_failure_provenance_at_worker_boundary() -> None:
    report = TestReport(
        passed=False,
        checks_run=2,
        checks=("authored", "baseline"),
        failures=(
            TestFailure("authored", "authored result"),
            TestFailure("baseline", "compiler result"),
        ),
    )

    serialized = _serialize_test_report(report, compiler_failure_names={"baseline"})

    assert serialized is not None
    assert [failure["source"] for failure in serialized["failures"]] == [
        "tests",
        "compiler",
    ]
