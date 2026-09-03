from __future__ import annotations

from dataclasses import asdict
from typing import Any

from articraft.compiler.feedback import (
    build_compile_report,
    compile_failure_signature,
    render_compile_report,
)
from articraft.sdk import FailureKind, TestArtifact, TestFailure, TestReport


def _serialized_report(
    report: TestReport,
    *,
    sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    serialized = asdict(report)
    source_values = sources or ("tests",) * len(report.failures)
    assert len(source_values) == len(report.failures)
    for failure, source in zip(serialized["failures"], source_values, strict=True):
        failure["source"] = source
    return serialized


def _failing_report(
    *failures: TestFailure,
    warnings: tuple[str, ...] = (),
    sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    return _serialized_report(
        TestReport(
            passed=False,
            checks_run=len(failures),
            checks=tuple(failure.name for failure in failures),
            failures=failures,
            warnings=warnings,
        ),
        sources=sources,
    )


def test_compile_report_clean_success_uses_articraft_style_block() -> None:
    report = build_compile_report(status="success")

    assert report["counts"] == {"failures": 0, "warnings": 0, "notes": 0}
    assert report["signal_bundle"]["summary"] == (
        "status=success failures=0 warnings=0 notes=0\nCompile passed cleanly."
    )
    assert report["signals_text"] == (
        "<compile_signals>\n"
        "<summary>\n"
        "status=success failures=0 warnings=0 notes=0\n"
        "Compile passed cleanly.\n"
        "</summary>\n"
        "</compile_signals>"
    )


def test_compile_report_lists_visual_evidence_paths() -> None:
    test_report = TestReport(
        passed=True,
        checks_run=0,
        checks=(),
        failures=(),
        artifacts=(
            TestArtifact(
                name="body section",
                path="qa/body_section.png",
                kind="image",
                caption="Internal support.",
            ),
        ),
    )

    report = build_compile_report(
        status="success",
        test_report=_serialized_report(test_report),
    )

    assert report["counts"]["notes"] == 1
    assert "workspace_path=qa/body_section.png" in report["signals_text"]
    assert "Open every relevant workspace image" in report["signals_text"]


def test_compile_report_omits_unsafe_visual_evidence_paths() -> None:
    test_report = TestReport(
        passed=True,
        checks_run=0,
        checks=(),
        failures=(),
        artifacts=(
            TestArtifact("safe", "qa/safe.png", "image"),
            TestArtifact("parent escape", "../secret.png", "image"),
            TestArtifact("absolute", "/tmp/secret.png", "image"),
            TestArtifact("backslash", "qa\\secret.png", "image"),
        ),
    )

    report = build_compile_report(
        status="success",
        test_report=_serialized_report(test_report),
    )

    assert report["counts"]["notes"] == 1
    assert "workspace_path=qa/safe.png" in report["signals_text"]
    assert "secret.png" not in report["signals_text"]


def test_compile_report_classifies_runtime_errors() -> None:
    report = build_compile_report(status="failure", error="ValueError: bad loft")

    signal = report["signal_bundle"]["signals"][0]
    assert signal["kind"] == "compile_runtime"
    assert signal["group"] == "build"
    assert signal["blocking"] is True
    assert compile_failure_signature(report)
    assert "<failures>" in report["signals_text"]
    assert "- FAILURE [compile_runtime] ValueError: bad loft" in report["signals_text"]


def test_runtime_signal_includes_one_sanitized_workspace_frame() -> None:
    report = build_compile_report(
        status="failure",
        error="NameError: name 'missing' is not defined",
        traceback_text=(
            "Traceback (most recent call last):\n"
            '  File "/private/repo/src/articraft/compiler/worker.py", line 1, in run\n'
            "    runpy.run_path(path)\n"
            '  File "/private/runs/demo/workspace/helpers/body.py", line 12, in build\n'
            "    return missing\n"
            "NameError: name 'missing' is not defined\n"
        ),
    )

    signal = report["signal_bundle"]["signals"][0]
    assert signal["details"] == "location=helpers/body.py:12 in build"
    assert signal["details"].count("NameError") == 0
    assert "/private/runs" not in report["signals_text"]


def test_compile_report_classifies_required_run_tests_errors() -> None:
    missing = build_compile_report(
        status="failure",
        error="ValueError: Missing required `run_tests()` in main.py.",
    )
    invalid = build_compile_report(
        status="failure",
        error="ValueError: run_tests() must return TestReport (got NoneType)",
    )

    assert missing["signal_bundle"]["signals"][0]["kind"] == "missing_run_tests"
    assert invalid["signal_bundle"]["signals"][0]["kind"] == "invalid_run_tests_report"


def test_compile_report_classifies_timeout_with_specific_guidance() -> None:
    report = build_compile_report(
        status="failure",
        error=(
            "Compile timed out after 900s while loading main.py and building the model. "
            "The worker and its child processes were stopped."
        ),
    )

    signal = report["signal_bundle"]["signals"][0]
    assert signal["kind"] == "compile_timeout"
    assert signal["code"] == "COMPILE_TIMEOUT"
    assert "Inspect the phase named in the timeout" in report["signals_text"]
    assert "coarser tolerance" in report["signals_text"]


def test_compile_report_classifies_runtime_unknown_shape_as_missing_geometry() -> None:
    report = build_compile_report(
        status="failure",
        error="ValidationError: unknown shape 'hinge_boss' on part 'frame'",
    )

    signal = report["signal_bundle"]["signals"][0]
    assert signal["kind"] == "missing_exact_geometry"
    assert signal["code"] == "TEST_MISSING_EXACT_GEOMETRY"
    assert signal["source"] == "tests"


def test_compile_report_maps_test_failures_warnings_and_allowances() -> None:
    test_report = _serialized_report(
        TestReport(
            passed=False,
            checks_run=3,
            checks=("prompt check", "fail_if_parts_overlap_in_current_pose()"),
            failures=(
                TestFailure("prompt check", "missing handle"),
                TestFailure(
                    "fail_if_parts_overlap_in_current_pose()",
                    "'drawer' vs 'frame': collided=True contacts=1",
                    kind=FailureKind.OVERLAP,
                ),
            ),
            warnings=("thin wall",),
            allowances=(
                "allow_overlap('shaft', 'hub', shape_a='steel', shape_b='liner'): intentional capture",
            ),
        ),
        sources=("tests", "compiler"),
    )

    report = build_compile_report(status="failure", test_report=test_report)
    kinds = [signal["kind"] for signal in report["signal_bundle"]["signals"]]

    assert kinds == ["real_overlap", "test_failure", "test_warning", "allowed_overlap"]
    assert report["counts"] == {"failures": 2, "warnings": 1, "notes": 1}
    assert "<warnings>" in report["signals_text"]
    assert "<notes>" in report["signals_text"]


def test_compile_report_maps_disconnected_geometry_warning() -> None:
    test_report = TestReport(
        passed=True,
        checks_run=1,
        checks=("warn_if_part_contains_disconnected_geometry_islands(contact_tol=1e-06)",),
        failures=(),
        warnings=(
            "Disconnected geometry islands detected:\n"
            "part='base' connected=1/2 contact_tol=1e-06 "
            "disconnected=[solid_002 nearest=solid_001 distance=0.2]",
        ),
    )

    report = build_compile_report(status="success", test_report=test_report)
    signal = report["signal_bundle"]["signals"][0]

    assert signal["kind"] == "disconnected_geometry"
    assert signal["code"] == "WARN_DISCONNECTED_GEOMETRY"
    assert signal["severity"] == "warning"
    assert signal["blocking"] is False


def test_compile_report_classifies_contact_and_scale() -> None:
    test_report = TestReport(
        passed=False,
        checks_run=1,
        checks=("contact",),
        failures=(
            TestFailure(
                "contact",
                "'lid' vs 'body': distance=0.015 collided=False contact_tol=1e-06",
                kind=FailureKind.CONTACT,
            ),
        ),
        warnings=("Scale warning:\nabsurd dimension: 'base'/'body' spans 2000m",),
    )

    report = build_compile_report(status="failure", test_report=test_report)
    signals = report["signal_bundle"]["signals"]

    assert [signal["kind"] for signal in signals] == [
        "exact_contact_gap",
        "geometry_scale",
    ]
    assert signals[1]["group"] == "design"


def test_compile_failure_signature_and_rendering_are_stable() -> None:
    left = build_compile_report(
        status="failure",
        test_report=TestReport(
            passed=False,
            checks_run=2,
            checks=("b", "a"),
            failures=(TestFailure("b", "bad b"), TestFailure("a", "bad a")),
            warnings=("warning one",),
        ),
    )
    right = build_compile_report(
        status="failure",
        test_report=TestReport(
            passed=False,
            checks_run=2,
            checks=("a", "b"),
            failures=(TestFailure("a", "bad a"), TestFailure("b", "bad b")),
            warnings=("a different warning",),
        ),
    )

    assert compile_failure_signature(left) == compile_failure_signature(right)
    rendered = render_compile_report(left, repeated=True, failure_streak=3)["signals_text"]
    assert "This failure matches the previous compile attempt." in rendered
    assert "This is compile failure 3 in a row." in rendered
    assert "`exec_command` inspection" in rendered


def test_contact_gap_is_classified_as_a_gap_not_an_overlap() -> None:
    report = build_compile_report(
        status="failure",
        test_report=_failing_report(
            TestFailure(
                name="expect_contact(base,lid)",
                details="pair=('base','lid') distance=0.0032 collided=False contact_tol=0.001",
                kind=FailureKind.CONTACT,
            )
        ),
    )

    signal = report["signal_bundle"]["signals"][0]
    assert signal["kind"] == "exact_contact_gap"
    assert signal["code"] == "TEST_EXACT_CONTACT_GAP"
    assert signal["source"] == "tests"
    assert "a gap, not an overlap" in report["signals_text"]


def test_compiler_owned_checks_keep_their_codes_and_sources() -> None:
    report = build_compile_report(
        status="failure",
        test_report=_failing_report(
            TestFailure(
                name="check_model_valid",
                details="ValidationError: bad shape",
                kind=FailureKind.MODEL_VALIDITY,
            ),
            TestFailure(
                name="fail_if_isolated_parts()",
                details="Isolated parts detected",
                kind=FailureKind.ISOLATED_PART,
            ),
            TestFailure(
                name="fail_if_parts_overlap_in_current_pose()",
                details="RigidBody overlaps detected",
                kind=FailureKind.OVERLAP,
            ),
            sources=("compiler",) * 3,
        ),
    )

    by_name = {s["check_name"]: s for s in report["signal_bundle"]["signals"]}
    assert by_name["check_model_valid"]["code"] == "QC_MODEL_VALIDITY"
    assert by_name["check_model_valid"]["source"] == "compiler"
    assert by_name["fail_if_isolated_parts()"]["code"] == "QC_ISOLATED_PART"
    assert by_name["fail_if_isolated_parts()"]["source"] == "compiler"
    assert by_name["fail_if_parts_overlap_in_current_pose()"]["code"] == "QC_REAL_OVERLAP"


def test_authored_checks_with_the_same_findings_map_to_test_codes() -> None:
    report = build_compile_report(
        status="failure",
        test_report=_failing_report(
            TestFailure(
                name="fail_if_isolated_parts(tight)",
                details="isolated parts detected: ['antenna']",
                kind=FailureKind.ISOLATED_PART,
            ),
            TestFailure(
                name="expect_no_collision(base,lid)",
                details="pair=('base','lid') collided=true overlap_volume=1e-6",
                kind=FailureKind.OVERLAP,
            ),
        ),
    )

    kinds = {signal["kind"]: signal for signal in report["signal_bundle"]["signals"]}
    assert kinds["isolated_part"]["code"] == "TEST_ISOLATED_PART"
    assert kinds["isolated_part"]["source"] == "tests"
    assert kinds["real_overlap"]["code"] == "TEST_REAL_OVERLAP"
    assert kinds["real_overlap"]["source"] == "tests"


def test_every_failure_kind_maps_to_a_specific_signal() -> None:
    expected_kinds = {
        FailureKind.MODEL_VALIDITY: "model_validity",
        FailureKind.MESH_HEALTH: "mesh_health",
        FailureKind.ISOLATED_PART: "isolated_part",
        FailureKind.DISCONNECTED_GEOMETRY: "disconnected_geometry",
        FailureKind.OVERLAP: "real_overlap",
        FailureKind.CONTACT: "exact_contact_gap",
        FailureKind.ARTICULATION_SEPARATION: "articulation_separation",
        FailureKind.LOOP_LIMITS: "loop_limits",
        FailureKind.MISSING_MASS: "missing_mass",
        FailureKind.AUTHORED: "test_failure",
    }
    assert set(FailureKind) == set(expected_kinds)
    for kind, signal_kind in expected_kinds.items():
        for source, prefix in (("tests", "TEST"), ("compiler", "QC")):
            report = build_compile_report(
                status="failure",
                test_report=_failing_report(
                    TestFailure(name="some_check", details="d", kind=kind),
                    sources=(source,),
                ),
            )
            signal = report["signal_bundle"]["signals"][0]
            assert signal["kind"] == signal_kind, (kind, source)
            assert signal["code"].startswith(f"{prefix}_"), (kind, source)


def test_new_kinds_carry_their_own_codes_and_guidance() -> None:
    report = build_compile_report(
        status="failure",
        test_report=_failing_report(
            TestFailure(
                name="fail_if_part_contains_disconnected_geometry_islands(contact_tol=1e-06)",
                details="Disconnected geometry islands detected",
                kind=FailureKind.DISCONNECTED_GEOMETRY,
            ),
            TestFailure(
                name="fail_if_articulation_separates_child(gap_tol=0.003)",
                details="articulation='lid_hinge' rest_gap=0m max_gap=0.02m",
                kind=FailureKind.ARTICULATION_SEPARATION,
            ),
            sources=("tests", "compiler"),
        ),
    )

    by_kind = {s["kind"]: s for s in report["signal_bundle"]["signals"]}
    assert by_kind["disconnected_geometry"]["code"] == "TEST_DISCONNECTED_GEOMETRY"
    assert by_kind["articulation_separation"]["code"] == "QC_ARTICULATION_SEPARATION"
    assert "throughout the motion range" in report["signals_text"]

    disconnected_only = build_compile_report(
        status="failure",
        test_report=_failing_report(
            TestFailure(
                name="fail_if_part_contains_disconnected_geometry_islands(contact_tol=1e-06)",
                details="Disconnected geometry islands detected",
                kind=FailureKind.DISCONNECTED_GEOMETRY,
            ),
        ),
    )
    assert "overlaps the nearest piece" in disconnected_only["signals_text"]


def test_invalid_run_tests_return_type_is_a_build_signal() -> None:
    report = build_compile_report(
        status="failure",
        error="ValueError: run_tests() must return TestReport (got dict)",
    )

    signal = report["signal_bundle"]["signals"][0]
    assert signal["kind"] == "invalid_run_tests_report"
    assert signal["code"] == "COMPILE_INVALID_RUN_TESTS_REPORT"
    assert signal["group"] == "build"


def test_mesh_authoring_error_is_a_mesh_health_signal() -> None:
    report = build_compile_report(
        status="failure",
        error=(
            "ValueError: weld produced unhealthy mesh geometry: "
            "sliver_faces=3 near ((0, 0, 0), (1, 1, 1))"
        ),
    )

    signal = report["signal_bundle"]["signals"][0]
    assert signal["kind"] == "mesh_health"
    assert signal["code"] == "COMPILE_MESH_HEALTH"
    assert "reported bounds" in report["signals_text"]
    assert "allow_mesh_issues" not in report["signals_text"]


def test_repeat_and_streak_guidance_escalates() -> None:
    report = build_compile_report(
        status="failure",
        test_report=_failing_report(TestFailure(name="expect_contact(a,b)", details="d")),
    )

    rendered = render_compile_report(report, repeated=True, failure_streak=3)

    assert "This failure matches the previous compile attempt." in rendered["signals_text"]
    assert "This is compile failure 3 in a row." in rendered["signals_text"]
    assert "exec_command" in rendered["signals_text"]


def test_failure_signature_ignores_warning_only_reports() -> None:
    report = build_compile_report(
        status="success",
        test_report=TestReport(
            passed=True,
            checks_run=1,
            checks=("check_model_valid",),
            failures=(),
            warnings=("Scale warning: geometry outlier dimensions",),
        ),
    )

    assert compile_failure_signature(report) is None
    codes = {signal["code"] for signal in report["signal_bundle"]["signals"]}
    assert codes == {"WARN_GEOMETRY_SCALE"}


def test_traceback_location_prefers_the_last_workspace_frame() -> None:
    report = build_compile_report(
        status="failure",
        error="ValueError: boom",
        traceback_text=(
            '  File "/opt/venv/lib/python3.11/site-packages/build123d/x.py", line 1, in f\n'
            '  File "/private/runs/demo/workspace/main.py", line 9, in build\n'
            '  File "/private/runs/demo/workspace/parts/lid.py", line 42, in hinge\n'
        ),
    )

    signal = report["signal_bundle"]["signals"][0]
    assert signal["details"] == "location=parts/lid.py:42 in hinge"
