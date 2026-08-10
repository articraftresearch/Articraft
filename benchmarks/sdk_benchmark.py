from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from build123d import Axis, Box, Cylinder, Pos

from articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    BoxGeometry,
    CylinderGeometry,
    ExtrudeWithHolesGeometry,
    MeshGeometry,
    Origin,
    RoundedBoxGeometry,
    SphereGeometry,
    SuperellipsoidGeometry,
    SweepGeometry,
    TestContext,
)
from articraft.sdk._collision import MeshCollisionKernel
from articraft.sdk._mesh.boolean import _boolean_union_many
from articraft.sdk.mesh import (
    LoftSection,
    SectionLoftSpec,
    ShellPartitionRegion,
    ShellPartitionSpec,
    SweepSection,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    build123d_to_mesh,
    partition_shell,
    refine_mesh,
    rounded_rect_profile,
    sample_catmull_rom_spline_3d,
    sample_cubic_bezier_spline_3d,
    section_loft,
    smooth_difference,
    smooth_mesh,
    subdivide_mesh,
    superellipse_profile,
    sweep_profile_along_spline,
    tube_from_spline_points,
    tube_network_from_paths,
    weld,
)

Result = object


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    group: str
    operation: Callable[[], Result]
    suite: str = "standard"


def _circle_points(radius: float, z: float, count: int) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            radius * math.cos(2.0 * math.pi * index / count),
            radius * math.sin(2.0 * math.pi * index / count),
            z,
        )
        for index in range(count)
    )


def _collision_model() -> ArticulatedObject:
    model = ArticulatedObject("benchmark_assembly")
    previous = None
    for index in range(10):
        part = model.part(f"part_{index:02d}")
        part.add(BoxGeometry((0.032, 0.024, 0.018)), name="body")
        part.add(
            CylinderGeometry(0.006, 0.028, radial_segments=32).translate(0.0, 0.0, 0.012),
            name="post",
        )
        if previous is not None:
            model.articulation(
                f"joint_{index:02d}",
                ArticulationType.FIXED,
                previous,
                part,
                origin=Origin(xyz=(0.03, 0.0, 0.0)),
            )
        previous = part
    return model


def _run_compiler_checks(model: ArticulatedObject) -> object:
    context = TestContext(model)
    context.fail_if_isolated_parts()
    context.warn_if_part_contains_disconnected_geometry_islands()
    context.fail_if_parts_overlap_in_current_pose()
    return context.report()


def make_cases() -> list[BenchmarkCase]:
    dense_sphere = SphereGeometry(0.08, width_segments=128, height_segments=64)
    transform_source = dense_sphere.copy()
    boolean_a = RoundedBoxGeometry((0.10, 0.07, 0.05), 0.008, tolerance=0.0015)
    boolean_b = CylinderGeometry(0.024, 0.12, radial_segments=96).rotate_y(math.pi / 2)
    boolean_c = SphereGeometry(0.037, width_segments=64, height_segments=32).translate(
        0.035, 0.0, 0.0
    )
    weld_a = BoxGeometry((0.07, 0.026, 0.026))
    weld_b = BoxGeometry((0.026, 0.07, 0.026)).translate(0.025, 0.0, 0.0)
    weld_c = CylinderGeometry(0.012, 0.055, radial_segments=48).translate(0.0, 0.0, 0.02)
    refine_source = SuperellipsoidGeometry(
        (0.07, 0.05, 0.035),
        radial_segments=64,
        height_segments=32,
    )
    section_spec = SectionLoftSpec(
        sections=tuple(
            LoftSection(_circle_points(radius, z, 96))
            for radius, z in ((0.055, 0.0), (0.072, 0.05), (0.060, 0.11), (0.038, 0.16))
        ),
        interpolation="catmull_rom",
        parameterization="centripetal",
        samples_per_span=8,
        tension=0.15,
    )
    sweep_path = sample_catmull_rom_spline_3d(
        (
            (0.0, 0.0, 0.0),
            (0.035, 0.005, 0.025),
            (0.070, -0.01, 0.055),
            (0.105, 0.015, 0.080),
            (0.145, 0.0, 0.10),
        ),
        samples_per_segment=24,
    )
    sweep_profile = superellipse_profile(0.018, 0.012, 2.6, segments=96)
    build123d_shape = Pos(X=0.018) * Box(0.08, 0.05, 0.035) + Cylinder(0.018, 0.07).rotate(
        Axis.Y, 90.0
    )
    shell = boolean_difference(
        BoxGeometry((0.12, 0.08, 0.06)),
        BoxGeometry((0.105, 0.065, 0.05)),
    )
    partition_spec = ShellPartitionSpec(
        shell=shell,
        regions=(
            ShellPartitionRegion("left", side="left"),
            ShellPartitionRegion("right", side="right"),
        ),
        center_gap=0.002,
    )
    collision_model = _collision_model()
    warm_kernel = MeshCollisionKernel(collision_model, mesh_tolerance=0.001)

    return [
        BenchmarkCase(
            "profile.catmull_rom_3d",
            "profiles",
            lambda: sample_catmull_rom_spline_3d(
                (
                    (0.0, 0.0, 0.0),
                    (0.02, 0.01, 0.03),
                    (0.05, -0.01, 0.05),
                    (0.08, 0.02, 0.08),
                    (0.12, 0.0, 0.11),
                ),
                samples_per_segment=128,
            ),
            "quick",
        ),
        BenchmarkCase(
            "profile.cubic_bezier_3d",
            "profiles",
            lambda: sample_cubic_bezier_spline_3d(
                (
                    (0.0, 0.0, 0.0),
                    (0.03, 0.02, 0.04),
                    (0.07, -0.02, 0.07),
                    (0.10, 0.0, 0.10),
                ),
                samples_per_segment=512,
            ),
            "quick",
        ),
        BenchmarkCase(
            "primitive.superellipsoid_dense",
            "construction",
            lambda: SuperellipsoidGeometry(
                (0.08, 0.055, 0.04),
                radial_segments=192,
                height_segments=96,
            ),
            "quick",
        ),
        BenchmarkCase(
            "primitive.rounded_box",
            "construction",
            lambda: RoundedBoxGeometry((0.12, 0.08, 0.05), 0.01, tolerance=0.0008),
        ),
        BenchmarkCase(
            "mesh.extrude_with_holes",
            "construction",
            lambda: ExtrudeWithHolesGeometry(
                rounded_rect_profile(0.14, 0.09, 0.012, corner_segments=20),
                [superellipse_profile(0.028, 0.018, 2.5, segments=96)],
                0.025,
            ),
            "quick",
        ),
        BenchmarkCase("loft.section_dense", "construction", lambda: section_loft(section_spec)),
        BenchmarkCase(
            "sweep.variable_dense",
            "construction",
            lambda: SweepGeometry(
                sweep_profile,
                sweep_path,
                cap=True,
                sections=(
                    SweepSection(0.0, scale=(1.0, 1.0)),
                    SweepSection(0.45, scale=(1.25, 0.8), rotation=0.2),
                    SweepSection(1.0, scale=(0.65, 0.7), rotation=-0.1),
                ),
                section_interpolation="catmull_rom",
                section_tension=0.15,
            ),
        ),
        BenchmarkCase(
            "sweep.tube_spline",
            "construction",
            lambda: tube_from_spline_points(
                (
                    (0.0, 0.0, 0.0),
                    (0.04, 0.02, 0.04),
                    (0.08, -0.02, 0.07),
                    (0.13, 0.0, 0.10),
                ),
                radius=0.008,
                samples_per_segment=32,
                radial_segments=64,
            ),
        ),
        BenchmarkCase(
            "sweep.profile_spline",
            "construction",
            lambda: sweep_profile_along_spline(
                (
                    (0.0, 0.0, 0.0),
                    (0.04, 0.02, 0.04),
                    (0.08, -0.02, 0.07),
                    (0.13, 0.0, 0.10),
                ),
                profile=sweep_profile,
                samples_per_segment=32,
            ),
        ),
        BenchmarkCase(
            "sweep.tube_network",
            "construction",
            lambda: tube_network_from_paths(
                (
                    ((0.0, 0.0, 0.0), (0.05, 0.0, 0.04), (0.10, 0.0, 0.0)),
                    ((0.05, 0.0, 0.04), (0.05, 0.05, 0.08)),
                    ((0.05, 0.0, 0.04), (0.05, -0.05, 0.08)),
                ),
                radius=0.005,
                radial_segments=32,
                corner_radius=0.008,
            ),
            "extended",
        ),
        BenchmarkCase(
            "conversion.build123d_dense",
            "conversion",
            lambda: build123d_to_mesh(build123d_shape, tolerance=0.0002),
        ),
        BenchmarkCase("core.validate_dense", "core", dense_sphere.validate, "quick"),
        BenchmarkCase("core.to_trimesh_dense", "core", dense_sphere.to_trimesh, "quick"),
        BenchmarkCase("core.bounds_dense", "core", lambda: dense_sphere.bounds, "quick"),
        BenchmarkCase(
            "core.transform_dense",
            "core",
            lambda: (
                transform_source.copy()
                .scale(1.1, 0.9, 1.05)
                .rotate((1.0, 2.0, 3.0), 0.7)
                .translate(0.1, -0.2, 0.3)
            ),
            "quick",
        ),
        BenchmarkCase(
            "boolean.union",
            "booleans",
            lambda: boolean_union(boolean_a, boolean_b),
        ),
        BenchmarkCase(
            "boolean.difference",
            "booleans",
            lambda: boolean_difference(boolean_a, boolean_b),
        ),
        BenchmarkCase(
            "boolean.intersection",
            "booleans",
            lambda: boolean_intersection(boolean_a, boolean_c),
        ),
        BenchmarkCase(
            "boolean.union_many",
            "booleans",
            lambda: _boolean_union_many((boolean_a, boolean_b, boolean_c)),
        ),
        BenchmarkCase(
            "weld.two_solids",
            "weld",
            lambda: weld(weld_a, weld_b, radius=0.006, tolerance=0.0015, profile="round"),
        ),
        BenchmarkCase(
            "weld.three_solids",
            "weld",
            lambda: weld(
                weld_a,
                weld_b,
                weld_c,
                radius=0.007,
                tolerance=0.002,
                profile="soft",
            ),
            "extended",
        ),
        BenchmarkCase(
            "weld.smooth_difference",
            "weld",
            lambda: smooth_difference(
                boolean_a,
                CylinderGeometry(0.018, 0.09, radial_segments=64),
                radius=0.005,
                tolerance=0.0015,
            ),
            "extended",
        ),
        BenchmarkCase(
            "refine.max_edge_length",
            "refinement",
            lambda: refine_mesh(refine_source, max_edge_length=0.006),
        ),
        BenchmarkCase(
            "refine.subdivide_two_levels",
            "refinement",
            lambda: subdivide_mesh(refine_source, levels=2),
        ),
        BenchmarkCase(
            "refine.taubin_smooth",
            "refinement",
            lambda: smooth_mesh(refine_source, iterations=20),
        ),
        BenchmarkCase(
            "shell.partition",
            "booleans",
            lambda: partition_shell(partition_spec),
            "extended",
        ),
        BenchmarkCase(
            "collision.distance_cold",
            "collision",
            lambda: MeshCollisionKernel(collision_model, mesh_tolerance=0.001).distance_between(
                "part_00", "part_09", {}
            ),
        ),
        BenchmarkCase(
            "collision.pair_distances_warm",
            "collision",
            lambda: warm_kernel.pair_distances({}),
        ),
        BenchmarkCase(
            "collision.overlaps_warm",
            "collision",
            lambda: warm_kernel.meaningful_overlaps({}, overlap_tol=0.005, overlap_volume_tol=5e-7),
        ),
        BenchmarkCase(
            "collision.disconnected_geometry_warm",
            "collision",
            lambda: warm_kernel.disconnected_geometry_islands(contact_tol=1e-6),
            "extended",
        ),
        BenchmarkCase(
            "testing.compiler_checks",
            "collision",
            lambda: _run_compiler_checks(collision_model),
            "extended",
        ),
    ]


def _mesh_quality(mesh: MeshGeometry | trimesh.Trimesh) -> dict[str, Any]:
    value = mesh if isinstance(mesh, trimesh.Trimesh) else mesh.to_trimesh(process=False)
    vertices = np.asarray(value.vertices, dtype=np.float64)
    faces = np.asarray(value.faces, dtype=np.int64)
    triangles = vertices[faces] if len(faces) else np.empty((0, 3, 3), dtype=np.float64)
    edges = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    areas = np.asarray(value.area_faces, dtype=np.float64)
    denominator = np.sum(edges * edges, axis=1)
    triangle_quality = np.divide(
        4.0 * math.sqrt(3.0) * areas,
        denominator,
        out=np.zeros_like(areas),
        where=denominator > 0.0,
    )
    edge_keys = np.sort(value.edges, axis=1)
    _unique_edges, edge_counts = np.unique(edge_keys, axis=0, return_counts=True)
    scale = float(np.max(np.ptp(vertices, axis=0))) if len(vertices) else 0.0
    area_epsilon = max(scale * scale * 1e-14, 1e-30)
    return {
        "vertices": len(vertices),
        "triangles": len(faces),
        "watertight": bool(value.is_watertight),
        "winding_consistent": bool(value.is_winding_consistent),
        "bodies": int(value.body_count),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(edge_counts > 2)),
        "degenerate_triangles": int(np.count_nonzero(areas <= area_epsilon)),
        "volume": float(value.volume),
        "surface_area": float(value.area),
        "bounds": np.asarray(value.bounds, dtype=np.float64).round(12).tolist(),
        "triangle_quality_p01": float(np.quantile(triangle_quality, 0.01)),
        "triangle_quality_median": float(np.median(triangle_quality)),
    }


def result_quality(result: Result) -> dict[str, Any]:
    if isinstance(result, MeshGeometry | trimesh.Trimesh):
        return _mesh_quality(result)
    if (
        isinstance(result, Mapping)
        and result
        and all(isinstance(value, MeshGeometry) for value in result.values())
    ):
        values = [_mesh_quality(value) for value in result.values()]
        return {
            "meshes": len(values),
            "watertight": all(value["watertight"] for value in values),
            "winding_consistent": all(value["winding_consistent"] for value in values),
            "bodies": sum(value["bodies"] for value in values),
            "boundary_edges": sum(value["boundary_edges"] for value in values),
            "nonmanifold_edges": sum(value["nonmanifold_edges"] for value in values),
            "degenerate_triangles": sum(value["degenerate_triangles"] for value in values),
            "vertices": sum(value["vertices"] for value in values),
            "triangles": sum(value["triangles"] for value in values),
            "volume": sum(value["volume"] for value in values),
            "surface_area": sum(value["surface_area"] for value in values),
            "triangle_quality_p01": min(value["triangle_quality_p01"] for value in values),
            "triangle_quality_median": statistics.median(
                value["triangle_quality_median"] for value in values
            ),
        }
    if isinstance(result, Sequence) and not isinstance(result, str | bytes):
        array = np.asarray(result)
        return {
            "items": len(result),
            "finite": bool(np.isfinite(array).all()) if array.dtype.kind in "fiu" else True,
        }
    return {}


def _time_operation(
    operation: Callable[[], Result],
    *,
    rounds: int,
    target_seconds: float,
) -> tuple[dict[str, Any], Result]:
    operation()
    started = time.perf_counter()
    result = operation()
    single_seconds = max(time.perf_counter() - started, 1e-9)
    iterations = max(1, min(100_000, math.ceil(target_seconds / single_seconds)))
    samples: list[float] = []
    for _round in range(rounds):
        gc.collect()
        gc.disable()
        started = time.perf_counter()
        try:
            for _iteration in range(iterations):
                result = operation()
        finally:
            elapsed = time.perf_counter() - started
            gc.enable()
        samples.append(elapsed / iterations)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return (
        {
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples),
            "p95_seconds": ordered[p95_index],
            "iterations_per_round": iterations,
            "rounds": rounds,
        },
        result,
    )


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("build123d", "libigl", "manifold3d", "numpy", "scipy", "trimesh"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unknown"
    return result


def run_benchmarks(
    cases: list[BenchmarkCase],
    *,
    rounds: int,
    target_seconds: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases):02d}] {case.name}", flush=True)
        timing, result = _time_operation(
            case.operation,
            rounds=rounds,
            target_seconds=target_seconds,
        )
        results.append(
            {
                "name": case.name,
                "group": case.group,
                "suite": case.suite,
                "timing": timing,
                "quality": result_quality(result),
            }
        )
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
        },
        "config": {"rounds": rounds, "target_seconds": target_seconds},
        "cases": results,
    }


def _relative_change(current: float, baseline: float) -> float:
    return 0.0 if baseline == 0.0 and current == 0.0 else (current - baseline) / abs(baseline)


def compare_results(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    quality_tolerance: float,
) -> list[str]:
    baseline_cases = {case["name"]: case for case in baseline["cases"]}
    failures: list[str] = []
    print("\nComparison")
    print(f"{'case':42} {'current':>10} {'before':>10} {'speed':>9}")
    for case in current["cases"]:
        before = baseline_cases.get(case["name"])
        if before is None:
            continue
        current_seconds = float(case["timing"]["median_seconds"])
        before_seconds = float(before["timing"]["median_seconds"])
        speed = before_seconds / current_seconds if current_seconds else math.inf
        print(f"{case['name']:42} {current_seconds:10.6f} {before_seconds:10.6f} {speed:8.2f}x")
        quality = case["quality"]
        old_quality = before["quality"]
        failures.extend(
            f"{case['name']}: {key} changed from true to false"
            for key in ("watertight", "winding_consistent")
            if old_quality.get(key) is True and quality.get(key) is not True
        )
        failures.extend(
            f"{case['name']}: {key} increased from {old_quality[key]} to {quality.get(key)}"
            for key in ("boundary_edges", "nonmanifold_edges", "degenerate_triangles")
            if key in old_quality and int(quality.get(key, 0)) > int(old_quality[key])
        )
        if "bodies" in old_quality and quality.get("bodies") != old_quality["bodies"]:
            failures.append(
                f"{case['name']}: bodies changed from {old_quality['bodies']} "
                f"to {quality.get('bodies')}"
            )
        for key in ("volume", "surface_area"):
            if key not in old_quality or key not in quality:
                continue
            change = abs(_relative_change(float(quality[key]), float(old_quality[key])))
            if change > quality_tolerance:
                failures.append(
                    f"{case['name']}: {key} changed by {change:.2%}, over {quality_tolerance:.2%}"
                )
        if "triangle_quality_p01" in old_quality:
            old_value = float(old_quality["triangle_quality_p01"])
            new_value = float(quality.get("triangle_quality_p01", 0.0))
            if new_value + 1e-12 < old_value * (1.0 - quality_tolerance):
                failures.append(
                    f"{case['name']}: low triangle quality fell from {old_value:.6g} "
                    f"to {new_value:.6g}"
                )
    return failures


def _selected_cases(
    cases: list[BenchmarkCase],
    *,
    suite: str,
    patterns: list[str],
) -> list[BenchmarkCase]:
    suite_level = {"quick": 0, "standard": 1, "extended": 2}[suite]
    case_level = {"quick": 0, "standard": 1, "extended": 2}
    return [
        case
        for case in cases
        if case_level[case.suite] <= suite_level
        and (not patterns or any(pattern in case.name for pattern in patterns))
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark articraft SDK operations.")
    parser.add_argument("--suite", choices=("quick", "standard", "extended"), default="standard")
    parser.add_argument("--case", action="append", default=[], help="Run names containing text.")
    parser.add_argument("--rounds", type=int, help="Measured rounds per case.")
    parser.add_argument("--target-seconds", type=float, help="Minimum target time per round.")
    parser.add_argument("--output", type=Path, help="Write benchmark JSON to this path.")
    parser.add_argument("--compare", type=Path, help="Compare against benchmark JSON.")
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    parser.add_argument("--list", action="store_true", help="List cases without running them.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = _selected_cases(make_cases(), suite=args.suite, patterns=args.case)
    if args.list:
        for case in cases:
            print(f"{case.name}\t{case.group}\t{case.suite}")
        return 0
    if not cases:
        print("No benchmark cases matched.", file=sys.stderr)
        return 2
    rounds = args.rounds or {"quick": 3, "standard": 5, "extended": 3}[args.suite]
    target_seconds = (
        args.target_seconds or {"quick": 0.03, "standard": 0.15, "extended": 0.1}[args.suite]
    )
    if rounds < 1 or target_seconds <= 0.0:
        print("rounds and target seconds must be positive", file=sys.stderr)
        return 2
    report = run_benchmarks(cases, rounds=rounds, target_seconds=target_seconds)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")
    if args.compare is None:
        return 0
    baseline = json.loads(args.compare.read_text(encoding="utf-8"))
    failures = compare_results(
        report,
        baseline,
        quality_tolerance=float(args.quality_tolerance),
    )
    if failures:
        print("\nQuality regressions")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nNo measured mesh quality regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
