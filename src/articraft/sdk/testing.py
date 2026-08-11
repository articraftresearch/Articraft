from __future__ import annotations

import math
import statistics
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import trimesh
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from articraft.sdk._collision import (
    Bounds,
    CollisionQuery,
    DistanceFinding,
    GeometryConnectivityFinding,
    MeshCollisionKernel,
    Vec3,
    _pair_key,
)
from articraft.sdk._mesh.core import geometry_to_trimesh
from articraft.sdk._mesh.health import MeshHealthIssue, analyze_mesh_health
from articraft.sdk.errors import ValidationError
from articraft.sdk.joints import Articulation, ArticulationType, partition_articulations
from articraft.sdk.mass import resolve_mass
from articraft.sdk.materials import LIBRARY
from articraft.sdk.object import ArticulatedObject, Part, PartRef

DEFAULT_MESH_TOLERANCE = 0.001
DEFAULT_CONTACT_TOLERANCE = 1e-6
DEFAULT_OVERLAP_TOLERANCE = 0.005
DEFAULT_OVERLAP_VOLUME_TOLERANCE = 5e-7


class FailureKind(StrEnum):
    """Machine-readable category of a recorded test failure.

    The producer (the check method) assigns the kind when recording; consumers
    such as the compile-signal pipeline map it directly to guidance instead of
    re-classifying failure prose. Authored checks via ``check()``/``fail()``
    always record ``AUTHORED``.
    """

    MODEL_VALIDITY = "model_validity"
    SINGLE_ROOT = "single_root"
    MESH_HEALTH = "mesh_health"
    ISOLATED_PART = "isolated_part"
    DISCONNECTED_GEOMETRY = "disconnected_geometry"
    OVERLAP = "overlap"
    CONTACT = "contact"
    ARTICULATION_SEPARATION = "articulation_separation"
    MISSING_MASS = "missing_mass"
    AUTHORED = "authored"


@dataclass(frozen=True)
class TestFailure:
    __test__: ClassVar[bool] = False

    name: str
    details: str
    kind: FailureKind = FailureKind.AUTHORED


@dataclass(frozen=True)
class AllowedOverlap:
    part_a: str
    part_b: str
    reason: str
    shape_a: str
    shape_b: str


@dataclass(frozen=True)
class AllowedMeshIssues:
    part: str
    shape: str
    issues: tuple[MeshHealthIssue, ...]
    reason: str


@dataclass(frozen=True)
class TestMetric:
    __test__: ClassVar[bool] = False

    name: str
    value: float
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    passed: bool | None = None
    details: str = ""


@dataclass(frozen=True)
class TestArtifact:
    __test__: ClassVar[bool] = False

    name: str
    path: str
    kind: str
    caption: str = ""


@dataclass(frozen=True)
class GeometryMetrics:
    bounds: Bounds
    dimensions: Vec3
    centroid: Vec3
    triangle_count: int
    component_count: int
    watertight: bool
    surface_area: float
    signed_volume: float
    orientation: str


@dataclass(frozen=True)
class PoseSample:
    positions: tuple[tuple[str, float], ...]
    label: str = ""

    def as_dict(self) -> dict[str, float]:
        return dict(self.positions)


@dataclass(frozen=True)
class TestReport:
    __test__: ClassVar[bool] = False

    passed: bool
    checks_run: int
    checks: tuple[str, ...]
    failures: tuple[TestFailure, ...]
    warnings: tuple[str, ...] = ()
    allowances: tuple[str, ...] = ()
    allowed_isolated_parts: tuple[str, ...] = ()
    allowed_overlaps: tuple[AllowedOverlap, ...] = ()
    allowed_mesh_issues: tuple[AllowedMeshIssues, ...] = ()
    metrics: tuple[TestMetric, ...] = ()
    artifacts: tuple[TestArtifact, ...] = ()


@dataclass
class TestContext:
    __test__: ClassVar[bool] = False

    model: ArticulatedObject
    mesh_tolerance: float = DEFAULT_MESH_TOLERANCE

    checks_run: int = 0
    _checks: list[str] = field(default_factory=list)
    _failures: list[TestFailure] = field(default_factory=list)
    _warnings: list[str] = field(default_factory=list)
    _allowances: list[str] = field(default_factory=list)
    _allowed_isolated_parts: dict[str, str] = field(default_factory=dict)
    _allowed_overlaps: list[AllowedOverlap] = field(default_factory=list)
    _allowed_mesh_issues: list[AllowedMeshIssues] = field(default_factory=list)
    _pose: dict[str, float] = field(default_factory=dict)
    _metrics: list[TestMetric] = field(default_factory=list)
    _artifacts: list[TestArtifact] = field(default_factory=list)
    _kernel_cache: MeshCollisionKernel | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, ArticulatedObject):
            raise ValidationError("TestContext model must be an ArticulatedObject")
        self.mesh_tolerance = float(self.mesh_tolerance)
        if self.mesh_tolerance <= 0.0 or not math.isfinite(self.mesh_tolerance):
            raise ValidationError("mesh_tolerance must be a positive finite number")

    def report(self) -> TestReport:
        failures = tuple(self._failures)
        return TestReport(
            passed=not failures,
            checks_run=self.checks_run,
            checks=tuple(self._checks),
            failures=failures,
            warnings=tuple(self._warnings),
            allowances=tuple(self._allowances),
            allowed_isolated_parts=tuple(self._allowed_isolated_parts),
            allowed_overlaps=tuple(self._allowed_overlaps),
            allowed_mesh_issues=tuple(self._allowed_mesh_issues),
            metrics=tuple(self._metrics),
            artifacts=tuple(self._artifacts),
        )

    def check(self, name: str, ok: bool, details: str = "") -> bool:
        return self._record(str(name), bool(ok), str(details or ""))

    def fail(self, name: str, details: str) -> bool:
        return self._record(str(name), False, str(details or ""))

    def warn(self, text: str) -> None:
        text = str(text or "").strip()
        if text and text not in self._warnings:
            self._warnings.append(text)

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        details: str = "",
    ) -> TestMetric:
        metric = TestMetric(
            name=_plain_name(name, "metric name"),
            value=_finite(value, "metric value"),
            unit=str(unit).strip(),
            details=str(details or ""),
        )
        self._metrics.append(metric)
        return metric

    def expect_metric(
        self,
        name: str,
        value: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        unit: str = "",
        details: str = "",
    ) -> bool:
        metric_name = _plain_name(name, "metric name")
        measured = _finite(value, "metric value")
        lower = None if minimum is None else _finite(minimum, "metric minimum")
        upper = None if maximum is None else _finite(maximum, "metric maximum")
        if lower is not None and upper is not None and upper < lower:
            raise ValidationError("metric maximum must be greater than or equal to minimum")
        ok = (lower is None or measured >= lower) and (upper is None or measured <= upper)
        self._metrics.append(
            TestMetric(
                name=metric_name,
                value=measured,
                unit=str(unit).strip(),
                minimum=lower,
                maximum=upper,
                passed=ok,
                details=str(details or ""),
            )
        )
        bounds = f"minimum={lower!r} maximum={upper!r}"
        return self._record(metric_name, ok, f"value={measured:.9g} {bounds} {details}".strip())

    def attach_artifact(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        kind: str | None = None,
        caption: str = "",
    ) -> TestArtifact:
        raw = Path(path)
        if raw.is_absolute() or ".." in raw.parts:
            raise ValidationError("artifact path must be relative and stay inside the workspace")
        resolved = (Path.cwd() / raw).resolve()
        try:
            resolved.relative_to(Path.cwd().resolve())
        except ValueError as exc:
            raise ValidationError("artifact path must stay inside the workspace") from exc
        if not resolved.is_file():
            raise ValidationError(f"artifact file does not exist: {raw.as_posix()}")
        artifact_kind = str(kind or _artifact_kind(raw)).strip().lower()
        if artifact_kind not in {"image", "json", "csv", "text"}:
            raise ValidationError("artifact kind must be image, json, csv, or text")
        artifact = TestArtifact(
            name=_plain_name(name or raw.stem, "artifact name"),
            path=raw.as_posix(),
            kind=artifact_kind,
            caption=str(caption or "").strip(),
        )
        self._artifacts.append(artifact)
        return artifact

    def allow_overlap(
        self,
        part_a: PartRef,
        part_b: PartRef,
        *,
        reason: str,
        shape_a: str,
        shape_b: str,
    ) -> None:
        part_a_name = _part_name(part_a, field_name="part_a")
        part_b_name = _part_name(part_b, field_name="part_b")
        self.model.get_part(part_a_name)
        self.model.get_part(part_b_name)
        shape_a = str(shape_a).strip()
        shape_b = str(shape_b).strip()
        self.model.get_part(part_a_name).get_shape(shape_a)
        self.model.get_part(part_b_name).get_shape(shape_b)
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("allow_overlap requires a non-empty reason")

        part_a_name, part_b_name, shape_a, shape_b = _canonical_selectors(
            part_a_name, part_b_name, shape_a, shape_b
        )
        allowed = AllowedOverlap(part_a_name, part_b_name, reason, shape_a, shape_b)
        self._allowed_overlaps.append(allowed)
        shape_text = f", shape_a={shape_a!r}, shape_b={shape_b!r}"
        self._allowances.append(
            f"allow_overlap({part_a_name!r}, {part_b_name!r}{shape_text}): {reason}"
        )

    def allow_isolated_part(self, part: PartRef, *, reason: str) -> None:
        part_name = _part_name(part, field_name="part")
        self.model.get_part(part_name)
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("allow_isolated_part requires a non-empty reason")
        self._allowed_isolated_parts[part_name] = reason
        self._allowances.append(f"allow_isolated_part({part_name!r}): {reason}")

    def allow_mesh_issues(
        self,
        part: PartRef,
        *,
        shape: str,
        issues: Sequence[MeshHealthIssue | str],
        reason: str,
    ) -> None:
        part_name = _part_name(part, field_name="part")
        model_part = self.model.get_part(part_name)
        shape_name = str(shape).strip()
        model_part.get_shape(shape_name)
        if isinstance(issues, (str, MeshHealthIssue)):
            issue_values = (MeshHealthIssue(issues),)
        else:
            try:
                issue_values = tuple(MeshHealthIssue(issue) for issue in issues)
            except (TypeError, ValueError) as exc:
                raise ValueError("issues must contain MeshHealthIssue values") from exc
        issue_values = tuple(dict.fromkeys(issue_values))
        if not issue_values:
            raise ValueError("allow_mesh_issues requires at least one issue")
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("allow_mesh_issues requires a non-empty reason")
        allowed = AllowedMeshIssues(part_name, shape_name, issue_values, reason)
        self._allowed_mesh_issues.append(allowed)
        issue_text = ", ".join(repr(issue.value) for issue in issue_values)
        self._allowances.append(
            f"allow_mesh_issues({part_name!r}, shape={shape_name!r}, "
            f"issues=({issue_text})): {reason}"
        )

    @contextmanager
    def pose(
        self,
        articulation_positions: Mapping[object, float] | None = None,
        **kwargs: float,
    ) -> Iterator[None]:
        previous = dict(self._pose)
        updates: dict[str, float] = {}
        articulation_names = {item.name for item in self.model.articulations}
        for key, value in {**dict(articulation_positions or {}), **kwargs}.items():
            name = _articulation_name(key)
            if name not in articulation_names:
                raise ValidationError(f"unknown articulation: {name!r}")
            self._reject_unposable(name)
            updates[name] = _finite(value, "articulation position")
        try:
            self._pose.update(updates)
            yield
        finally:
            self._pose = previous

    def _reject_unposable(self, name: str) -> None:
        """A driven or loop-closing joint has no value of its own to pose.

        Accepting one silently would sweep N identical poses and pass every
        check vacuously, so it is an error: pose or sample the driving joint.
        """

        articulation = self.model.get_articulation(name)
        if articulation.drive is not None:
            raise ValidationError(
                f"articulation {name!r} is driven; its value is solved from the "
                "mechanism, so pose or sample the joint that drives it instead"
            )
        _tree, loops = partition_articulations(self.model.articulations)
        if any(item.name == name for item in loops):
            raise ValidationError(
                f"articulation {name!r} closes a loop and cannot be posed; pose "
                "the tree joints that move its parts instead"
            )

    def part_world_position(self, part: PartRef) -> Vec3:
        return self._kernel().part_world_position(_part_name(part, field_name="part"), self._pose)

    def part_world_bounds(self, part: PartRef) -> Bounds:
        return self._kernel().part_world_bounds(_part_name(part, field_name="part"), self._pose)

    def shape_world_bounds(self, part: PartRef, shape: str) -> Bounds:
        return self._kernel().shape_world_bounds(
            _part_name(part, field_name="part"), shape, self._pose
        )

    def part_world_point(self, part: PartRef, point: Sequence[float]) -> Vec3:
        part_name = _part_name(part, field_name="part")
        local = np.asarray(_vec3(point, "point"), dtype=np.float64)
        transform = self._kernel().world_transforms(self._pose).get(part_name)
        if transform is None:
            raise ValidationError(f"unknown part: {part_name!r}")
        world = transform @ np.asarray([*local, 1.0], dtype=np.float64)
        return (float(world[0]), float(world[1]), float(world[2]))

    def measure_geometry(
        self,
        part: PartRef | None = None,
        *,
        shape: str | None = None,
    ) -> GeometryMetrics:
        meshes = self._selected_world_meshes(part, shape)
        vertices = np.concatenate([np.asarray(mesh.vertices) for mesh in meshes], axis=0)
        combined = trimesh.util.concatenate(meshes)
        bounds_array = np.asarray([vertices.min(axis=0), vertices.max(axis=0)])
        dimensions = bounds_array[1] - bounds_array[0]
        area = sum(float(mesh.area) for mesh in meshes)
        signed_volume = sum(_signed_mesh_volume(mesh) for mesh in meshes)
        health_reports = [
            analyze_mesh_health(mesh, tolerance=self.mesh_tolerance) for mesh in meshes
        ]
        components = sum(report.component_count for report in health_reports)
        watertight = all(report.watertight for report in health_reports)
        if not watertight:
            orientation = "open"
        elif signed_volume > 0.0:
            orientation = "outward"
        elif signed_volume < 0.0:
            orientation = "inward"
        else:
            orientation = "degenerate"
        return GeometryMetrics(
            bounds=(
                tuple(float(value) for value in bounds_array[0]),
                tuple(float(value) for value in bounds_array[1]),
            ),  # type: ignore[arg-type]
            dimensions=tuple(float(value) for value in dimensions),  # type: ignore[arg-type]
            centroid=tuple(float(value) for value in combined.centroid),  # type: ignore[arg-type]
            triangle_count=sum(len(mesh.faces) for mesh in meshes),
            component_count=components,
            watertight=watertight,
            surface_area=area,
            signed_volume=signed_volume,
            orientation=orientation,
        )

    def sample_joint(
        self,
        articulation: str | Articulation,
        positions: Sequence[float] | None = None,
        *,
        samples: int = 5,
    ) -> tuple[PoseSample, ...]:
        joint = self.model.get_articulation(articulation)
        self._reject_unposable(joint.name)
        values = (
            _articulation_sweep_values(joint, max(2, int(samples)))
            if positions is None
            else [_finite(value, "joint sample") for value in positions]
        )
        if not values:
            raise ValidationError("positions must contain at least one joint value")
        return tuple(
            PoseSample(
                positions=tuple(sorted({**self._pose, joint.name: value}.items())),
                label=f"{joint.name}={value:.6g}",
            )
            for value in values
        )

    def track_point(
        self,
        part: PartRef,
        point: Sequence[float],
        poses: Sequence[PoseSample | Mapping[object, float]],
    ) -> tuple[Vec3, ...]:
        tracked: list[Vec3] = []
        for pose in _pose_dicts(poses):
            with self.pose(pose):
                tracked.append(self.part_world_point(part, point))
        return tuple(tracked)

    def expect_bounds(
        self,
        part: PartRef | None = None,
        *,
        shape: str | None = None,
        minimum: Sequence[float] | None = None,
        maximum: Sequence[float] | None = None,
        tolerance: float = 0.0,
        name: str | None = None,
    ) -> bool:
        if minimum is None and maximum is None:
            raise ValidationError("expect_bounds requires minimum, maximum, or both")
        metrics = self.measure_geometry(part, shape=shape)
        expected_min = None if minimum is None else _vec3(minimum, "minimum")
        expected_max = None if maximum is None else _vec3(maximum, "maximum")
        tolerance = _non_negative(tolerance, "tolerance")
        ok = True
        if expected_min is not None:
            ok = ok and all(
                abs(metrics.bounds[0][axis] - expected_min[axis]) <= tolerance for axis in range(3)
            )
        if expected_max is not None:
            ok = ok and all(
                abs(metrics.bounds[1][axis] - expected_max[axis]) <= tolerance for axis in range(3)
            )
        selector = _geometry_selector(part, shape)
        return self._record(
            name or f"expect_bounds({selector})",
            ok,
            f"bounds={metrics.bounds!r} expected_min={expected_min!r} "
            f"expected_max={expected_max!r} tolerance={tolerance:.6g}",
        )

    def expect_radial_extent(
        self,
        part: PartRef | None = None,
        *,
        shape: str | None = None,
        axis: Sequence[float] = (0.0, 0.0, 1.0),
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        minimum: float | None = None,
        maximum: float | None = None,
        name: str | None = None,
    ) -> bool:
        direction = np.asarray(_vec3(axis, "axis"), dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            raise ValidationError("axis must be non-zero")
        direction /= length
        center = np.asarray(_vec3(origin, "origin"), dtype=np.float64)
        vertices = self._selected_world_vertices(part, shape)
        offsets = vertices - center
        projected = offsets - np.outer(offsets @ direction, direction)
        extent = float(np.linalg.norm(projected, axis=1).max(initial=0.0))
        selector = _geometry_selector(part, shape)
        return self.expect_metric(
            name or f"radial_extent({selector})",
            extent,
            minimum=minimum,
            maximum=maximum,
            unit="m",
            details=f"axis={tuple(direction)!r} origin={tuple(center)!r}",
        )

    def expect_component_count(
        self,
        expected: int,
        part: PartRef | None = None,
        *,
        shape: str | None = None,
        name: str | None = None,
    ) -> bool:
        expected = int(expected)
        if expected < 1:
            raise ValidationError("expected component count must be at least 1")
        metrics = self.measure_geometry(part, shape=shape)
        selector = _geometry_selector(part, shape)
        return self._record(
            name or f"expect_component_count({selector})",
            metrics.component_count == expected,
            f"components={metrics.component_count} expected={expected}",
        )

    def expect_watertight(
        self,
        part: PartRef | None = None,
        *,
        shape: str | None = None,
        name: str | None = None,
    ) -> bool:
        metrics = self.measure_geometry(part, shape=shape)
        selector = _geometry_selector(part, shape)
        return self._record(
            name or f"expect_watertight({selector})",
            metrics.watertight,
            f"watertight={metrics.watertight} components={metrics.component_count}",
        )

    def expect_positive_volume(
        self,
        part: PartRef | None = None,
        *,
        shape: str | None = None,
        minimum: float = 0.0,
        name: str | None = None,
    ) -> bool:
        minimum = _non_negative(minimum, "minimum")
        metrics = self.measure_geometry(part, shape=shape)
        selector = _geometry_selector(part, shape)
        metric_name = name or f"signed_volume({selector})"
        ok = metrics.signed_volume > minimum
        details = f"orientation={metrics.orientation} watertight={metrics.watertight}"
        self._metrics.append(
            TestMetric(
                name=metric_name,
                value=metrics.signed_volume,
                unit="m^3",
                minimum=minimum,
                passed=ok,
                details=details,
            )
        )
        return self._record(
            metric_name,
            ok,
            f"value={metrics.signed_volume:.9g} must be greater than {minimum:.9g} {details}",
        )

    def expect_symmetry(
        self,
        part: PartRef | None = None,
        *,
        shape: str | None = None,
        plane_origin: Sequence[float] = (0.0, 0.0, 0.0),
        plane_normal: Sequence[float] = (1.0, 0.0, 0.0),
        tolerance: float = 0.001,
        name: str | None = None,
    ) -> bool:
        origin = np.asarray(_vec3(plane_origin, "plane_origin"), dtype=np.float64)
        normal = np.asarray(_vec3(plane_normal, "plane_normal"), dtype=np.float64)
        length = float(np.linalg.norm(normal))
        if length <= 0.0:
            raise ValidationError("plane_normal must be non-zero")
        normal /= length
        tolerance = _non_negative(tolerance, "tolerance")
        vertices = self._selected_world_vertices(part, shape)
        if len(vertices) > 20_000:
            vertices = vertices[np.linspace(0, len(vertices) - 1, 20_000, dtype=int)]
        reflected = vertices - 2.0 * np.outer((vertices - origin) @ normal, normal)
        distances, _indices = cKDTree(vertices).query(reflected, workers=1)
        deviation = float(np.max(distances, initial=0.0))
        selector = _geometry_selector(part, shape)
        return self.expect_metric(
            name or f"symmetry_deviation({selector})",
            deviation,
            maximum=tolerance,
            unit="m",
            details=f"plane_origin={tuple(origin)!r} plane_normal={tuple(normal)!r}",
        )

    def distance_between(
        self,
        part_a: PartRef,
        part_b: PartRef,
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
    ) -> DistanceFinding:
        return self._distance_query(part_a, part_b, shape_a=shape_a, shape_b=shape_b)

    def expect_no_collision_at_poses(
        self,
        part_a: PartRef,
        part_b: PartRef,
        poses: Sequence[PoseSample | Mapping[object, float]],
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
        name: str | None = None,
    ) -> bool:
        failures: list[str] = []
        for index, pose in enumerate(_pose_dicts(poses)):
            with self.pose(pose):
                query = self._collision_query(part_a, part_b, shape_a=shape_a, shape_b=shape_b)
            if query.collided:
                failures.append(f"sample={index} pose={pose!r} {_collision_details(query)}")
        return self._record(
            name
            or f"expect_no_collision_at_poses({_part_name(part_a, field_name='part_a')},"
            f"{_part_name(part_b, field_name='part_b')})",
            not failures,
            "all samples clear" if not failures else "\n".join(failures[:10]),
            kind=FailureKind.OVERLAP,
        )

    def expect_distance_at_poses(
        self,
        part_a: PartRef,
        part_b: PartRef,
        poses: Sequence[PoseSample | Mapping[object, float]],
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
        minimum: float = 0.0,
        maximum: float | None = None,
        name: str | None = None,
    ) -> bool:
        minimum = _non_negative(minimum, "minimum")
        maximum = None if maximum is None else _non_negative(maximum, "maximum")
        if maximum is not None and maximum < minimum:
            raise ValidationError("maximum must be greater than or equal to minimum")
        failures: list[str] = []
        measured: list[float] = []
        for index, pose in enumerate(_pose_dicts(poses)):
            with self.pose(pose):
                result = self._distance_query(part_a, part_b, shape_a=shape_a, shape_b=shape_b)
            measured.append(result.distance)
            if result.distance < minimum or (maximum is not None and result.distance > maximum):
                failures.append(f"sample={index} pose={pose!r} {_distance_details(result)}")
        return self._record(
            name
            or f"expect_distance_at_poses({_part_name(part_a, field_name='part_a')},"
            f"{_part_name(part_b, field_name='part_b')})",
            not failures,
            f"range=({min(measured):.6g},{max(measured):.6g}) minimum={minimum:.6g} "
            f"maximum={maximum!r}"
            if not failures
            else "\n".join(failures[:10]),
        )

    def expect_contact_at_poses(
        self,
        part_a: PartRef,
        part_b: PartRef,
        poses: Sequence[PoseSample | Mapping[object, float]],
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
        contact_tol: float = DEFAULT_CONTACT_TOLERANCE,
        name: str | None = None,
    ) -> bool:
        tolerance = _non_negative(contact_tol, "contact_tol")
        failures: list[str] = []
        for index, pose in enumerate(_pose_dicts(poses)):
            with self.pose(pose):
                result = self._distance_query(part_a, part_b, shape_a=shape_a, shape_b=shape_b)
            if not result.collided and result.distance > tolerance:
                failures.append(f"sample={index} pose={pose!r} {_distance_details(result)}")
        return self._record(
            name
            or f"expect_contact_at_poses({_part_name(part_a, field_name='part_a')},"
            f"{_part_name(part_b, field_name='part_b')})",
            not failures,
            "contact held at every sample" if not failures else "\n".join(failures[:10]),
            kind=FailureKind.CONTACT,
        )

    def expect_within_at_poses(
        self,
        inner_part: PartRef,
        outer_part: PartRef,
        poses: Sequence[PoseSample | Mapping[object, float]],
        *,
        inner_shape: str | None = None,
        outer_shape: str | None = None,
        axes: str | Sequence[str] = "xyz",
        margin: float = 0.0,
        name: str | None = None,
    ) -> bool:
        inner_name = _part_name(inner_part, field_name="inner_part")
        outer_name = _part_name(outer_part, field_name="outer_part")
        axis_names = _axis_names(axes)
        margin = _non_negative(margin, "margin")
        failures: list[str] = []
        for sample_index, pose in enumerate(_pose_dicts(poses)):
            with self.pose(pose):
                inner_bounds = self._bounds(inner_name, inner_shape)
                outer_bounds = self._bounds(outer_name, outer_shape)
            failed_axes = [
                axis_name
                for axis_name in axis_names
                if (
                    inner_bounds[0][_axis_index(axis_name)]
                    < outer_bounds[0][_axis_index(axis_name)] - margin
                    or inner_bounds[1][_axis_index(axis_name)]
                    > outer_bounds[1][_axis_index(axis_name)] + margin
                )
            ]
            if failed_axes:
                failures.append(
                    f"sample={sample_index} pose={pose!r} axes={failed_axes!r} "
                    f"inner_bounds={inner_bounds!r} outer_bounds={outer_bounds!r}"
                )
        return self._record(
            name or f"expect_within_at_poses({inner_name},{outer_name})",
            not failures,
            "contained at every sample" if not failures else "\n".join(failures[:10]),
        )

    def expect_no_collision(
        self,
        part_a: PartRef,
        part_b: PartRef,
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
        name: str | None = None,
    ) -> bool:
        query = self._collision_query(part_a, part_b, shape_a=shape_a, shape_b=shape_b)
        check_name = name or _check_name("expect_no_collision", query)
        return self._record(
            check_name, not query.collided, _collision_details(query), kind=FailureKind.OVERLAP
        )

    def expect_collision(
        self,
        part_a: PartRef,
        part_b: PartRef,
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
        name: str | None = None,
    ) -> bool:
        query = self._collision_query(part_a, part_b, shape_a=shape_a, shape_b=shape_b)
        check_name = name or _check_name("expect_collision", query)
        return self._record(
            check_name, query.collided, _collision_details(query), kind=FailureKind.CONTACT
        )

    def expect_contact(
        self,
        part_a: PartRef,
        part_b: PartRef,
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
        contact_tol: float = DEFAULT_CONTACT_TOLERANCE,
        name: str | None = None,
    ) -> bool:
        result = self._distance_query(part_a, part_b, shape_a=shape_a, shape_b=shape_b)
        contact_tol = _non_negative(contact_tol, "contact_tol")
        check_name = name or _check_name("expect_contact", result)
        return self._record(
            check_name,
            result.collided or result.distance <= contact_tol,
            f"{_distance_details(result)} contact_tol={contact_tol:.6g}",
            kind=FailureKind.CONTACT,
        )

    def expect_distance(
        self,
        part_a: PartRef,
        part_b: PartRef,
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
        min_distance: float = 0.0,
        max_distance: float | None = None,
        name: str | None = None,
    ) -> bool:
        result = self._distance_query(part_a, part_b, shape_a=shape_a, shape_b=shape_b)
        min_distance = _non_negative(min_distance, "min_distance")
        maximum = None if max_distance is None else _non_negative(max_distance, "max_distance")
        check_name = name or _check_name("expect_distance", result)
        if maximum is not None and maximum < min_distance:
            return self._record(check_name, False, "max_distance must be >= min_distance")
        ok = result.distance >= min_distance and (maximum is None or result.distance <= maximum)
        upper = "inf" if maximum is None else f"{maximum:.6g}"
        return self._record(
            check_name,
            ok,
            f"{_distance_details(result)} min_distance={min_distance:.6g} max_distance={upper}",
        )

    def expect_gap(
        self,
        positive_part: PartRef,
        negative_part: PartRef,
        *,
        axis: str,
        positive_shape: str | None = None,
        negative_shape: str | None = None,
        min_gap: float | None = None,
        max_gap: float | None = None,
        max_penetration: float | None = None,
        name: str | None = None,
    ) -> bool:
        positive_name = _part_name(positive_part, field_name="positive_part")
        negative_name = _part_name(negative_part, field_name="negative_part")
        axis_name = _axis_name(axis)
        positive_bounds = self._bounds(positive_name, positive_shape)
        negative_bounds = self._bounds(negative_name, negative_shape)
        axis_index = _axis_index(axis_name)
        gap = positive_bounds[0][axis_index] - negative_bounds[1][axis_index]
        if min_gap is None:
            penetration = (
                0.0
                if max_penetration is None
                else _non_negative(max_penetration, "max_penetration")
            )
            minimum = -penetration
        else:
            minimum = _finite(min_gap, "min_gap")
            penetration = max(0.0, -minimum)
        maximum = None if max_gap is None else _finite(max_gap, "max_gap")
        selectors = _selector_text(
            positive_shape=positive_shape,
            negative_shape=negative_shape,
        )
        check_name = name or (
            f"expect_gap({positive_name},{negative_name},axis={axis_name}{selectors})"
        )
        if maximum is not None and maximum < minimum:
            return self._record(check_name, False, "max_gap must be >= min_gap")
        ok = gap >= minimum and (maximum is None or gap <= maximum)
        upper = "inf" if maximum is None else f"{maximum:.6g}"
        return self._record(
            check_name,
            ok,
            f"mesh_gap_{axis_name}={gap:.6g} min_gap={minimum:.6g} "
            f"max_gap={upper} max_penetration={penetration:.6g}",
        )

    def expect_within(
        self,
        inner_part: PartRef,
        outer_part: PartRef,
        *,
        inner_shape: str | None = None,
        outer_shape: str | None = None,
        axes: str | Sequence[str] = "xy",
        margin: float = 0.0,
        name: str | None = None,
    ) -> bool:
        inner_name = _part_name(inner_part, field_name="inner_part")
        outer_name = _part_name(outer_part, field_name="outer_part")
        inner_bounds = self._bounds(inner_name, inner_shape)
        outer_bounds = self._bounds(outer_name, outer_shape)
        axis_names = _axis_names(axes)
        margin = _non_negative(margin, "margin")
        selectors = _selector_text(inner_shape=inner_shape, outer_shape=outer_shape)
        check_name = name or (
            f"expect_within({inner_name},{outer_name},axes={''.join(axis_names)}{selectors})"
        )
        ok = True
        details: list[str] = []
        for axis_name in axis_names:
            index = _axis_index(axis_name)
            axis_ok = (
                inner_bounds[0][index] >= outer_bounds[0][index] - margin
                and inner_bounds[1][index] <= outer_bounds[1][index] + margin
            )
            ok = ok and axis_ok
            details.append(
                f"{axis_name}=({inner_bounds[0][index]:.6g},{inner_bounds[1][index]:.6g}) "
                f"within ({outer_bounds[0][index]:.6g},{outer_bounds[1][index]:.6g})"
            )
        return self._record(check_name, ok, " ".join(details) + f" margin={margin:.6g}")

    def expect_overlap(
        self,
        part_a: PartRef,
        part_b: PartRef,
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
        axes: str | Sequence[str] = "xy",
        min_overlap: float = 0.0,
        name: str | None = None,
    ) -> bool:
        part_a_name = _part_name(part_a, field_name="part_a")
        part_b_name = _part_name(part_b, field_name="part_b")
        bounds_a = self._bounds(part_a_name, shape_a)
        bounds_b = self._bounds(part_b_name, shape_b)
        axis_names = _axis_names(axes)
        minimum = _non_negative(min_overlap, "min_overlap")
        selectors = _selector_text(shape_a=shape_a, shape_b=shape_b)
        check_name = name or (
            f"expect_overlap({part_a_name},{part_b_name},axes={''.join(axis_names)}{selectors})"
        )
        ok = True
        details: list[str] = []
        for axis_name in axis_names:
            index = _axis_index(axis_name)
            overlap = min(bounds_a[1][index], bounds_b[1][index]) - max(
                bounds_a[0][index], bounds_b[0][index]
            )
            ok = ok and overlap >= minimum
            details.append(f"overlap_{axis_name}={overlap:.6g}")
        return self._record(check_name, ok, " ".join(details) + f" min_overlap={minimum:.6g}")

    def check_model_valid(self) -> bool:
        try:
            self.model.validate()
        except Exception as exc:
            return self._record(
                "check_model_valid",
                False,
                f"{type(exc).__name__}: {exc}",
                kind=FailureKind.MODEL_VALIDITY,
            )
        return self._record("check_model_valid", True)

    def check_single_root_part(self) -> bool:
        roots = _root_part_names(self.model)
        if len(roots) != 1:
            return self._record(
                "check_single_root_part",
                False,
                f"Expected exactly one root part, found {len(roots)}: {roots!r}",
                kind=FailureKind.SINGLE_ROOT,
            )
        return self._record("check_single_root_part", True)

    def fail_if_mesh_unhealthy(
        self,
        part: PartRef | None = None,
        *,
        shape: str | None = None,
        name: str | None = None,
    ) -> bool:
        check_name = name or f"fail_if_mesh_unhealthy({_geometry_selector(part, shape)})"
        findings: list[str] = []
        allowed_findings: list[str] = []
        for part_name, shape_name, geometry in self._selected_geometries(part, shape):
            report = analyze_mesh_health(geometry, tolerance=self.mesh_tolerance)
            for finding in report.findings:
                line = (
                    f"part={part_name!r} shape={shape_name!r} "
                    f"issue={finding.issue.value!r} count={finding.count}"
                )
                if finding.bounds is not None:
                    line += f" bounds={finding.bounds!r}"
                if finding.details:
                    line += f" details={finding.details}"
                if self._mesh_issue_is_allowed(part_name, shape_name, finding.issue):
                    allowed_findings.append(line)
                else:
                    findings.append(line)
        if allowed_findings:
            self.warn(
                "Mesh health findings detected but allowed by justification:\n"
                + "\n".join(allowed_findings)
            )
        return self._record(
            check_name,
            not findings,
            "" if not findings else "Unhealthy mesh geometry detected:\n" + "\n".join(findings),
            kind=FailureKind.MESH_HEALTH,
        )

    def fail_if_parts_have_no_mass(self) -> bool:
        """Every part must be weighable when physics is enabled.

        A part is weighable when its shapes say what they are made of, or when the
        part carries an explicit mass or density. Anything else would export a
        silently invented weight.
        """

        unresolved: list[str] = []
        for part in self.model.parts:
            try:
                resolve_mass(
                    part.mass_properties,
                    [
                        (geometry_to_trimesh(shape.geometry, self.mesh_tolerance), shape.material)
                        for shape in part._iter_shapes()
                    ],
                    part_name=part.name,
                )
            except Exception as exc:
                unresolved.append(str(exc))
        if unresolved:
            materials = ", ".join(f"{item.name} ({item.density:g})" for item in LIBRARY)
            return self._record(
                "fail_if_parts_have_no_mass",
                False,
                "Physics is enabled, so every part must be weighable.\n"
                + "\n".join(unresolved)
                + f".\nMaterials and their densities in kg/m^3: {materials}.",
                kind=FailureKind.MISSING_MASS,
            )
        return self._record("fail_if_parts_have_no_mass", True)

    def fail_if_isolated_parts(
        self,
        *,
        contact_tol: float = DEFAULT_CONTACT_TOLERANCE,
        name: str | None = None,
    ) -> bool:
        contact_tol = _non_negative(contact_tol, "contact_tol")
        check_name = name or (
            "fail_if_isolated_parts()"
            if contact_tol == DEFAULT_CONTACT_TOLERANCE
            else f"fail_if_isolated_parts(contact_tol={contact_tol:.4g})"
        )
        part_names = [part.name for part in self.model.parts]
        if len(part_names) <= 1:
            return self._record(check_name, True)

        adjacency: dict[str, set[str]] = {part_name: set() for part_name in part_names}
        distances = self._kernel().pair_distances(self._pose)
        for distance in distances:
            if distance.collided or distance.distance <= contact_tol:
                adjacency[distance.part_a].add(distance.part_b)
                adjacency[distance.part_b].add(distance.part_a)

        reachable = _reachable(_root_part_names(self.model), adjacency)
        floating = set(part_names) - reachable
        groups = _groups(floating, adjacency)
        remaining = [
            group
            for group in groups
            if not all(part_name in self._allowed_isolated_parts for part_name in group)
        ]
        allowed = [group for group in groups if group not in remaining]
        if allowed:
            allowed_names = sorted(part_name for group in allowed for part_name in group)
            self.warn(
                "Isolated parts detected but allowed by justification: "
                + ", ".join(repr(item) for item in allowed_names)
            )
        if not remaining:
            return self._record(check_name, True)

        lines: list[str] = []
        for group in remaining[:10]:
            nearest_text = "nearest=unknown"
            candidates = [
                item
                for item in distances
                if (
                    (item.part_a in group and item.part_b in reachable)
                    or (item.part_b in group and item.part_a in reachable)
                )
            ]
            if candidates:
                distance = min(candidates, key=lambda item: item.distance)
                root_part = distance.part_b if distance.part_a in group else distance.part_a
                nearest_text = (
                    f"nearest_root_part={root_part!r} nearest_gap={distance.distance:.6g}m"
                )
            lines.append(f"floating_group={sorted(group)!r} {nearest_text}")
        return self._record(
            check_name,
            False,
            "Isolated parts detected (physically disconnected from the root):\n" + "\n".join(lines),
            kind=FailureKind.ISOLATED_PART,
        )

    def fail_if_part_contains_disconnected_geometry_islands(
        self,
        *,
        contact_tol: float = DEFAULT_CONTACT_TOLERANCE,
        name: str | None = None,
    ) -> bool:
        return self._check_disconnected_geometry(contact_tol=contact_tol, name=name, warn=False)

    def warn_if_part_contains_disconnected_geometry_islands(
        self,
        *,
        contact_tol: float = DEFAULT_CONTACT_TOLERANCE,
        name: str | None = None,
    ) -> bool:
        return self._check_disconnected_geometry(contact_tol=contact_tol, name=name, warn=True)

    def fail_if_articulation_separates_child(
        self,
        *,
        samples: int = 4,
        gap_tol: float = 0.003,
        name: str | None = None,
    ) -> bool:
        """Fail if a hinge/pivot pulls its moving child away from its parent.

        For every REVOLUTE and CONTINUOUS articulation, this samples the joint
        across its motion range and measures the closest distance between the child
        part and its parent. A real hinge or pivot keeps them touching throughout;
        if the child instead separates by more than `gap_tol` beyond its rest gap as
        the joint moves, the joint origin/axis is misplaced -- the classic "lid that
        floats off when opened". PRISMATIC joints are exempt, since their parts
        often separate on purpose (lift-off kettles, sliding drawers).

        The other geometry checks only look at the rest pose, so this is what
        catches articulation defects that appear only in motion.
        """
        samples = max(2, int(samples))
        gap_tol = _non_negative(gap_tol, "gap_tol")
        check_name = name or f"fail_if_articulation_separates_child(gap_tol={gap_tol:.6g})"
        findings: list[str] = []
        _tree, loop_closures = partition_articulations(self.model.articulations)
        loop_names = {item.name for item in loop_closures}
        for articulation in self.model.articulations:
            if articulation.articulation_type not in (
                ArticulationType.REVOLUTE,
                ArticulationType.CONTINUOUS,
            ):
                continue
            # A driven joint has no value of its own to sweep, and a loop
            # closure never places its child; both move when the joints that
            # drive them are swept, so skipping them here loses no coverage a
            # posable sweep could give.
            if articulation.drive is not None or articulation.name in loop_names:
                continue
            parent = _part_name(articulation.parent, field_name="parent")
            child = _part_name(articulation.child, field_name="child")
            values = _articulation_sweep_values(articulation, samples)
            with self.pose({articulation.name: values[0]}):
                rest_gap = self.distance_between(parent, child).distance
            worst_gap, worst_value = rest_gap, values[0]
            for value in values[1:]:
                with self.pose({articulation.name: value}):
                    gap = self.distance_between(parent, child).distance
                if gap > worst_gap:
                    worst_gap, worst_value = gap, value
            if worst_gap - rest_gap > gap_tol:
                findings.append(
                    f"articulation={articulation.name!r} "
                    f"type={articulation.articulation_type} parent={parent!r} "
                    f"child={child!r} rest_gap={rest_gap:.4g}m max_gap={worst_gap:.4g}m "
                    f"at value={worst_value:.4g} "
                    f"(child pulls {worst_gap - rest_gap:.4g}m off the parent as it moves)"
                )
        if not findings:
            return self._record(check_name, True)
        details = (
            "A hinge/pivot pulls its moving child away from its parent -- the parts "
            "touch at rest but separate when the joint is actuated (e.g. a lid that "
            "floats off when opened). Place the articulation origin and axis so the "
            "child stays connected to the parent throughout its motion range:\n"
            + "\n".join(findings)
        )
        return self._record(check_name, False, details, kind=FailureKind.ARTICULATION_SEPARATION)

    def warn_if_absurd_dimensions(
        self,
        *,
        max_dimension: float = 1000.0,
        outlier_ratio: float = 100.0,
        name: str | None = None,
    ) -> bool:
        maximum = _non_negative(max_dimension, "max_dimension")
        ratio = _non_negative(outlier_ratio, "outlier_ratio")
        check_name = name or "warn_if_absurd_dimensions()"
        spans: list[tuple[str, str, float]] = []
        for part in self.model.parts:
            for shape in part._iter_shapes():
                bounds = self.shape_world_bounds(part, shape.name)
                span = max(bounds[1][axis] - bounds[0][axis] for axis in range(3))
                spans.append((part.name, shape.name, span))
        positive = [span for _part, _shape, span in spans if span > 0.0]
        median = statistics.median(positive) if positive else 0.0
        warnings: list[str] = []
        for part_name, shape_name, span in spans:
            if span > maximum:
                warnings.append(f"absurd dimension: {part_name!r}/{shape_name!r} spans {span:.4g}m")
            elif median > 0.0 and ratio > 0.0 and span > median * ratio:
                warnings.append(
                    f"extreme scale outlier: {part_name!r}/{shape_name!r} spans {span:.4g}m "
                    f"while median shape span is {median:.4g}m"
                )
        if warnings:
            self.warn("Scale warning:\n" + "\n".join(warnings[:10]))
        return self._record(check_name, True)

    def fail_if_parts_overlap_in_current_pose(
        self,
        *,
        overlap_tol: float = DEFAULT_OVERLAP_TOLERANCE,
        overlap_volume_tol: float = DEFAULT_OVERLAP_VOLUME_TOLERANCE,
        name: str | None = None,
    ) -> bool:
        overlap_tol = _non_negative(overlap_tol, "overlap_tol")
        volume_tol = _non_negative(overlap_volume_tol, "overlap_volume_tol")
        check_name = name or (
            "fail_if_parts_overlap_in_current_pose()"
            if (
                overlap_tol == DEFAULT_OVERLAP_TOLERANCE
                and volume_tol == DEFAULT_OVERLAP_VOLUME_TOLERANCE
            )
            else (
                "fail_if_parts_overlap_in_current_pose("
                f"overlap_tol={overlap_tol:.4g},overlap_volume_tol={volume_tol:.4g})"
            )
        )
        overlaps = self._kernel().meaningful_overlaps(
            self._pose,
            overlap_tol=overlap_tol,
            overlap_volume_tol=volume_tol,
        )
        allowed: list[CollisionQuery] = []
        remaining: list[CollisionQuery] = []
        for query in overlaps:
            (allowed if self._overlap_is_allowed(query) else remaining).append(query)
        if allowed:
            self.warn(
                "Overlaps detected but allowed by justification: "
                f"{len(allowed)} named shape pair(s)."
            )
        if not remaining:
            return self._record(check_name, True)

        representative: dict[tuple[str, str], CollisionQuery] = {}
        for query in remaining:
            key = _pair_key(query.part_a, query.part_b)
            current = representative.get(key)
            if current is None or _overlap_rank(query) > _overlap_rank(current):
                representative[key] = query
        details = [_collision_details(representative[key]) for key in sorted(representative)]
        return self._record(
            check_name,
            False,
            "Part overlaps detected "
            f"(overlap_tol={overlap_tol:.4g}, overlap_volume_tol={volume_tol:.4g}):\n"
            + "\n".join(details[:10]),
            kind=FailureKind.OVERLAP,
        )

    def _check_disconnected_geometry(
        self,
        *,
        contact_tol: float,
        name: str | None,
        warn: bool,
    ) -> bool:
        contact_tol = _non_negative(contact_tol, "contact_tol")
        prefix = (
            "warn_if_part_contains_disconnected_geometry_islands"
            if warn
            else "fail_if_part_contains_disconnected_geometry_islands"
        )
        check_name = name or f"{prefix}(contact_tol={contact_tol:.6g})"
        findings = self._kernel().disconnected_geometry_islands(contact_tol=contact_tol)
        if not findings:
            return self._record(check_name, True)
        lines = [_geometry_connectivity_details(finding) for finding in findings[:10]]
        details = "Disconnected geometry islands detected:\n" + "\n".join(lines)
        if warn:
            self.warn(details)
            return self._record(check_name, True)
        return self._record(check_name, False, details, kind=FailureKind.DISCONNECTED_GEOMETRY)

    def _collision_query(
        self,
        part_a: PartRef,
        part_b: PartRef,
        *,
        shape_a: str | None,
        shape_b: str | None,
    ) -> CollisionQuery:
        return self._kernel().collision_between(
            _part_name(part_a, field_name="part_a"),
            _part_name(part_b, field_name="part_b"),
            self._pose,
            shape_a=shape_a,
            shape_b=shape_b,
        )

    def _distance_query(
        self,
        part_a: PartRef,
        part_b: PartRef,
        *,
        shape_a: str | None,
        shape_b: str | None,
    ) -> DistanceFinding:
        return self._kernel().distance_between(
            _part_name(part_a, field_name="part_a"),
            _part_name(part_b, field_name="part_b"),
            self._pose,
            shape_a=shape_a,
            shape_b=shape_b,
        )

    def _bounds(self, part_name: str, shape_name: str | None) -> Bounds:
        if shape_name is None:
            return self._kernel().part_world_bounds(part_name, self._pose)
        return self._kernel().shape_world_bounds(part_name, shape_name, self._pose)

    def _selected_world_meshes(
        self,
        part: PartRef | None,
        shape: str | None,
    ) -> list[trimesh.Trimesh]:
        if shape is not None and part is None:
            raise ValidationError("shape requires a part selector")
        selected_part = None if part is None else _part_name(part, field_name="part")
        transforms = self._kernel().world_transforms(self._pose)
        meshes: list[trimesh.Trimesh] = []
        for model_part in self.model.parts:
            if selected_part is not None and model_part.name != selected_part:
                continue
            transform = transforms.get(model_part.name)
            if transform is None:
                continue
            for entry in model_part._iter_shapes():
                if shape is not None and entry.name != shape:
                    continue
                mesh = geometry_to_trimesh(entry.geometry, self.mesh_tolerance).copy()
                mesh.apply_transform(transform)
                meshes.append(mesh)
        if selected_part is not None:
            self.model.get_part(selected_part)
            if shape is not None:
                self.model.get_part(selected_part).get_shape(shape)
        if not meshes:
            raise ValidationError("geometry selector did not match any shapes")
        return meshes

    def _selected_geometries(
        self,
        part: PartRef | None,
        shape: str | None,
    ) -> list[tuple[str, str, object]]:
        if shape is not None and part is None:
            raise ValidationError("shape requires a part selector")
        selected_part = None if part is None else _part_name(part, field_name="part")
        geometries: list[tuple[str, str, object]] = []
        for model_part in self.model.parts:
            if selected_part is not None and model_part.name != selected_part:
                continue
            geometries.extend(
                (model_part.name, entry.name, entry.geometry)
                for entry in model_part._iter_shapes()
                if shape is None or entry.name == shape
            )
        if selected_part is not None:
            self.model.get_part(selected_part)
            if shape is not None:
                self.model.get_part(selected_part).get_shape(shape)
        if not geometries:
            raise ValidationError("geometry selector did not match any shapes")
        return geometries

    def _selected_world_vertices(
        self,
        part: PartRef | None,
        shape: str | None,
    ) -> np.ndarray:
        return np.concatenate(
            [np.asarray(mesh.vertices) for mesh in self._selected_world_meshes(part, shape)],
            axis=0,
        )

    def _overlap_is_allowed(self, query: CollisionQuery) -> bool:
        if query.shape_a is None or query.shape_b is None:
            return False
        part_a, part_b, shape_a, shape_b = _canonical_selectors(
            query.part_a, query.part_b, query.shape_a, query.shape_b
        )
        for allowance in self._allowed_overlaps:
            if allowance.part_a != part_a or allowance.part_b != part_b:
                continue
            if allowance.shape_a == shape_a and allowance.shape_b == shape_b:
                return True
        return False

    def _mesh_issue_is_allowed(
        self,
        part_name: str,
        shape_name: str,
        issue: MeshHealthIssue,
    ) -> bool:
        return any(
            allowance.part == part_name
            and allowance.shape == shape_name
            and issue in allowance.issues
            for allowance in self._allowed_mesh_issues
        )

    def _record(
        self,
        name: str,
        ok: bool,
        details: str = "",
        *,
        kind: FailureKind = FailureKind.AUTHORED,
    ) -> bool:
        self.checks_run += 1
        self._checks.append(name)
        if not ok:
            self._failures.append(TestFailure(name=name, details=details, kind=kind))
        return ok

    def _kernel(self) -> MeshCollisionKernel:
        if self._kernel_cache is None:
            self._kernel_cache = MeshCollisionKernel(self.model, mesh_tolerance=self.mesh_tolerance)
        return self._kernel_cache


def _part_name(value: PartRef, *, field_name: str) -> str:
    raw = value.name if isinstance(value, Part) else value
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError(f"{field_name} must be a part name or Part")
    return raw.strip()


def _plain_name(value: object, field_name: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise ValidationError(f"{field_name} must be non-empty")
    return name


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in {".txt", ".md"}:
        return "text"
    raise ValidationError(
        "artifact type could not be inferred; use PNG, JPEG, WebP, JSON, CSV, or text"
    )


def _vec3(value: Sequence[float], field_name: str) -> Vec3:
    if isinstance(value, str | bytes):
        raise ValidationError(f"{field_name} must contain 3 numeric values")
    try:
        values = tuple(_finite(item, field_name) for item in value)
    except TypeError as exc:
        raise ValidationError(f"{field_name} must contain 3 numeric values") from exc
    if len(values) != 3:
        raise ValidationError(f"{field_name} must contain 3 numeric values")
    return cast(Vec3, values)


def _geometry_selector(part: PartRef | None, shape: str | None) -> str:
    if part is None:
        return "model"
    part_name = _part_name(part, field_name="part")
    return part_name if shape is None else f"{part_name}/{shape}"


def _signed_mesh_volume(mesh: trimesh.Trimesh) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    return float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        ).sum()
        / 6.0
    )


def _pose_dicts(
    poses: Sequence[PoseSample | Mapping[object, float]],
) -> list[dict[object, float]]:
    if not poses:
        raise ValidationError("poses must contain at least one sample")
    result: list[dict[object, float]] = []
    for pose in poses:
        values = cast(
            Mapping[object, float],
            pose.as_dict() if isinstance(pose, PoseSample) else pose,
        )
        result.append(dict(values))
    return result


def _articulation_sweep_values(articulation: Articulation, samples: int) -> list[float]:
    """Joint values to sample across an articulation's motion range.

    The first value is the rest pose. A bounded joint sweeps lower..upper; a
    continuous or unbounded joint samples a half turn (0..pi), which is enough to
    reveal a child that separates as it rotates.
    """
    limits = articulation.motion_limits
    if limits is not None and limits.lower is not None and limits.upper is not None:
        low, high = float(limits.lower), float(limits.upper)
    else:
        low, high = 0.0, math.pi
    if high <= low:
        return [low]
    return [low + (high - low) * index / (samples - 1) for index in range(samples)]


def _articulation_name(value: object) -> str:
    raw = value.name if isinstance(value, Articulation) else value
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError("pose keys must be articulation names or Articulation objects")
    return raw.strip()


def _canonical_selectors(
    part_a: str,
    part_b: str,
    shape_a: str,
    shape_b: str,
) -> tuple[str, str, str, str]:
    if part_a <= part_b:
        return part_a, part_b, shape_a, shape_b
    return part_b, part_a, shape_b, shape_a


def _axis_name(axis: str) -> str:
    value = str(axis).strip().lower()
    if value not in {"x", "y", "z"}:
        raise ValidationError("axis must be one of: x, y, z")
    return value


def _axis_names(axes: str | Sequence[str]) -> tuple[str, ...]:
    raw = list(axes.strip().lower()) if isinstance(axes, str) else list(axes)
    normalized = tuple(dict.fromkeys(_axis_name(str(axis)) for axis in raw))
    if not normalized:
        raise ValidationError("axes must include at least one axis")
    return normalized


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _finite(value: object, field_name: str) -> float:
    try:
        result = float(cast(str, value))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{field_name} must be finite")
    return result


def _non_negative(value: object, field_name: str) -> float:
    result = _finite(value, field_name)
    if result < 0.0:
        raise ValidationError(f"{field_name} must be non-negative")
    return result


def _collision_details(query: CollisionQuery) -> str:
    contact = query.contacts[0] if query.contacts else None
    overlap = query.overlap_depth or (0.0, 0.0, 0.0)
    return (
        f"pair=({query.part_a!r},{query.part_b!r}) "
        f"shape_a={query.shape_a!r} shape_b={query.shape_b!r} collided={query.collided} "
        f"contacts={len(query.contacts)} max_depth={_format_optional(query.max_depth)} "
        f"overlap_depth=({overlap[0]:.4g},{overlap[1]:.4g},{overlap[2]:.4g}) "
        f"overlap_volume={query.overlap_volume:.4g} "
        f"normal={None if contact is None else contact.normal} "
        f"position={None if contact is None else contact.position}"
    )


def _distance_details(query: DistanceFinding) -> str:
    return (
        f"pair=({query.part_a!r},{query.part_b!r}) "
        f"shape_a={query.shape_a!r} shape_b={query.shape_b!r} "
        f"distance={query.distance:.6g} collided={query.collided} "
        f"nearest_a={query.nearest_a} nearest_b={query.nearest_b}"
    )


def _geometry_connectivity_details(finding: GeometryConnectivityFinding) -> str:
    return (
        f"part={finding.part!r} connected={finding.connected}/{finding.total} "
        f"contact_tol={finding.contact_tol:.6g} "
        f"disconnected=[{', '.join(finding.disconnected)}]"
    )


def _check_name(prefix: str, query: CollisionQuery | DistanceFinding) -> str:
    selectors = ""
    if query.shape_a is not None or query.shape_b is not None:
        selectors = f",shape_a={query.shape_a},shape_b={query.shape_b}"
    return f"{prefix}({query.part_a},{query.part_b}{selectors})"


def _selector_text(**selectors: str | None) -> str:
    values = [f"{name}={value}" for name, value in selectors.items() if value is not None]
    return "" if not values else "," + ",".join(values)


def _format_optional(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.6g}"


def _root_part_names(model: ArticulatedObject) -> list[str]:
    part_names = {part.name for part in model.parts}
    child_names = {item.child for item in model.articulations}
    return sorted(part_names - child_names)


def _reachable(roots: list[str], adjacency: dict[str, set[str]]) -> set[str]:
    found: set[str] = set()
    stack = list(roots)
    while stack:
        part_name = stack.pop()
        if part_name in found:
            continue
        found.add(part_name)
        stack.extend(sorted(adjacency.get(part_name, set()) - found))
    return found


def _groups(names: set[str], adjacency: dict[str, set[str]]) -> list[set[str]]:
    remaining = set(names)
    groups: list[set[str]] = []
    while remaining:
        start = min(remaining)
        group = _reachable([start], adjacency) & names
        groups.append(group)
        remaining -= group
    return groups


def _overlap_rank(query: CollisionQuery) -> tuple[float, float]:
    depth = query.overlap_depth or (0.0, 0.0, 0.0)
    return min(depth), query.overlap_volume


__all__ = [
    "AllowedMeshIssues",
    "AllowedOverlap",
    "DistanceFinding",
    "FailureKind",
    "GeometryMetrics",
    "PoseSample",
    "TestArtifact",
    "TestContext",
    "TestFailure",
    "TestMetric",
    "TestReport",
]
