from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal, cast

import numpy as np
import trimesh

from articraft.sdk._mesh.core import LoftGeometry, MeshGeometry
from articraft.sdk._mesh.sweeps import (
    _initial_frame,
    _path_frames,
    _validated_up_hint,
)
from articraft.sdk.errors import ValidationError

Vec3 = tuple[float, float, float]
RepairMode = Literal["off", "mesh"]
LoftInterpolation = Literal["linear", "catmull_rom"]
LoftParameterization = Literal["uniform", "chord", "centripetal"]
LoftCapStyle = Literal["flat", "round"]
LoftFrameMode = Literal["parallel_transport", "fixed_up"]


def _as_vec3(value: Sequence[float], *, name: str) -> Vec3:
    if len(value) != 3:
        raise ValidationError(f"{name} must have 3 values")
    try:
        point = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must have 3 numeric values") from exc
    if not np.isfinite(point).all():
        raise ValidationError(f"{name} values must be finite")
    return point


def _normalize_loop(points: Sequence[Sequence[float]], *, name: str) -> tuple[Vec3, ...]:
    result: list[Vec3] = []
    for point in points:
        value = _as_vec3(point, name=f"{name}[]")
        if not result or value != result[-1]:
            result.append(value)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    if len(result) < 3:
        raise ValidationError(f"{name} must contain at least 3 distinct points")
    return tuple(result)


@dataclass(frozen=True)
class LoftSection:
    points: tuple[Vec3, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _normalize_loop(self.points, name="section.points"))


@dataclass(frozen=True)
class SectionLoftSpec:
    sections: tuple[LoftSection, ...]
    path: tuple[Vec3, ...] | None = None
    cap: bool = True
    symmetry: Literal["mirror_yz"] | None = None
    repair: RepairMode = "off"
    interpolation: LoftInterpolation = "linear"
    samples_per_span: int = 1
    close_path: bool = False
    align_sections: bool = True
    parameterization: LoftParameterization = "uniform"
    tension: float = 0.0
    cap_style: LoftCapStyle = "flat"
    cap_segments: int = 6
    cap_length: float | None = None
    orient_to_path: bool = False
    frame_mode: LoftFrameMode = "parallel_transport"
    up_hint: Vec3 = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        sections = tuple(
            section if isinstance(section, LoftSection) else LoftSection(tuple(section))
            for section in self.sections
        )
        if len(sections) < 2:
            raise ValidationError("SectionLoftSpec requires at least two sections")
        object.__setattr__(self, "sections", sections)
        if self.path is not None:
            path = tuple(_as_vec3(point, name="path[]") for point in self.path)
            if len(path) < 2:
                raise ValidationError("path must contain at least two points")
            if any(
                float(np.linalg.norm(np.subtract(end, start))) <= 1e-12
                for start, end in pairwise(path)
            ):
                raise ValidationError("consecutive path points must be distinct")
            object.__setattr__(self, "path", path)
        if self.symmetry not in {None, "mirror_yz"}:
            raise ValidationError("symmetry must be 'mirror_yz' or None")
        if self.repair not in {"off", "mesh"}:
            raise ValidationError("repair must be 'off' or 'mesh'")
        if self.interpolation not in {"linear", "catmull_rom"}:
            raise ValidationError("interpolation must be 'linear' or 'catmull_rom'")
        if self.parameterization not in {"uniform", "chord", "centripetal"}:
            raise ValidationError("parameterization must be 'uniform', 'chord', or 'centripetal'")
        tension = float(self.tension)
        if not np.isfinite(tension) or not 0.0 <= tension <= 1.0:
            raise ValidationError("tension must be finite and between 0 and 1")
        object.__setattr__(self, "tension", tension)
        samples = int(self.samples_per_span)
        if samples < 1:
            raise ValidationError("samples_per_span must be at least 1")
        object.__setattr__(self, "samples_per_span", samples)
        if self.close_path and len(sections) < 3:
            raise ValidationError("close_path requires at least three sections")
        if self.cap_style not in {"flat", "round"}:
            raise ValidationError("cap_style must be 'flat' or 'round'")
        segments = int(self.cap_segments)
        if segments < 2:
            raise ValidationError("cap_segments must be at least 2")
        object.__setattr__(self, "cap_segments", segments)
        if self.cap_length is not None:
            length = float(self.cap_length)
            if not np.isfinite(length) or length <= 0.0:
                raise ValidationError("cap_length must be finite and positive")
            object.__setattr__(self, "cap_length", length)
        if self.frame_mode not in {"parallel_transport", "fixed_up"}:
            raise ValidationError("frame_mode must be 'parallel_transport' or 'fixed_up'")
        try:
            hint = _validated_up_hint(self.up_hint)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        object.__setattr__(self, "up_hint", hint)
        if self.orient_to_path and self.path is None:
            raise ValidationError("orient_to_path requires a path")
        if self.orient_to_path and self.close_path:
            raise ValidationError("orient_to_path does not support close_path")


def _coerce_spec(
    value: SectionLoftSpec | Sequence[LoftSection | Sequence[Sequence[float]]],
) -> SectionLoftSpec:
    if isinstance(value, SectionLoftSpec):
        return value
    return SectionLoftSpec(
        sections=tuple(
            section
            if isinstance(section, LoftSection)
            else LoftSection(cast(tuple[Vec3, ...], tuple(section)))
            for section in value
        )
    )


def _resample_loop(points: Sequence[Vec3], count: int) -> list[Vec3]:
    if len(points) == count:
        return list(points)
    closed = [*points, points[0]]
    lengths = [0.0]
    for start, end in pairwise(closed):
        lengths.append(lengths[-1] + float(np.linalg.norm(np.subtract(end, start))))
    total = lengths[-1]
    if total <= 1e-12:
        raise ValidationError("loft section perimeter must be non-zero")
    result: list[Vec3] = []
    edge = 0
    for index in range(count):
        target = total * index / count
        while edge + 1 < len(lengths) and lengths[edge + 1] < target:
            edge += 1
        span = lengths[edge + 1] - lengths[edge]
        amount = 0.0 if span <= 1e-12 else (target - lengths[edge]) / span
        start, end = closed[edge], closed[edge + 1]
        result.append(
            (
                start[0] + (end[0] - start[0]) * amount,
                start[1] + (end[1] - start[1]) * amount,
                start[2] + (end[2] - start[2]) * amount,
            )
        )
    return result


def _loop_normal(points: Sequence[Vec3]) -> np.ndarray:
    center = np.mean(np.asarray(points, dtype=np.float64), axis=0)
    normal = np.zeros(3, dtype=np.float64)
    for start, end in pairwise([*points, points[0]]):
        normal += np.cross(np.subtract(start, center), np.subtract(end, center))
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        raise ValidationError("loft section must enclose a non-zero planar area")
    return normal / length


def _align_loop(reference: Sequence[Vec3], candidate: Sequence[Vec3]) -> list[Vec3]:
    values = list(candidate)
    if float(np.dot(_loop_normal(reference), _loop_normal(values))) < 0.0:
        values.reverse()
    reference_centered = np.asarray(reference, dtype=np.float64) - np.mean(reference, axis=0)
    candidate_centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0)
    shift = min(
        range(len(values)),
        key=lambda index: float(
            np.sum((reference_centered - np.roll(candidate_centered, -index, axis=0)) ** 2)
        ),
    )
    return values[shift:] + values[:shift]


def _sample_polyline(points: Sequence[Vec3], amount: float) -> Vec3:
    lengths = [0.0]
    for start, end in pairwise(points):
        lengths.append(lengths[-1] + float(np.linalg.norm(np.subtract(end, start))))
    target = lengths[-1] * amount
    index = 0
    while index + 1 < len(lengths) and lengths[index + 1] < target:
        index += 1
    span = lengths[index + 1] - lengths[index]
    local = 0.0 if span <= 1e-12 else (target - lengths[index]) / span
    start, end = points[index], points[index + 1]
    return (
        start[0] + (end[0] - start[0]) * local,
        start[1] + (end[1] - start[1]) * local,
        start[2] + (end[2] - start[2]) * local,
    )


def _sample_polyline_tangent(points: Sequence[Vec3], amount: float) -> Vec3:
    delta = 1e-5
    before = np.asarray(_sample_polyline(points, max(0.0, amount - delta)))
    after = np.asarray(_sample_polyline(points, min(1.0, amount + delta)))
    direction = after - before
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValidationError("path tangent must be non-zero")
    tangent = direction / length
    return (float(tangent[0]), float(tangent[1]), float(tangent[2]))


def _path_adjusted_sections(spec: SectionLoftSpec, count: int) -> list[list[Vec3]]:
    sections = [_resample_loop(section.points, count) for section in spec.sections]
    if spec.align_sections:
        aligned_sections = [sections[0]]
        for section in sections[1:]:
            aligned_sections.append(_align_loop(aligned_sections[-1], section))
        sections = aligned_sections
    path = spec.path
    if path is None:
        return sections
    start, end = path[0], path[-1]
    adjusted: list[list[Vec3]] = []
    amounts: list[float] = []
    divisor = max(1, len(sections) - 1)
    for index, section in enumerate(sections):
        amount = index / divisor
        amounts.append(amount)
        path_point = _sample_polyline(path, amount)
        baseline = tuple(start[axis] + (end[axis] - start[axis]) * amount for axis in range(3))
        residual = tuple(path_point[axis] - baseline[axis] for axis in range(3))
        adjusted.append(
            [
                cast(Vec3, tuple(point[axis] + residual[axis] for axis in range(3)))
                for point in section
            ]
        )
    if not spec.orient_to_path:
        return adjusted

    centers = [np.mean(np.asarray(section, dtype=np.float64), axis=0) for section in sections]
    baseline = centers[-1] - centers[0]
    baseline_length = float(np.linalg.norm(baseline))
    if baseline_length <= 1e-12:
        raise ValidationError("orient_to_path requires distinct first and last section centers")
    baseline_tangent = cast(Vec3, tuple(float(value) for value in baseline / baseline_length))
    base_normal, base_binormal = _initial_frame(baseline_tangent, spec.up_hint)
    tangents = [_sample_polyline_tangent(path, amount) for amount in amounts]
    path_points = [_sample_polyline(path, amount) for amount in amounts]
    normals, binormals = _path_frames(
        path_points,
        tangents,
        closed=False,
        up_hint=spec.up_hint,
        frame_mode=spec.frame_mode,
    )
    oriented: list[list[Vec3]] = []
    for index, section in enumerate(sections):
        adjusted_center = np.mean(np.asarray(adjusted[index], dtype=np.float64), axis=0)
        ring: list[Vec3] = []
        for point in section:
            relative = np.asarray(point, dtype=np.float64) - centers[index]
            local = (
                float(np.dot(relative, base_normal)),
                float(np.dot(relative, base_binormal)),
                float(np.dot(relative, baseline_tangent)),
            )
            placed = (
                adjusted_center
                + np.asarray(normals[index]) * local[0]
                + np.asarray(binormals[index]) * local[1]
                + np.asarray(tangents[index]) * local[2]
            )
            ring.append(cast(Vec3, tuple(float(value) for value in placed)))
        oriented.append(ring)
    return oriented


def _repair_mesh(geometry: MeshGeometry) -> MeshGeometry:
    if not geometry.vertices or not geometry.faces:
        return geometry.copy()
    mesh = trimesh.Trimesh(
        vertices=np.asarray(geometry.vertices, dtype=np.float64),
        faces=np.asarray(geometry.faces, dtype=np.int64),
        process=False,
    )
    mesh.update_faces(mesh.unique_faces() & mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces() & mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=False)
    return MeshGeometry.from_trimesh(mesh)


def _mirror_yz_profiles(profiles: list[list[Vec3]]) -> list[list[Vec3]]:
    mirrored_profiles: list[list[Vec3]] = []
    for index, profile in enumerate(profiles):
        positive = [point for point in profile if point[0] > 1e-12]
        if len(positive) < 2 or any(point[0] < -1e-12 for point in profile):
            raise ValidationError(
                f"sections[{index}] must describe one positive-x half for mirror_yz symmetry"
            )
        mirrored = [(-point[0], point[1], point[2]) for point in reversed(positive)]
        mirrored_profiles.append([*positive, *mirrored])
    count = max(len(profile) for profile in mirrored_profiles)
    return [_resample_loop(profile, count) for profile in mirrored_profiles]


def section_loft(
    spec: SectionLoftSpec | Sequence[LoftSection | Sequence[Sequence[float]]],
    /,
    **overrides: object,
) -> MeshGeometry:
    value = _coerce_spec(spec)
    if overrides:
        value = replace(value, **overrides)
    count = max(len(section.points) for section in value.sections)
    profiles = _path_adjusted_sections(value, count)
    if value.symmetry == "mirror_yz":
        profiles = _mirror_yz_profiles(profiles)
    geometry = LoftGeometry(
        profiles,
        cap=value.cap,
        closed=True,
        interpolation=value.interpolation,
        samples_per_span=value.samples_per_span,
        close_path=value.close_path,
        parameterization=value.parameterization,
        tension=value.tension,
        cap_style=value.cap_style,
        cap_segments=value.cap_segments,
        cap_length=value.cap_length,
    )
    if value.repair == "mesh":
        geometry = _repair_mesh(geometry)
    return geometry


def repair_loft(
    geometry_or_spec: MeshGeometry
    | SectionLoftSpec
    | Sequence[LoftSection | Sequence[Sequence[float]]],
    /,
    *,
    repair: RepairMode = "mesh",
) -> MeshGeometry:
    if repair not in {"off", "mesh"}:
        raise ValidationError("repair must be 'off' or 'mesh'")
    if isinstance(geometry_or_spec, MeshGeometry):
        return geometry_or_spec.copy() if repair == "off" else _repair_mesh(geometry_or_spec)
    return section_loft(_coerce_spec(geometry_or_spec), repair=repair)


__all__ = [
    "LoftSection",
    "SectionLoftSpec",
    "repair_loft",
    "section_loft",
]
