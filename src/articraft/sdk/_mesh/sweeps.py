from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from articraft.sdk._mesh.core import (
    _EPS,
    MeshGeometry,
    SphereGeometry,
    Vec2,
    Vec3,
    _ensure_ccw,
    _profile_2d,
    _triangulate_simple,
    _v_add,
    _v_cross,
    _v_dot,
    _v_lerp,
    _v_norm,
    _v_normalize,
    _v_scale,
    _v_sub,
    _vec2,
)
from articraft.sdk._mesh.profiles import (
    sample_arc_3d,
    sample_catmull_rom_spline_3d,
    sample_cubic_bezier_spline_3d,
)


def _path_points(path: Iterable[Sequence[float]], *, minimum: int = 2) -> list[Vec3]:
    result: list[Vec3] = []
    for raw in path:
        if len(raw) != 3:
            raise ValueError("path points must have 3 values")
        point = (float(raw[0]), float(raw[1]), float(raw[2]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("path points must be finite")
        if not result or _v_norm(_v_sub(point, result[-1])) > _EPS:
            result.append(point)
    if len(result) < minimum:
        raise ValueError(f"path requires at least {minimum} distinct points")
    return result


def _validated_up_hint(value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError("up_hint must have 3 values")
    hint = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(component) for component in hint) or _v_norm(hint) <= _EPS:
        raise ValueError("up_hint must be finite and non-zero")
    return hint


@dataclass(frozen=True)
class SweepSection:
    """A profile control at one normalized position along a sweep path."""

    position: float
    profile: Sequence[Vec2] | None = None
    scale: float | Sequence[float] = 1.0
    rotation: float = 0.0
    offset: Sequence[float] = (0.0, 0.0)
    interpolation: str | None = None
    tension: float | None = None

    def __post_init__(self) -> None:
        position = float(self.position)
        if not math.isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError("sweep section position must be finite and between 0 and 1")
        object.__setattr__(self, "position", position)

        scale = (
            (float(self.scale), float(self.scale))
            if isinstance(self.scale, (int, float))
            else _vec2(self.scale, name="sweep section scale")
        )
        if not all(math.isfinite(value) and value > 0.0 for value in scale):
            raise ValueError("sweep section scale values must be finite and positive")
        object.__setattr__(self, "scale", scale)

        rotation = float(self.rotation)
        if not math.isfinite(rotation):
            raise ValueError("sweep section rotation must be finite")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "offset", _vec2(self.offset, name="sweep section offset"))

        if self.interpolation is not None:
            interpolation = str(self.interpolation).strip().lower().replace("-", "_")
            if interpolation not in {"linear", "catmull_rom"}:
                raise ValueError("sweep section interpolation must be 'linear' or 'catmull_rom'")
            object.__setattr__(self, "interpolation", interpolation)
        if self.tension is not None:
            tension = float(self.tension)
            if not math.isfinite(tension) or not 0.0 <= tension <= 1.0:
                raise ValueError("sweep section tension must be finite and between 0 and 1")
            object.__setattr__(self, "tension", tension)

        if self.profile is not None:
            object.__setattr__(
                self,
                "profile",
                tuple(_profile_2d(self.profile, minimum=2)),
            )


@dataclass(frozen=True)
class _SectionProfile:
    position: float
    points: list[Vec2]
    interpolation: str
    tension: float


def _section_interpolation(value: str) -> str:
    interpolation = str(value).strip().lower().replace("-", "_")
    if interpolation not in {"linear", "catmull_rom"}:
        raise ValueError("section_interpolation must be 'linear' or 'catmull_rom'")
    return interpolation


def _section_tension(value: float) -> float:
    tension = float(value)
    if not math.isfinite(tension) or not 0.0 <= tension <= 1.0:
        raise ValueError("section_tension must be finite and between 0 and 1")
    return tension


class SweepGeometry(MeshGeometry):
    def __init__(
        self,
        profile: Iterable[Sequence[float]],
        path: Iterable[Sequence[float]],
        *,
        cap: bool = False,
        closed: bool = True,
        path_closed: bool = False,
        up_hint: Vec3 = (0.0, 0.0, 1.0),
        frame_mode: str = "parallel_transport",
        sections: Sequence[SweepSection] = (),
        section_interpolation: str = "linear",
        section_tension: float = 0.0,
        max_segment_length: float | None = None,
        cap_style: str = "flat",
        cap_segments: int = 6,
        cap_length: float | None = None,
    ):
        sweep = PipeGeometry(
            profile,
            path,
            cap=cap,
            closed=closed,
            path_closed=path_closed,
            up_hint=up_hint,
            frame_mode=frame_mode,
            sections=sections,
            section_interpolation=section_interpolation,
            section_tension=section_tension,
            max_segment_length=max_segment_length,
            cap_style=cap_style,
            cap_segments=cap_segments,
            cap_length=cap_length,
        )
        super().__init__(sweep.vertices, sweep.faces)


def _tangents(path: Sequence[Vec3], *, closed: bool) -> list[Vec3]:
    result = []
    for index in range(len(path)):
        if closed:
            direction = _v_sub(path[(index + 1) % len(path)], path[index - 1])
        elif index == 0:
            direction = _v_sub(path[1], path[0])
        elif index == len(path) - 1:
            direction = _v_sub(path[-1], path[-2])
        else:
            direction = _v_sub(path[index + 1], path[index - 1])
        result.append(_v_normalize(direction))
    return result


def _initial_frame(tangent: Vec3, up_hint: Vec3) -> tuple[Vec3, Vec3]:
    up = _v_normalize(up_hint)
    if abs(_v_dot(up, tangent)) > 0.95:
        up = (0.0, 1.0, 0.0) if abs(tangent[1]) < 0.95 else (1.0, 0.0, 0.0)
    normal = _v_normalize(_v_cross(up, tangent))
    return normal, _v_normalize(_v_cross(tangent, normal))


def _rotate_vector(value: Vec3, axis: Vec3, angle: float) -> Vec3:
    cosine, sine = math.cos(angle), math.sin(angle)
    return _v_add(
        _v_add(_v_scale(value, cosine), _v_scale(_v_cross(axis, value), sine)),
        _v_scale(axis, _v_dot(axis, value) * (1.0 - cosine)),
    )


def _transport_frame(
    normal: Vec3,
    binormal: Vec3,
    previous_tangent: Vec3,
    tangent: Vec3,
) -> tuple[Vec3, Vec3]:
    axis = _v_cross(previous_tangent, tangent)
    sine = _v_norm(axis)
    cosine = max(-1.0, min(1.0, _v_dot(previous_tangent, tangent)))
    if sine > _EPS:
        next_normal = _rotate_vector(
            normal,
            _v_scale(axis, 1.0 / sine),
            math.atan2(sine, cosine),
        )
    elif cosine < 0.0:
        next_normal = _rotate_vector(normal, binormal, math.pi)
    else:
        next_normal = normal
    next_normal = _v_sub(next_normal, _v_scale(tangent, _v_dot(next_normal, tangent)))
    if _v_norm(next_normal) <= _EPS:
        next_normal = _v_cross(binormal, tangent)
    next_normal = _v_normalize(next_normal)
    return next_normal, _v_normalize(_v_cross(tangent, next_normal))


def _signed_angle(start: Vec3, end: Vec3, axis: Vec3) -> float:
    return math.atan2(_v_dot(axis, _v_cross(start, end)), _v_dot(start, end))


def _path_fractions(path: Sequence[Vec3], *, closed: bool) -> tuple[list[float], float]:
    lengths = [0.0]
    for index in range(1, len(path)):
        lengths.append(lengths[-1] + _v_norm(_v_sub(path[index], path[index - 1])))
    total = lengths[-1]
    if closed:
        total += _v_norm(_v_sub(path[0], path[-1]))
    if total <= _EPS:
        raise ValueError("sweep path length must be positive")
    return [length / total for length in lengths], total


def _path_point_at_distance(
    path: Sequence[Vec3],
    distance: float,
    *,
    closed: bool,
) -> Vec3:
    segment_count = len(path) if closed else len(path) - 1
    traveled = 0.0
    for index in range(segment_count):
        start, end = path[index], path[(index + 1) % len(path)]
        length = _v_norm(_v_sub(end, start))
        if distance <= traveled + length + _EPS or index == segment_count - 1:
            amount = 0.0 if length <= _EPS else (distance - traveled) / length
            return _v_lerp(start, end, max(0.0, min(1.0, amount)))
        traveled += length
    return path[0] if closed else path[-1]


def _insert_path_fractions(
    path: Sequence[Vec3],
    targets: Iterable[float],
    *,
    closed: bool,
) -> list[Vec3]:
    fractions, total = _path_fractions(path, closed=closed)
    values = list(fractions)
    for target in targets:
        fraction = float(target)
        if closed and fraction >= 1.0 - _EPS:
            continue
        if not any(abs(existing - fraction) <= _EPS for existing in values):
            values.append(fraction)
    values.sort()
    return [_path_point_at_distance(path, fraction * total, closed=closed) for fraction in values]


def _subdivide_path_segments(
    path: Sequence[Vec3],
    *,
    closed: bool,
    max_segment_length: float | None,
) -> list[Vec3]:
    if max_segment_length is None:
        return list(path)
    maximum = float(max_segment_length)
    if maximum <= 0.0 or not math.isfinite(maximum):
        raise ValueError("max_segment_length must be finite and positive")
    segment_count = len(path) if closed else len(path) - 1
    result: list[Vec3] = []
    for index in range(segment_count):
        start, end = path[index], path[(index + 1) % len(path)]
        divisions = max(1, math.ceil(_v_norm(_v_sub(end, start)) / maximum))
        result.extend(_v_lerp(start, end, sample / divisions) for sample in range(divisions))
    if not closed:
        result.append(path[-1])
    return result


def _path_frames(
    path: Sequence[Vec3],
    tangents: Sequence[Vec3],
    *,
    closed: bool,
    up_hint: Vec3,
    frame_mode: str,
) -> tuple[list[Vec3], list[Vec3]]:
    if frame_mode not in {"parallel_transport", "fixed_up"}:
        raise ValueError("frame_mode must be 'parallel_transport' or 'fixed_up'")
    normal, binormal = _initial_frame(tangents[0], up_hint)
    normals = [normal]
    binormals = [binormal]
    for index, tangent in enumerate(tangents[1:], start=1):
        if frame_mode == "fixed_up" and _v_norm(_v_cross(up_hint, tangent)) > _EPS:
            normal = _v_normalize(_v_cross(up_hint, tangent))
            binormal = _v_normalize(_v_cross(tangent, normal))
        else:
            normal, binormal = _transport_frame(
                normals[-1],
                binormals[-1],
                tangents[index - 1],
                tangent,
            )
        normals.append(normal)
        binormals.append(binormal)

    if not closed or frame_mode == "fixed_up":
        return normals, binormals

    closing_normal, _ = _transport_frame(
        normals[-1],
        binormals[-1],
        tangents[-1],
        tangents[0],
    )
    correction = _signed_angle(closing_normal, normals[0], tangents[0])
    fractions, _ = _path_fractions(path, closed=True)
    for index in range(1, len(path)):
        angle = correction * fractions[index]
        normals[index] = _v_normalize(_rotate_vector(normals[index], tangents[index], angle))
        binormals[index] = _v_normalize(_v_cross(tangents[index], normals[index]))
    return normals, binormals


def _resample_profile(points: Sequence[Vec2], count: int, *, closed: bool) -> list[Vec2]:
    if len(points) == count:
        return list(points)
    edge_count = len(points) if closed else len(points) - 1
    lengths = [0.0]
    for index in range(edge_count):
        start, end = points[index], points[(index + 1) % len(points)]
        lengths.append(lengths[-1] + math.dist(start, end))
    total = lengths[-1]
    if total <= _EPS:
        raise ValueError("sweep profile perimeter must be positive")
    targets = (
        [total * index / count for index in range(count)]
        if closed
        else [total * index / (count - 1) for index in range(count)]
    )
    result: list[Vec2] = []
    edge = 0
    for target in targets:
        while edge + 1 < len(lengths) - 1 and lengths[edge + 1] < target:
            edge += 1
        span = lengths[edge + 1] - lengths[edge]
        amount = 0.0 if span <= _EPS else (target - lengths[edge]) / span
        start, end = points[edge], points[(edge + 1) % len(points)]
        result.append(
            (
                start[0] + (end[0] - start[0]) * amount,
                start[1] + (end[1] - start[1]) * amount,
            )
        )
    return result


def _profile_distance(a: Sequence[Vec2], b: Sequence[Vec2]) -> float:
    return sum(
        (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2
        for left, right in zip(a, b, strict=True)
    )


def _align_profile(
    reference: Sequence[Vec2], candidate: Sequence[Vec2], *, closed: bool
) -> list[Vec2]:
    values = list(candidate)
    if not closed:
        return (
            values
            if _profile_distance(reference, values)
            <= _profile_distance(reference, list(reversed(values)))
            else list(reversed(values))
        )
    shift = min(
        range(len(values)),
        key=lambda index: _profile_distance(reference, values[index:] + values[:index]),
    )
    return values[shift:] + values[:shift]


def _transform_profile(points: Sequence[Vec2], section: SweepSection) -> list[Vec2]:
    scale_x, scale_y = cast(Vec2, section.scale)
    cosine, sine = math.cos(section.rotation), math.sin(section.rotation)
    return [
        (
            x * scale_x * cosine - y * scale_y * sine + section.offset[0],
            x * scale_x * sine + y * scale_y * cosine + section.offset[1],
        )
        for x, y in points
    ]


def _sweep_section_profiles(
    base_profile: Sequence[Vec2],
    sections: Sequence[SweepSection],
    *,
    closed: bool,
    path_closed: bool,
    interpolation: str,
    tension: float,
) -> list[_SectionProfile]:
    values = list(sections)
    if any(not isinstance(section, SweepSection) for section in values):
        raise TypeError("sections must contain SweepSection values")
    values.sort(key=lambda section: section.position)
    if any(
        abs(values[index].position - values[index - 1].position) <= _EPS
        for index in range(1, len(values))
    ):
        raise ValueError("sweep section positions must be unique")
    if not values or values[0].position > _EPS:
        values.insert(0, SweepSection(0.0))
    has_final = values[-1].position >= 1.0 - _EPS
    if not has_final:
        values.append(SweepSection(1.0))

    raw_profiles = [
        list(base_profile)
        if section.profile is None
        else (_ensure_ccw(list(section.profile)) if closed else list(section.profile))
        for section in values
    ]
    count = max(len(base_profile), *(len(profile) for profile in raw_profiles))
    aligned: list[list[Vec2]] = []
    for index, profile in enumerate(raw_profiles):
        sampled = _resample_profile(profile, count, closed=closed)
        if aligned:
            reference = (
                aligned[0] if path_closed and values[index].position >= 1.0 - _EPS else aligned[-1]
            )
            sampled = _align_profile(reference, sampled, closed=closed)
        aligned.append(sampled)
    profiles = [
        _SectionProfile(
            section.position,
            _transform_profile(profile, section),
            section.interpolation or interpolation,
            tension if section.tension is None else section.tension,
        )
        for section, profile in zip(values, aligned, strict=True)
    ]
    if path_closed and not has_final:
        profiles[-1] = _SectionProfile(
            1.0,
            list(profiles[0].points),
            profiles[0].interpolation,
            profiles[0].tension,
        )
    if path_closed and _profile_distance(profiles[0].points, profiles[-1].points) > 1e-12:
        raise ValueError("a closed sweep path needs matching section profiles at positions 0 and 1")
    return profiles


def _profile_derivative(
    previous: _SectionProfile,
    following: _SectionProfile,
    *,
    previous_position: float,
    following_position: float,
    tension: float,
) -> list[Vec2]:
    span = following_position - previous_position
    if span <= _EPS:
        return [(0.0, 0.0) for _ in previous.points]
    scale = (1.0 - tension) / span
    return [
        ((end[0] - start[0]) * scale, (end[1] - start[1]) * scale)
        for start, end in zip(previous.points, following.points, strict=True)
    ]


def _smooth_profile_span(
    profiles: Sequence[_SectionProfile],
    start_index: int,
    end_index: int,
    amount: float,
    *,
    path_closed: bool,
) -> list[Vec2]:
    start, end = profiles[start_index], profiles[end_index]
    if start_index > 0:
        previous = profiles[start_index - 1]
        previous_position = previous.position
    elif path_closed:
        previous = profiles[-2]
        previous_position = previous.position - 1.0
    else:
        previous = start
        previous_position = start.position - (end.position - start.position)

    if end_index + 1 < len(profiles):
        following = profiles[end_index + 1]
        following_position = following.position
    elif path_closed:
        following = profiles[1]
        following_position = following.position + 1.0
    else:
        following = end
        following_position = end.position + (end.position - start.position)

    start_derivative = _profile_derivative(
        previous,
        end,
        previous_position=previous_position,
        following_position=end.position,
        tension=start.tension,
    )
    end_derivative = _profile_derivative(
        start,
        following,
        previous_position=start.position,
        following_position=following_position,
        tension=start.tension,
    )
    amount2, amount3 = amount * amount, amount * amount * amount
    h00 = 2.0 * amount3 - 3.0 * amount2 + 1.0
    h10 = amount3 - 2.0 * amount2 + amount
    h01 = -2.0 * amount3 + 3.0 * amount2
    h11 = amount3 - amount2
    span = end.position - start.position
    return [
        (
            h00 * first[0]
            + h10 * span * first_derivative[0]
            + h01 * second[0]
            + h11 * span * second_derivative[0],
            h00 * first[1]
            + h10 * span * first_derivative[1]
            + h01 * second[1]
            + h11 * span * second_derivative[1],
        )
        for first, second, first_derivative, second_derivative in zip(
            start.points,
            end.points,
            start_derivative,
            end_derivative,
            strict=True,
        )
    ]


def _profile_at(
    profiles: Sequence[_SectionProfile],
    position: float,
    *,
    path_closed: bool,
) -> list[Vec2]:
    if position <= profiles[0].position:
        return list(profiles[0].points)
    for index in range(1, len(profiles)):
        end_position, end_profile = profiles[index].position, profiles[index].points
        if position <= end_position + _EPS:
            start = profiles[index - 1]
            start_position, start_profile = start.position, start.points
            span = end_position - start_position
            amount = 0.0 if span <= _EPS else (position - start_position) / span
            if start.interpolation == "catmull_rom":
                return _smooth_profile_span(
                    profiles,
                    index - 1,
                    index,
                    amount,
                    path_closed=path_closed,
                )
            return [
                (
                    start[0] + (end[0] - start[0]) * amount,
                    start[1] + (end[1] - start[1]) * amount,
                )
                for start, end in zip(start_profile, end_profile, strict=True)
            ]
    return list(profiles[-1].points)


def _connect_profile_rings(
    faces: list[tuple[int, int, int]],
    first_offset: int,
    second_offset: int,
    count: int,
) -> None:
    for index in range(count):
        following = (index + 1) % count
        faces.extend(
            (
                (first_offset + index, first_offset + following, second_offset + following),
                (first_offset + index, second_offset + following, second_offset + index),
            )
        )


def _rounded_cap_length(profile: Sequence[Vec2], requested: float | None) -> float:
    if requested is not None:
        length = float(requested)
        if length <= 0.0 or not math.isfinite(length):
            raise ValueError("cap_length must be finite and positive")
        return length
    center = (
        sum(point[0] for point in profile) / len(profile),
        sum(point[1] for point in profile) / len(profile),
    )
    length = max(math.dist(point, center) for point in profile)
    if length <= _EPS:
        raise ValueError("round cap profile radius must be positive")
    return length


def _add_rounded_cap(
    vertices: list[Vec3],
    faces: list[tuple[int, int, int]],
    *,
    profile: Sequence[Vec2],
    path_point: Vec3,
    tangent: Vec3,
    normal: Vec3,
    binormal: Vec3,
    base_offset: int,
    start: bool,
    segments: int,
    length: float | None,
) -> None:
    center = (
        sum(point[0] for point in profile) / len(profile),
        sum(point[1] for point in profile) / len(profile),
    )
    cap_length = _rounded_cap_length(profile, length)
    direction = _v_scale(tangent, -1.0 if start else 1.0)
    ring_offsets: list[int] = []
    for sample in range(1, segments):
        angle = math.pi * sample / (2.0 * segments)
        scale = math.cos(angle)
        ring_center = _v_add(path_point, _v_scale(direction, cap_length * math.sin(angle)))
        ring_offsets.append(len(vertices))
        vertices.extend(
            _v_add(
                ring_center,
                _v_add(
                    _v_scale(normal, center[0] + (point[0] - center[0]) * scale),
                    _v_scale(binormal, center[1] + (point[1] - center[1]) * scale),
                ),
            )
            for point in profile
        )

    ordered = [*reversed(ring_offsets), base_offset] if start else [base_offset, *ring_offsets]
    for first, second in pairwise(ordered):
        _connect_profile_rings(faces, first, second, len(profile))

    tip_center = _v_add(path_point, _v_scale(direction, cap_length))
    tip = _v_add(
        tip_center,
        _v_add(_v_scale(normal, center[0]), _v_scale(binormal, center[1])),
    )
    tip_index = len(vertices)
    vertices.append(tip)
    terminal_ring = ordered[0] if start else ordered[-1]
    for index in range(len(profile)):
        following = (index + 1) % len(profile)
        faces.append(
            (tip_index, terminal_ring + following, terminal_ring + index)
            if start
            else (terminal_ring + index, terminal_ring + following, tip_index)
        )


class PipeGeometry(MeshGeometry):
    """Sweep a profile along a path using a parallel-transport style frame."""

    def __init__(
        self,
        profile: Iterable[Sequence[float]],
        path: Iterable[Sequence[float]],
        *,
        cap: bool = False,
        closed: bool = True,
        path_closed: bool = False,
        up_hint: Vec3 = (0.0, 0.0, 1.0),
        frame_mode: str = "parallel_transport",
        sections: Sequence[SweepSection] = (),
        section_interpolation: str = "linear",
        section_tension: float = 0.0,
        max_segment_length: float | None = None,
        cap_style: str = "flat",
        cap_segments: int = 6,
        cap_length: float | None = None,
    ):
        base_profile = (
            _ensure_ccw(_profile_2d(profile)) if closed else _profile_2d(profile, minimum=2)
        )
        path_values = _path_points(path)
        up_hint = _validated_up_hint(up_hint)
        section_interpolation = _section_interpolation(section_interpolation)
        section_tension = _section_tension(section_tension)
        cap_style = str(cap_style).strip().lower()
        if cap_style not in {"flat", "round"}:
            raise ValueError("cap_style must be 'flat' or 'round'")
        if not isinstance(cap_segments, int) or isinstance(cap_segments, bool) or cap_segments < 2:
            raise ValueError("cap_segments must be an integer of at least 2")
        if cap_length is not None:
            cap_length = _rounded_cap_length(base_profile, cap_length)
        if path_closed:
            if _v_norm(_v_sub(path_values[0], path_values[-1])) <= _EPS:
                path_values.pop()
            if len(path_values) < 3:
                raise ValueError("a closed sweep path requires at least three distinct points")
            cap = False
        section_profiles = _sweep_section_profiles(
            base_profile,
            sections,
            closed=closed,
            path_closed=path_closed,
            interpolation=section_interpolation,
            tension=section_tension,
        )
        path_values = _insert_path_fractions(
            path_values,
            (section.position for section in section_profiles),
            closed=path_closed,
        )
        path_values = _subdivide_path_segments(
            path_values,
            closed=path_closed,
            max_segment_length=max_segment_length,
        )
        tangent_values = _tangents(path_values, closed=path_closed)
        normals, binormals = _path_frames(
            path_values,
            tangent_values,
            closed=path_closed,
            up_hint=up_hint,
            frame_mode=frame_mode,
        )
        fractions, _ = _path_fractions(path_values, closed=path_closed)
        rings: list[list[Vec3]] = []
        ring_profiles: list[list[Vec2]] = []
        for index, point in enumerate(path_values):
            profile_points = _profile_at(
                section_profiles,
                fractions[index],
                path_closed=path_closed,
            )
            ring_profiles.append(profile_points)
            rings.append(
                [
                    _v_add(
                        point,
                        _v_add(_v_scale(normals[index], x), _v_scale(binormals[index], y)),
                    )
                    for x, y in profile_points
                ]
            )
        count = len(ring_profiles[0])
        vertices = [point for ring in rings for point in ring]
        faces: list[tuple[int, int, int]] = []
        segment_count = count if closed else count - 1
        path_segment_count = len(rings) if path_closed else len(rings) - 1
        for ring_index in range(path_segment_count):
            start = ring_index * count
            following_start = ((ring_index + 1) % len(rings)) * count
            for index in range(segment_count):
                following = (index + 1) % count
                faces.extend(
                    (
                        (start + index, start + following, following_start + following),
                        (start + index, following_start + following, following_start + index),
                    )
                )
        if cap and closed and cap_style == "flat":
            start_triangles = _triangulate_simple(ring_profiles[0])
            end_triangles = _triangulate_simple(ring_profiles[-1])
            end_offset = (len(rings) - 1) * count
            faces.extend((c, b, a) for a, b, c in start_triangles)
            faces.extend(
                (end_offset + a, end_offset + b, end_offset + c) for a, b, c in end_triangles
            )
        elif cap and closed:
            _add_rounded_cap(
                vertices,
                faces,
                profile=ring_profiles[0],
                path_point=path_values[0],
                tangent=tangent_values[0],
                normal=normals[0],
                binormal=binormals[0],
                base_offset=0,
                start=True,
                segments=cap_segments,
                length=cap_length,
            )
            _add_rounded_cap(
                vertices,
                faces,
                profile=ring_profiles[-1],
                path_point=path_values[-1],
                tangent=tangent_values[-1],
                normal=normals[-1],
                binormal=binormals[-1],
                base_offset=(len(rings) - 1) * count,
                start=False,
                segments=cap_segments,
                length=cap_length,
            )
        super().__init__(vertices, faces)


class ArcPipeGeometry(MeshGeometry):
    def __init__(
        self,
        profile: Iterable[Sequence[float]],
        *,
        start_point: Vec3,
        center: Vec3,
        normal: Vec3,
        angle: float,
        arc_segments: int = 20,
        cap: bool = False,
        closed: bool = True,
        up_hint: Vec3 = (0.0, 0.0, 1.0),
        frame_mode: str = "parallel_transport",
        sections: Sequence[SweepSection] = (),
        section_interpolation: str = "linear",
        section_tension: float = 0.0,
        max_segment_length: float | None = None,
        cap_style: str = "flat",
        cap_segments: int = 6,
        cap_length: float | None = None,
    ):
        pipe = PipeGeometry(
            profile,
            sample_arc_3d(
                start_point=start_point,
                center=center,
                normal=normal,
                angle=angle,
                segments=arc_segments,
            ),
            cap=cap,
            closed=closed,
            up_hint=up_hint,
            frame_mode=frame_mode,
            sections=sections,
            section_interpolation=section_interpolation,
            section_tension=section_tension,
            max_segment_length=max_segment_length,
            cap_style=cap_style,
            cap_segments=cap_segments,
            cap_length=cap_length,
        )
        super().__init__(pipe.vertices, pipe.faces)


def _rounded_centerline(
    points: list[Vec3], *, mode: str, radius: float, segments: int, closed: bool
) -> list[Vec3]:
    if mode not in {"miter", "bevel", "fillet"}:
        raise ValueError("corner_mode must be 'miter', 'bevel', or 'fillet'")
    if radius <= _EPS or mode == "miter" or len(points) < 3:
        return points
    ring = points[:-1] if closed and _v_norm(_v_sub(points[0], points[-1])) <= _EPS else points
    output: list[Vec3] = []
    count = len(ring)
    indices = range(count) if closed else range(1, count - 1)
    if not closed:
        output.append(ring[0])
    for index in indices:
        previous = ring[(index - 1) % count]
        corner = ring[index]
        following = ring[(index + 1) % count]
        incoming_length = _v_norm(_v_sub(corner, previous))
        outgoing_length = _v_norm(_v_sub(following, corner))
        trim = min(float(radius), incoming_length * 0.45, outgoing_length * 0.45)
        if trim <= _EPS:
            output.append(corner)
            continue
        before = _v_lerp(corner, previous, trim / incoming_length)
        after = _v_lerp(corner, following, trim / outgoing_length)
        output.append(before)
        if mode == "bevel":
            output.append(after)
        else:
            for sample in range(1, max(2, int(segments)) + 1):
                amount = sample / max(2, int(segments))
                inverse = 1.0 - amount
                output.append(
                    (
                        inverse**2 * before[0]
                        + 2.0 * inverse * amount * corner[0]
                        + amount**2 * after[0],
                        inverse**2 * before[1]
                        + 2.0 * inverse * amount * corner[1]
                        + amount**2 * after[1],
                        inverse**2 * before[2]
                        + 2.0 * inverse * amount * corner[2]
                        + amount**2 * after[2],
                    )
                )
    if not closed:
        output.append(ring[-1])
    else:
        output.append(output[0])
    return output


class WirePolylineGeometry(MeshGeometry):
    def __init__(
        self,
        points: Iterable[Sequence[float]],
        *,
        radius: float,
        radial_segments: int = 16,
        closed_path: bool = False,
        cap_ends: bool = False,
        corner_mode: str = "fillet",
        corner_radius: float = 0.0,
        corner_segments: int = 8,
        up_hint: Vec3 = (0.0, 0.0, 1.0),
        frame_mode: str = "parallel_transport",
        sections: Sequence[SweepSection] = (),
        section_interpolation: str = "linear",
        section_tension: float = 0.0,
        max_segment_length: float | None = None,
        cap_style: str = "flat",
        cap_segments: int = 6,
        cap_length: float | None = None,
        min_segment_length: float = 1e-6,
    ):
        radius = float(radius)
        if radius <= 0.0 or not math.isfinite(radius):
            raise ValueError("radius must be positive")
        if radial_segments < 6:
            raise ValueError("radial_segments must be at least 6")
        corner_radius = float(corner_radius)
        min_segment_length = float(min_segment_length)
        if corner_radius < 0.0 or not math.isfinite(corner_radius):
            raise ValueError("corner_radius must be finite and non-negative")
        if corner_segments < 2:
            raise ValueError("corner_segments must be at least 2")
        if min_segment_length <= 0.0 or not math.isfinite(min_segment_length):
            raise ValueError("min_segment_length must be finite and positive")
        raw = _path_points(points)
        filtered = [raw[0]]
        for point in raw[1:]:
            if _v_norm(_v_sub(point, filtered[-1])) >= min_segment_length:
                filtered.append(point)
        if closed_path and _v_norm(_v_sub(filtered[0], filtered[-1])) > _EPS:
            filtered.append(filtered[0])
        centerline = _rounded_centerline(
            filtered,
            mode=corner_mode,
            radius=corner_radius,
            segments=corner_segments,
            closed=closed_path,
        )
        profile = [
            (
                radius * math.cos(2.0 * math.pi * index / radial_segments),
                radius * math.sin(2.0 * math.pi * index / radial_segments),
            )
            for index in range(radial_segments)
        ]
        pipe = PipeGeometry(
            profile,
            centerline,
            cap=cap_ends and not closed_path,
            path_closed=closed_path,
            up_hint=up_hint,
            frame_mode=frame_mode,
            sections=sections,
            section_interpolation=section_interpolation,
            section_tension=section_tension,
            max_segment_length=max_segment_length,
            cap_style=cap_style,
            cap_segments=cap_segments,
            cap_length=cap_length,
        )
        super().__init__(pipe.vertices, pipe.faces)


def wire_from_points(
    points: Iterable[Sequence[float]],
    *,
    radius: float,
    radial_segments: int = 16,
    closed_path: bool = False,
    cap_ends: bool = False,
    corner_mode: str = "fillet",
    corner_radius: float = 0.0,
    corner_segments: int = 8,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
    min_segment_length: float = 1e-6,
) -> MeshGeometry:
    return WirePolylineGeometry(
        points,
        radius=radius,
        radial_segments=radial_segments,
        closed_path=closed_path,
        cap_ends=cap_ends,
        corner_mode=corner_mode,
        corner_radius=corner_radius,
        corner_segments=corner_segments,
        up_hint=up_hint,
        frame_mode=frame_mode,
        sections=sections,
        section_interpolation=section_interpolation,
        section_tension=section_tension,
        max_segment_length=max_segment_length,
        cap_style=cap_style,
        cap_segments=cap_segments,
        cap_length=cap_length,
        min_segment_length=min_segment_length,
    )


def _sample_spline(
    points: Iterable[Vec3],
    *,
    spline: str,
    samples_per_segment: int,
    closed: bool,
    alpha: float,
) -> list[Vec3]:
    values = list(points)
    key = spline.strip().lower().replace("-", "_")
    if key in {"catmullrom", "catmull_rom"}:
        return sample_catmull_rom_spline_3d(
            values,
            samples_per_segment=samples_per_segment,
            closed=closed,
            alpha=alpha,
        )
    if key in {"bezier", "cubic_bezier", "cubicbezier"}:
        sampled = sample_cubic_bezier_spline_3d(values, samples_per_segment=samples_per_segment)
        if closed and _v_norm(_v_sub(sampled[0], sampled[-1])) > _EPS:
            raise ValueError("a closed Bezier spline must end where it starts")
        return sampled
    raise ValueError("spline must be 'catmull_rom' or 'bezier'")


def tube_from_spline_points(
    points: Iterable[Vec3],
    *,
    radius: float,
    samples_per_segment: int = 12,
    closed_spline: bool = False,
    spline: str = "catmull_rom",
    alpha: float = 0.5,
    radial_segments: int = 16,
    cap_ends: bool = True,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
    min_segment_length: float = 1e-6,
) -> MeshGeometry:
    return wire_from_points(
        _sample_spline(
            points,
            spline=spline,
            samples_per_segment=samples_per_segment,
            closed=closed_spline,
            alpha=alpha,
        ),
        radius=radius,
        radial_segments=radial_segments,
        closed_path=closed_spline,
        cap_ends=cap_ends,
        corner_mode="miter",
        up_hint=up_hint,
        frame_mode=frame_mode,
        sections=sections,
        section_interpolation=section_interpolation,
        section_tension=section_tension,
        max_segment_length=max_segment_length,
        cap_style=cap_style,
        cap_segments=cap_segments,
        cap_length=cap_length,
        min_segment_length=min_segment_length,
    )


def sweep_profile_along_spline(
    points: Iterable[Vec3],
    *,
    profile: Iterable[Vec2],
    samples_per_segment: int = 12,
    closed_spline: bool = False,
    spline: str = "catmull_rom",
    alpha: float = 0.5,
    cap_profile: bool = True,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
    min_segment_length: float = 1e-6,
) -> MeshGeometry:
    min_segment_length = float(min_segment_length)
    if min_segment_length <= 0.0 or not math.isfinite(min_segment_length):
        raise ValueError("min_segment_length must be finite and positive")
    centerline = _sample_spline(
        points,
        spline=spline,
        samples_per_segment=samples_per_segment,
        closed=closed_spline,
        alpha=alpha,
    )
    centerline = _path_points(centerline)
    if any(
        _v_norm(_v_sub(centerline[index], centerline[index - 1])) < min_segment_length
        for index in range(1, len(centerline))
    ):
        raise ValueError("spline contains segments shorter than min_segment_length")
    return PipeGeometry(
        profile,
        centerline,
        cap=cap_profile and not closed_spline,
        path_closed=closed_spline,
        up_hint=up_hint,
        frame_mode=frame_mode,
        sections=sections,
        section_interpolation=section_interpolation,
        section_tension=section_tension,
        max_segment_length=max_segment_length,
        cap_style=cap_style,
        cap_segments=cap_segments,
        cap_length=cap_length,
    )


def tube_network_from_paths(
    paths: Iterable[Iterable[Vec3]],
    *,
    radius: float,
    radial_segments: int = 16,
    cap_ends: bool = True,
    corner_mode: str = "fillet",
    corner_radius: float = 0.0,
    corner_segments: int = 8,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
    min_segment_length: float = 1e-6,
    shared_node_radius: float | None = None,
) -> MeshGeometry:
    path_values = [_path_points(path) for path in paths]
    if not path_values:
        raise ValueError("tube network requires at least one path")
    radius = float(radius)
    node_radius = radius if shared_node_radius is None else float(shared_node_radius)
    if (
        radius <= 0.0
        or node_radius < 0.0
        or not math.isfinite(radius)
        or not math.isfinite(node_radius)
    ):
        raise ValueError(
            "tube and node radii must be finite and non-negative, with a positive tube radius"
        )
    solids: list[MeshGeometry] = [
        WirePolylineGeometry(
            path,
            radius=radius,
            radial_segments=radial_segments,
            cap_ends=cap_ends,
            corner_mode=corner_mode,
            corner_radius=corner_radius,
            corner_segments=corner_segments,
            up_hint=up_hint,
            min_segment_length=min_segment_length,
        )
        for path in path_values
    ]
    nodes: dict[tuple[int, int, int], tuple[Vec3, int]] = {}
    step = max(float(min_segment_length), _EPS)
    for path in path_values:
        for point in path:
            key = (
                round(point[0] / step),
                round(point[1] / step),
                round(point[2] / step),
            )
            previous = nodes.get(key)
            nodes[key] = (point, 1 if previous is None else previous[1] + 1)
    for point, occurrences in nodes.values():
        if occurrences < 2 or node_radius <= 0.0:
            continue
        solids.append(
            SphereGeometry(
                node_radius,
                width_segments=max(12, radial_segments),
                height_segments=max(8, radial_segments // 2),
            ).translate(*point)
        )
    if not solids:
        raise ValueError("tube network produced no geometry")
    if not cap_ends:
        result = MeshGeometry()
        for solid in solids:
            result.merge(solid)
        return result

    from articraft.sdk._mesh.boolean import _boolean_union_many

    try:
        return _boolean_union_many(solids)
    except ValueError as exc:
        raise ValueError("tube network could not form one closed manifold solid") from exc


__all__ = [
    "ArcPipeGeometry",
    "PipeGeometry",
    "SweepGeometry",
    "SweepSection",
    "WirePolylineGeometry",
    "sweep_profile_along_spline",
    "tube_from_spline_points",
    "tube_network_from_paths",
    "wire_from_points",
]
