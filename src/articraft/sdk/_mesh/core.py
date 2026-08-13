from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Integral
from typing import TYPE_CHECKING, Any, ClassVar, SupportsIndex, TypeAlias, cast

import numpy as np
import trimesh
from scipy.interpolate import CubicHermiteSpline  # pyright: ignore[reportMissingTypeStubs]

from articraft.sdk.errors import ValidationError

if TYPE_CHECKING:
    from build123d.topology import Shape
    from manifold3d import DoubleNx2

Vec2: TypeAlias = tuple[float, float]
Vec3: TypeAlias = tuple[float, float, float]
Face: TypeAlias = tuple[int, int, int]

_EPS = 1e-10


class _RevisionList(list[Any]):
    def __init__(
        self,
        values: Iterable[Any],
        *,
        changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(_as_tuple(value) for value in values)
        self._changed = changed or _ignore_change

    def __setitem__(self, key: SupportsIndex | slice, value: Any) -> None:
        normalized = (
            [_as_tuple(item) for item in value] if isinstance(key, slice) else _as_tuple(value)
        )
        super().__setitem__(key, normalized)
        self._changed()

    def __delitem__(self, key: SupportsIndex | slice) -> None:
        super().__delitem__(key)
        self._changed()

    def __iadd__(self, values: Iterable[Any]):
        super().__iadd__([_as_tuple(value) for value in values])
        self._changed()
        return self

    def __imul__(self, count: SupportsIndex):
        super().__imul__(count)
        self._changed()
        return self

    def append(self, value: Any) -> None:
        super().append(_as_tuple(value))
        self._changed()

    def clear(self) -> None:
        super().clear()
        self._changed()

    def extend(self, values: Iterable[Any]) -> None:
        super().extend(_as_tuple(value) for value in values)
        self._changed()

    def insert(self, index: SupportsIndex, value: Any) -> None:
        super().insert(index, _as_tuple(value))
        self._changed()

    def pop(self, index: SupportsIndex = -1) -> Any:
        value = super().pop(index)
        self._changed()
        return value

    def remove(self, value: Any) -> None:
        super().remove(value)
        self._changed()

    def reverse(self) -> None:
        super().reverse()
        self._changed()

    def sort(self, *, key: Callable[[Any], Any] | None = None, reverse: bool = False) -> None:
        super().sort(key=key, reverse=reverse)
        self._changed()


def _as_tuple(value: Any) -> Any:
    if isinstance(value, tuple):
        return value
    try:
        return tuple(value)
    except TypeError:
        return value


def _ignore_change() -> None:
    pass


def _vertex_array(vertices: Sequence[Sequence[float]]) -> np.ndarray:
    if not vertices:
        return np.empty((0, 3), dtype=np.float64)
    try:
        values = np.asarray(vertices, dtype=np.float64)
    except (TypeError, ValueError):
        values = np.empty((0, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        for index, vertex in enumerate(vertices):
            try:
                valid = len(vertex) == 3
            except TypeError:
                valid = False
            if not valid:
                raise ValidationError(f"vertices[{index}] must contain 3 finite values")
        raise ValidationError("vertices must contain 3 finite values")
    invalid = np.flatnonzero(~np.isfinite(values).all(axis=1))
    if len(invalid):
        raise ValidationError(f"vertices[{int(invalid[0])}] must contain 3 finite values")
    return np.ascontiguousarray(values, dtype=np.float64)


def _face_array(faces: Sequence[Sequence[int]], vertex_count: int) -> np.ndarray:
    if not faces:
        return np.empty((0, 3), dtype=np.int64)
    try:
        raw = np.asarray(faces)
    except (TypeError, ValueError):
        raw = np.empty((0, 3), dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        for index, face in enumerate(faces):
            try:
                valid = len(face) == 3
            except TypeError:
                valid = False
            if not valid:
                raise ValidationError(f"faces[{index}] must contain exactly 3 indices")
        raise ValidationError("faces must contain exactly 3 indices")
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
        for index, face in enumerate(faces):
            if any(not isinstance(value, Integral) or isinstance(value, bool) for value in face):
                raise ValidationError(f"faces[{index}] indices must be integers")
        raise ValidationError("face indices must be integers")
    values = np.ascontiguousarray(raw, dtype=np.int64)
    repeated = np.flatnonzero(
        (values[:, 0] == values[:, 1])
        | (values[:, 0] == values[:, 2])
        | (values[:, 1] == values[:, 2])
    )
    if len(repeated):
        raise ValidationError(f"faces[{int(repeated[0])}] must reference 3 distinct vertices")
    outside = np.flatnonzero((values < 0).any(axis=1) | (values >= vertex_count).any(axis=1))
    if len(outside):
        raise ValidationError(f"faces[{int(outside[0])}] references a vertex outside the mesh")
    return values


def _validated_mesh_arrays(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    vertex_values = _vertex_array(vertices)
    return vertex_values, _face_array(faces, len(vertex_values))


def _mean_point(points: Sequence[Vec3]) -> Vec3:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )


def _vec2(value: Sequence[float], *, name: str = "point") -> Vec2:
    if len(value) != 2:
        raise ValueError(f"{name} must have 2 elements")
    point = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in point):
        raise ValueError(f"{name} values must be finite")
    return point


def _vec3(value: Sequence[float], *, name: str = "point") -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{name} must have 3 elements")
    point = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(component) for component in point):
        raise ValueError(f"{name} values must be finite")
    return point


def _v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_scale(value: Vec3, scale: float) -> Vec3:
    return (value[0] * scale, value[1] * scale, value[2] * scale)


def _v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _v_norm(value: Vec3) -> float:
    return math.sqrt(_v_dot(value, value))


def _v_normalize(value: Vec3) -> Vec3:
    length = _v_norm(value)
    if length <= _EPS:
        raise ValueError("cannot normalize a zero-length vector")
    return _v_scale(value, 1.0 / length)


def _v_lerp(a: Vec3, b: Vec3, amount: float) -> Vec3:
    return _v_add(a, _v_scale(_v_sub(b, a), amount))


def _same2(a: Vec2, b: Vec2, tolerance: float = 1e-9) -> bool:
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def _same3(a: Vec3, b: Vec3, tolerance: float = 1e-9) -> bool:
    return all(abs(a[index] - b[index]) <= tolerance for index in range(3))


def _profile_2d(profile: Iterable[Sequence[float]], *, minimum: int = 3) -> list[Vec2]:
    points: list[Vec2] = []
    for index, raw in enumerate(profile):
        point = _vec2(raw, name=f"profile[{index}]")
        if not points or not _same2(points[-1], point):
            points.append(point)
    if len(points) > 1 and _same2(points[0], points[-1]):
        points.pop()
    if len(points) < minimum:
        raise ValueError(f"profile must contain at least {minimum} distinct points")
    return points


def _profile_3d(profile: Iterable[Sequence[float]], *, minimum: int = 2) -> list[Vec3]:
    points: list[Vec3] = []
    for index, raw in enumerate(profile):
        point = _vec3(raw, name=f"profile[{index}]")
        if not points or not _same3(points[-1], point):
            points.append(point)
    if len(points) > 1 and _same3(points[0], points[-1]):
        points.pop()
    if len(points) < minimum:
        raise ValueError(f"profile must contain at least {minimum} distinct points")
    return points


def _polygon_area(points: Sequence[Vec2]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _ensure_ccw(points: list[Vec2]) -> list[Vec2]:
    area = _polygon_area(points)
    if abs(area) <= _EPS:
        raise ValueError("profile area must be non-zero")
    return points if area > 0.0 else list(reversed(points))


def _point_in_polygon(point: Vec2, polygon: Sequence[Vec2]) -> bool:
    inside = False
    for start, end in pairwise([*polygon, polygon[0]]):
        cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
            point[0] - start[0]
        )
        if (
            abs(cross) <= 1e-9
            and min(start[0], end[0]) - 1e-9 <= point[0] <= max(start[0], end[0]) + 1e-9
            and min(start[1], end[1]) - 1e-9 <= point[1] <= max(start[1], end[1]) + 1e-9
        ):
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing = start[0] + (point[1] - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            if crossing >= point[0]:
                inside = not inside
    return inside


def _triangulate_simple(points: list[Vec2]) -> list[Face]:
    return _triangulate_with_holes(points, [])[1]


def _triangulate_with_holes(
    outer: list[Vec2], holes: list[list[Vec2]]
) -> tuple[list[Vec2], list[Face]]:
    outer = _ensure_ccw(outer)
    holes = [_ensure_ccw(hole) for hole in holes]
    for hole in holes:
        center = (
            sum(point[0] for point in hole) / len(hole),
            sum(point[1] for point in hole) / len(hole),
        )
        if not _point_in_polygon(center, outer):
            raise ValueError("hole profile must lie inside the outer profile")
    rings = [outer, *(list(reversed(hole)) for hole in holes)]
    flat = [point for ring in rings for point in ring]
    try:
        import manifold3d

        polygons = cast(
            "list[DoubleNx2[int]]",
            [np.asarray(ring, dtype=np.float64) for ring in rings],
        )
        raw = manifold3d.triangulate(polygons, epsilon=1e-9)
        triangles: list[Face] = [
            (int(face[0]), int(face[1]), int(face[2])) for face in np.asarray(raw)
        ]
        return flat, triangles
    except Exception as exc:
        raise ValueError("failed to triangulate profile") from exc


@dataclass
class MeshGeometry:
    """A small, mutable triangle mesh in meters."""

    vertices: list[Vec3] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    normal_crease_angle: float | None = None
    _revision: ClassVar[int] = 0
    _array_cache: ClassVar[tuple[int, np.ndarray, np.ndarray] | None] = None
    _watertight_cache: ClassVar[tuple[int, bool] | None] = None

    def __post_init__(self) -> None:
        vertices, faces = _validated_mesh_arrays(self.vertices, self.faces)
        crease = self.normal_crease_angle
        if crease is not None:
            crease = float(crease)
            if not math.isfinite(crease) or not 0.0 <= crease <= math.pi:
                raise ValidationError("normal_crease_angle must be between 0 and pi radians")
        object.__setattr__(self, "normal_crease_angle", crease)
        object.__setattr__(self, "_revision", 0)
        object.__setattr__(self, "_array_cache", None)
        object.__setattr__(self, "_watertight_cache", None)
        object.__setattr__(
            self,
            "vertices",
            _RevisionList(
                ((float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in vertices),
                changed=self._mark_changed,
            ),
        )
        object.__setattr__(
            self,
            "faces",
            _RevisionList(
                ((int(face[0]), int(face[1]), int(face[2])) for face in faces),
                changed=self._mark_changed,
            ),
        )
        self._store_array_cache(vertices, faces)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"vertices", "faces"} and "_array_cache" in self.__dict__:
            object.__setattr__(
                self,
                name,
                _RevisionList(value, changed=self._mark_changed),
            )
            self._mark_changed()
            return
        object.__setattr__(self, name, value)

    def _mark_changed(self) -> None:
        object.__setattr__(self, "_revision", self._revision + 1)
        object.__setattr__(self, "_array_cache", None)
        object.__setattr__(self, "_watertight_cache", None)

    def _store_array_cache(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        vertices.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "_array_cache", (self._revision, vertices, faces))

    def _mesh_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        cached = self._array_cache
        if cached is not None and cached[0] == self._revision:
            return cached[1], cached[2]
        vertices, faces = _validated_mesh_arrays(self.vertices, self.faces)
        self._store_array_cache(vertices, faces)
        return vertices, faces

    @property
    def _cache_token(self) -> tuple[int, int]:
        return id(self), self._revision

    def validate(self) -> None:
        self._mesh_arrays()
        crease = self.normal_crease_angle
        if crease is not None and (not math.isfinite(crease) or not 0.0 <= crease <= math.pi):
            raise ValidationError("normal_crease_angle must be between 0 and pi radians")

    @classmethod
    def from_trimesh(
        cls,
        mesh: trimesh.Trimesh,
        *,
        process: bool = False,
        normal_crease_angle: float | None = None,
    ) -> MeshGeometry:
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("mesh must be trimesh.Trimesh")
        if process:
            mesh = mesh.copy()
            mesh.process(validate=True)
        return cls(
            vertices=[
                (float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in mesh.vertices
            ],
            faces=[(int(face[0]), int(face[1]), int(face[2])) for face in mesh.faces],
            normal_crease_angle=normal_crease_angle,
        )

    def to_trimesh(self, *, process: bool = False) -> trimesh.Trimesh:
        vertices, faces = self._mesh_arrays()
        return trimesh.Trimesh(
            vertices=vertices.copy(),
            faces=faces.copy(),
            process=process,
            validate=process,
        )

    @property
    def bounds(self) -> tuple[Vec3, Vec3]:
        if not self.vertices:
            raise ValidationError("mesh has no vertices")
        values, _faces = self._mesh_arrays()
        return (
            (
                float(values[:, 0].min()),
                float(values[:, 1].min()),
                float(values[:, 2].min()),
            ),
            (
                float(values[:, 0].max()),
                float(values[:, 1].max()),
                float(values[:, 2].max()),
            ),
        )

    @property
    def is_watertight(self) -> bool:
        cached = self._watertight_cache
        if cached is not None and cached[0] == self._revision:
            return cached[1]
        watertight = bool(self.faces) and bool(self.to_trimesh().is_watertight)
        object.__setattr__(self, "_watertight_cache", (self._revision, watertight))
        return watertight

    def copy(self) -> MeshGeometry:
        vertices, faces = self._mesh_arrays()
        copied = MeshGeometry.__new__(MeshGeometry)
        object.__setattr__(copied, "_revision", 0)
        object.__setattr__(copied, "_array_cache", None)
        watertight = None if self._watertight_cache is None else (0, self._watertight_cache[1])
        object.__setattr__(copied, "_watertight_cache", watertight)
        object.__setattr__(copied, "normal_crease_angle", self.normal_crease_angle)
        object.__setattr__(
            copied,
            "vertices",
            _RevisionList(self.vertices, changed=copied._mark_changed),
        )
        object.__setattr__(
            copied,
            "faces",
            _RevisionList(self.faces, changed=copied._mark_changed),
        )
        copied._store_array_cache(vertices.copy(), faces.copy())
        return copied

    def set_normal_crease(self, angle: float) -> MeshGeometry:
        """Use hard normals where adjacent faces differ by more than ``angle``."""
        value = float(angle)
        if not math.isfinite(value) or not 0.0 <= value <= math.pi:
            raise ValueError("normal crease angle must be between 0 and pi radians")
        self.normal_crease_angle = value
        return self

    def use_smooth_normals(self) -> MeshGeometry:
        """Blend normals across every shared vertex."""
        self.normal_crease_angle = math.pi
        return self

    def use_hard_normals(self) -> MeshGeometry:
        """Keep a separate normal for every triangle corner."""
        self.normal_crease_angle = 0.0
        return self

    def add_vertex(self, x: float, y: float, z: float) -> int:
        vertex = _vec3((x, y, z), name="vertex")
        self.vertices.append(vertex)
        return len(self.vertices) - 1

    def add_face(self, a: int, b: int, c: int) -> None:
        face = (a, b, c)
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in face):
            raise ValidationError("face indices must be integers")
        normalized = (int(a), int(b), int(c))
        if len(set(normalized)) != 3:
            raise ValidationError("face must reference 3 distinct vertices")
        if min(normalized) < 0 or max(normalized) >= len(self.vertices):
            raise ValidationError("face references a vertex outside the mesh")
        self.faces.append(normalized)

    def merge(self, other: MeshGeometry) -> MeshGeometry:
        if not isinstance(other, MeshGeometry):
            raise TypeError("other must be MeshGeometry")
        other.validate()
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.faces.extend((a + offset, b + offset, c + offset) for a, b, c in other.faces)
        return self

    def translate(self, dx: float, dy: float, dz: float) -> MeshGeometry:
        delta = _vec3((dx, dy, dz), name="translation")
        self.vertices = [_v_add(vertex, delta) for vertex in self.vertices]
        return self

    def scale(self, sx: float, sy: float | None = None, sz: float | None = None) -> MeshGeometry:
        values = (float(sx), float(sx if sy is None else sy), float(sx if sz is None else sz))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("scale values must be finite")
        if any(abs(value) <= _EPS for value in values):
            raise ValueError("scale values must be non-zero")
        self.vertices = [
            (vertex[0] * values[0], vertex[1] * values[1], vertex[2] * values[2])
            for vertex in self.vertices
        ]
        if values[0] * values[1] * values[2] < 0.0:
            self.faces = [(a, c, b) for a, b, c in self.faces]
        return self

    def rotate(
        self,
        axis: Sequence[float],
        angle: float,
        *,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> MeshGeometry:
        direction = _v_normalize(_vec3(axis, name="axis"))
        pivot = _vec3(origin, name="origin")
        angle = float(angle)
        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
        matrix = trimesh.transformations.rotation_matrix(angle, direction, point=pivot)
        self.vertices = [
            (float(x), float(y), float(z))
            for x, y, z in trimesh.transform_points(self._mesh_arrays()[0], matrix)
        ]
        return self

    def rotate_x(self, angle: float) -> MeshGeometry:
        return self.rotate((1.0, 0.0, 0.0), angle)

    def rotate_y(self, angle: float) -> MeshGeometry:
        return self.rotate((0.0, 1.0, 0.0), angle)

    def rotate_z(self, angle: float) -> MeshGeometry:
        return self.rotate((0.0, 0.0, 1.0), angle)

    def to_obj(self) -> str:
        self.validate()
        lines = ["o mesh"]
        lines.extend(f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in self.vertices)
        lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.faces)
        return "\n".join([*lines, ""])


def _from_created(mesh: trimesh.Trimesh) -> MeshGeometry:
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return MeshGeometry.from_trimesh(mesh)


class BoxGeometry(MeshGeometry):
    def __init__(self, size: Sequence[float]):
        dimensions = _vec3(size, name="size")
        if any(value <= 0.0 for value in dimensions):
            raise ValueError("box size values must be positive")
        created = _from_created(trimesh.creation.box(extents=dimensions))
        super().__init__(created.vertices, created.faces)


class RoundedBoxGeometry(MeshGeometry):
    def __init__(
        self,
        size: Sequence[float],
        radius: float,
        *,
        tolerance: float = 0.0005,
        angular_tolerance: float = 0.08,
    ):
        from build123d import Box

        dimensions = _vec3(size, name="size")
        if any(value <= 0.0 for value in dimensions):
            raise ValueError("rounded box size values must be positive")
        radius = float(radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("rounded box radius must be finite and positive")
        if radius >= min(dimensions) * 0.5:
            raise ValueError("rounded box radius must be less than half its smallest size")
        box = Box(*dimensions)
        rounded = box.fillet(radius, box.edges())
        converted = build123d_to_mesh(
            rounded,
            tolerance=tolerance,
            angular_tolerance=angular_tolerance,
        )
        super().__init__(converted.vertices, converted.faces)


class CylinderGeometry(MeshGeometry):
    def __init__(
        self, radius: float, height: float, *, radial_segments: int = 24, closed: bool = True
    ):
        radius, height = float(radius), float(height)
        if radius <= 0.0 or height <= 0.0:
            raise ValueError("cylinder radius and height must be positive")
        sections = max(3, int(radial_segments))
        created = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        if not closed:
            created.update_faces(np.abs(created.face_normals[:, 2]) < 0.999)
            created.remove_unreferenced_vertices()
        converted = _from_created(created)
        super().__init__(converted.vertices, converted.faces)


class ConeGeometry(MeshGeometry):
    def __init__(
        self, radius: float, height: float, *, radial_segments: int = 24, closed: bool = True
    ):
        radius, height = float(radius), float(height)
        if radius <= 0.0 or height <= 0.0:
            raise ValueError("cone radius and height must be positive")
        sections = max(3, int(radial_segments))
        created = trimesh.creation.cone(radius=radius, height=height, sections=sections)
        created.apply_translation((0.0, 0.0, -height * 0.5))
        if not closed:
            normals = created.face_normals
            keep = np.abs(normals[:, 2]) < 0.999
            created.update_faces(keep)
            created.remove_unreferenced_vertices()
        converted = _from_created(created)
        super().__init__(converted.vertices, converted.faces)


class SphereGeometry(MeshGeometry):
    def __init__(self, radius: float, *, width_segments: int = 24, height_segments: int = 16):
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("sphere radius must be positive")
        count = (max(8, int(width_segments)), max(4, int(height_segments)))
        created = _from_created(trimesh.creation.uv_sphere(radius=radius, count=count))
        super().__init__(created.vertices, created.faces)


def _signed_power(value: float, exponent: float) -> float:
    return math.copysign(abs(value) ** exponent, value)


class SuperellipsoidGeometry(MeshGeometry):
    def __init__(
        self,
        radius: float | Sequence[float],
        *,
        latitude_exponent: float = 1.0,
        longitude_exponent: float = 1.0,
        radial_segments: int = 48,
        height_segments: int = 24,
    ):
        radii = (float(radius),) * 3 if isinstance(radius, (int, float)) else _vec3(radius)
        if any(not math.isfinite(value) or value <= 0.0 for value in radii):
            raise ValueError("superellipsoid radii must be finite and positive")
        latitude_exponent = float(latitude_exponent)
        longitude_exponent = float(longitude_exponent)
        if not math.isfinite(latitude_exponent) or latitude_exponent <= 0.0:
            raise ValueError("latitude_exponent must be finite and positive")
        if not math.isfinite(longitude_exponent) or longitude_exponent <= 0.0:
            raise ValueError("longitude_exponent must be finite and positive")
        if isinstance(radial_segments, bool) or not isinstance(radial_segments, Integral):
            raise ValueError("radial_segments must be an integer")
        if isinstance(height_segments, bool) or not isinstance(height_segments, Integral):
            raise ValueError("height_segments must be an integer")
        radial_segments = int(radial_segments)
        height_segments = int(height_segments)
        if radial_segments < 8:
            raise ValueError("radial_segments must be at least 8")
        if height_segments < 4:
            raise ValueError("height_segments must be at least 4")

        vertices: list[Vec3] = [(0.0, 0.0, -radii[2])]
        for row in range(1, height_segments):
            latitude = -0.5 * math.pi + math.pi * row / height_segments
            latitude_radius = _signed_power(math.cos(latitude), latitude_exponent)
            z = radii[2] * _signed_power(math.sin(latitude), latitude_exponent)
            for column in range(radial_segments):
                longitude = 2.0 * math.pi * column / radial_segments
                vertices.append(
                    (
                        radii[0]
                        * latitude_radius
                        * _signed_power(math.cos(longitude), longitude_exponent),
                        radii[1]
                        * latitude_radius
                        * _signed_power(math.sin(longitude), longitude_exponent),
                        z,
                    )
                )
        top = len(vertices)
        vertices.append((0.0, 0.0, radii[2]))

        faces: list[Face] = []
        first_row = 1
        for column in range(radial_segments):
            following = (column + 1) % radial_segments
            faces.append((0, first_row + following, first_row + column))
        for row in range(height_segments - 2):
            start = 1 + row * radial_segments
            following_start = start + radial_segments
            for column in range(radial_segments):
                following = (column + 1) % radial_segments
                faces.extend(
                    (
                        (start + column, start + following, following_start + following),
                        (start + column, following_start + following, following_start + column),
                    )
                )
        last_row = 1 + (height_segments - 2) * radial_segments
        for column in range(radial_segments):
            following = (column + 1) % radial_segments
            faces.append((last_row + column, last_row + following, top))
        super().__init__(vertices, faces)


class DomeGeometry(MeshGeometry):
    def __init__(
        self,
        radius: float | Sequence[float],
        *,
        radial_segments: int = 24,
        height_segments: int = 12,
        closed: bool = True,
    ):
        if isinstance(radius, (int, float)):
            radii = (float(radius),) * 3
        else:
            radii = _vec3(radius, name="radius")
        if any(value <= 0.0 for value in radii):
            raise ValueError("dome radii must be positive")
        radial_segments = max(3, int(radial_segments))
        height_segments = max(1, int(height_segments))
        phi = np.linspace(0.0, 0.5 * math.pi, height_segments + 1)
        profile = np.column_stack((radii[0] * np.sin(phi), radii[2] * np.cos(phi)))
        if closed:
            profile = np.vstack((profile, (0.0, 0.0)))
        created = trimesh.creation.revolve(profile, sections=radial_segments, cap=False)
        created.apply_scale((1.0, radii[1] / radii[0], 1.0))
        converted = _from_created(created)
        super().__init__(converted.vertices, converted.faces)


class CapsuleGeometry(MeshGeometry):
    def __init__(
        self,
        radius: float,
        length: float,
        *,
        radial_segments: int = 24,
        height_segments: int = 8,
    ):
        radius, length = float(radius), float(length)
        if radius <= 0.0 or length < 0.0:
            raise ValueError("capsule radius must be positive and length non-negative")
        if length <= _EPS:
            sphere = SphereGeometry(
                radius,
                width_segments=radial_segments,
                height_segments=max(4, height_segments * 2),
            )
            super().__init__(sphere.vertices, sphere.faces)
            return
        created = trimesh.creation.capsule(
            height=length,
            radius=radius,
            count=(max(8, int(radial_segments)), max(4, int(height_segments))),
        )
        converted = _from_created(created)
        super().__init__(converted.vertices, converted.faces)


class TorusGeometry(MeshGeometry):
    def __init__(
        self,
        radius: float,
        tube: float,
        *,
        radial_segments: int = 16,
        tubular_segments: int = 32,
    ):
        radius, tube = float(radius), float(tube)
        if radius <= 0.0 or tube <= 0.0:
            raise ValueError("torus radius and tube must be positive")
        created = _from_created(
            trimesh.creation.torus(
                major_radius=radius,
                minor_radius=tube,
                major_sections=max(3, int(tubular_segments)),
                minor_sections=max(3, int(radial_segments)),
            )
        )
        super().__init__(created.vertices, created.faces)


def _cap_connector(
    start: Vec2, end: Vec2, mode: str, samples: int, *, outward_z: float
) -> list[Vec2]:
    if mode not in {"flat", "round"}:
        raise ValueError("lathe cap mode must be 'flat' or 'round'")
    if mode == "flat":
        return [start, end]
    samples = max(2, int(samples))
    distance = math.hypot(start[0] - end[0], start[1] - end[1])
    radial_direction = 1.0 if start[0] >= end[0] else -1.0
    control = (
        (start[0] + end[0]) * 0.5 + radial_direction * distance * 0.35,
        (start[1] + end[1]) * 0.5 + math.copysign(distance * 0.5, outward_z),
    )
    return [
        (
            (1.0 - amount) ** 2 * start[0]
            + 2.0 * (1.0 - amount) * amount * control[0]
            + amount**2 * end[0],
            (1.0 - amount) ** 2 * start[1]
            + 2.0 * (1.0 - amount) * amount * control[1]
            + amount**2 * end[1],
        )
        for amount in (index / samples for index in range(samples + 1))
    ]


class LatheGeometry(MeshGeometry):
    @classmethod
    def from_shell_profiles(
        cls,
        outer_profile: Iterable[Sequence[float]],
        inner_profile: Iterable[Sequence[float]],
        *,
        segments: int = 32,
        start_cap: str = "flat",
        end_cap: str = "flat",
        lip_samples: int = 6,
    ) -> LatheGeometry:
        outer = _profile_2d(outer_profile, minimum=2)
        inner = _profile_2d(inner_profile, minimum=2)
        axis_direction = 1.0 if outer[-1][1] + inner[-1][1] >= outer[0][1] + inner[0][1] else -1.0
        end = _cap_connector(outer[-1], inner[-1], end_cap, lip_samples, outward_z=axis_direction)
        start = _cap_connector(
            inner[0], outer[0], start_cap, lip_samples, outward_z=-axis_direction
        )
        profile = [*outer, *end[1:], *reversed(inner[:-1]), *start[1:]]
        return cls(profile, segments=segments, closed=True)

    def __init__(
        self,
        profile: Iterable[Sequence[float]],
        *,
        segments: int = 32,
        closed: bool = True,
        angle: float = math.tau,
        start_angle: float = 0.0,
        axis: str = "z",
        cap_ends: bool = True,
    ):
        points = _profile_2d(profile, minimum=3) if closed else _profile_2d(profile, minimum=2)
        if closed:
            points = _ensure_ccw(points)
        if any(radius < -_EPS for radius, _z in points):
            raise ValueError("lathe profile radii must be non-negative")
        angle = float(angle)
        start_angle = float(start_angle)
        if not math.isfinite(angle) or not 0.0 < angle <= math.tau:
            raise ValueError("lathe angle must be greater than 0 and at most 2*pi")
        if not math.isfinite(start_angle):
            raise ValueError("lathe start_angle must be finite")
        axis = str(axis).lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError("lathe axis must be 'x', 'y', or 'z'")
        partial = not math.isclose(angle, math.tau, abs_tol=1e-10)
        if cap_ends and partial and not closed:
            raise ValueError("partial lathe end caps require a closed profile")
        revolve_points = [*points, points[0]] if closed else points
        if partial and closed:
            section_count = max(3, int(segments)) + 1
            theta = np.linspace(0.0, angle, section_count)
            vertices = np.asarray(
                [
                    (radius * math.cos(value), radius * math.sin(value), height)
                    for value in theta
                    for radius, height in points
                ],
                dtype=np.float64,
            )
            per_section = len(points)
            faces: list[Face] = []
            for section in range(section_count - 1):
                start = section * per_section
                following = start + per_section
                for index in range(per_section):
                    next_index = (index + 1) % per_section
                    faces.append((start + index, following + index, start + next_index))
                    faces.append((start + next_index, following + index, following + next_index))
            if cap_ends:
                cap_faces = _triangulate_simple(points)
                faces.extend(cap_faces)
                end_offset = (section_count - 1) * per_section
                faces.extend(
                    (c + end_offset, b + end_offset, a + end_offset) for a, b, c in cap_faces
                )
            created = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        else:
            created = trimesh.creation.revolve(
                np.asarray(revolve_points, dtype=np.float64),
                angle=angle,
                sections=max(3, int(segments)),
                cap=False,
            )
        if start_angle:
            created.apply_transform(
                trimesh.transformations.rotation_matrix(start_angle, (0.0, 0.0, 1.0))
            )
        if axis == "x":
            created.apply_transform(
                trimesh.transformations.rotation_matrix(math.pi / 2.0, (0.0, 1.0, 0.0))
            )
        elif axis == "y":
            created.apply_transform(
                trimesh.transformations.rotation_matrix(-math.pi / 2.0, (1.0, 0.0, 0.0))
            )
        converted = _from_created(created) if closed else MeshGeometry.from_trimesh(created)
        super().__init__(converted.vertices, converted.faces)


def _interpolate_loft_span(
    p0: Sequence[Vec3],
    p1: Sequence[Vec3],
    p2: Sequence[Vec3],
    p3: Sequence[Vec3],
    amounts: np.ndarray,
    *,
    interpolation: str,
    parameters: tuple[float, float, float, float],
    tension: float,
) -> list[list[Vec3]]:
    """Rings between ``p1`` and ``p2``, one per sample amount.

    The smooth mode is a cardinal (tension-scaled Catmull-Rom) segment; scipy
    evaluates the same Hermite cubic from the ring values and end derivatives.
    """

    start = np.asarray(p1, dtype=np.float64)
    end = np.asarray(p2, dtype=np.float64)
    t0, t1, t2, t3 = parameters
    if interpolation == "linear" or min(t2 - t0, t3 - t1, t2 - t1) <= _EPS:
        rings = start + (end - start) * amounts[:, None, None]
    else:
        slope_scale = 1.0 - tension
        first_derivative = slope_scale * (end - np.asarray(p0, dtype=np.float64)) / (t2 - t0)
        second_derivative = slope_scale * (np.asarray(p3, dtype=np.float64) - start) / (t3 - t1)
        spline = CubicHermiteSpline(
            np.asarray((t1, t2)),
            np.stack((start, end)),
            np.stack((first_derivative, second_derivative)),
        )
        rings = spline(t1 + (t2 - t1) * amounts)
    return [[(float(x), float(y), float(z)) for x, y, z in ring] for ring in rings]


def _loft_parameter_step(start: Vec3, end: Vec3, parameterization: str) -> float:
    if parameterization == "uniform":
        return 1.0
    distance = _v_norm(_v_sub(end, start))
    return distance if parameterization == "chord" else math.sqrt(distance)


def _interpolate_loft_rings(
    rings: Sequence[Sequence[Vec3]],
    centers: Sequence[Vec3],
    *,
    interpolation: str,
    samples_per_span: int,
    close_path: bool,
    parameterization: str,
    tension: float,
) -> list[list[Vec3]]:
    if interpolation not in {"linear", "catmull_rom"}:
        raise ValueError("loft interpolation must be 'linear' or 'catmull_rom'")
    if parameterization not in {"uniform", "chord", "centripetal"}:
        raise ValueError("loft parameterization must be 'uniform', 'chord', or 'centripetal'")
    tension = float(tension)
    if not math.isfinite(tension) or not 0.0 <= tension <= 1.0:
        raise ValueError("loft tension must be finite and between 0 and 1")
    samples = int(samples_per_span)
    if samples < 1:
        raise ValueError("loft samples_per_span must be at least 1")
    parameters = [0.0]
    for index in range(1, len(centers)):
        parameters.append(
            parameters[-1]
            + _loft_parameter_step(centers[index - 1], centers[index], parameterization)
        )
    closing_step = _loft_parameter_step(centers[-1], centers[0], parameterization)
    total = parameters[-1] + closing_step
    span_count = len(rings) if close_path else len(rings) - 1
    result: list[list[Vec3]] = []
    for span in range(span_count):
        p1 = rings[span]
        p2 = rings[(span + 1) % len(rings)]
        t1 = parameters[span]
        t2 = parameters[span + 1] if span + 1 < len(rings) else total
        if close_path:
            p0 = rings[span - 1]
            p3 = rings[(span + 2) % len(rings)]
            t0 = parameters[span - 1] if span > 0 else parameters[-1] - total
            if span == len(rings) - 1:
                t3 = total + parameters[1]
            elif span + 2 < len(rings):
                t3 = parameters[span + 2]
            else:
                t3 = total
        else:
            p0 = (
                rings[span - 1]
                if span > 0
                else [
                    _v_sub(_v_scale(point, 2.0), following)
                    for point, following in zip(p1, p2, strict=True)
                ]
            )
            t0 = parameters[span - 1] if span > 0 else 2.0 * t1 - t2
            p3 = (
                rings[span + 2]
                if span + 2 < len(rings)
                else [
                    _v_sub(_v_scale(point, 2.0), previous)
                    for point, previous in zip(p2, p1, strict=True)
                ]
            )
            t3 = parameters[span + 2] if span + 2 < len(rings) else 2.0 * t2 - t1
        result.extend(
            _interpolate_loft_span(
                p0,
                p1,
                p2,
                p3,
                np.arange(samples, dtype=np.float64) / samples,
                interpolation=interpolation,
                parameters=(t0, t1, t2, t3),
                tension=tension,
            )
        )
    if not close_path:
        result.append(list(rings[-1]))
    return result


def _connect_loft_rings(
    faces: list[Face],
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


def _add_rounded_loft_cap(
    vertices: list[Vec3],
    faces: list[Face],
    *,
    ring: Sequence[Vec3],
    neighbor: Sequence[Vec3],
    base_offset: int,
    start: bool,
    segments: int,
    length: float | None,
) -> None:
    center = _mean_point(ring)
    neighbor_center = _mean_point(neighbor)
    direction = _v_normalize(_v_sub(center, neighbor_center))
    radius = max(_v_norm(_v_sub(point, center)) for point in ring)
    cap_length = radius if length is None else float(length)
    if cap_length <= 0.0 or not math.isfinite(cap_length):
        raise ValueError("loft cap_length must be finite and positive")
    ring_offsets: list[int] = []
    for sample in range(1, segments):
        angle = math.pi * sample / (2.0 * segments)
        scale = math.cos(angle)
        ring_center = _v_add(center, _v_scale(direction, cap_length * math.sin(angle)))
        ring_offsets.append(len(vertices))
        vertices.extend(
            _v_add(ring_center, _v_scale(_v_sub(point, center), scale)) for point in ring
        )
    ordered = [*reversed(ring_offsets), base_offset] if start else [base_offset, *ring_offsets]
    for first, second in pairwise(ordered):
        _connect_loft_rings(faces, first, second, len(ring))
    tip_index = len(vertices)
    vertices.append(_v_add(center, _v_scale(direction, cap_length)))
    terminal_ring = ordered[0] if start else ordered[-1]
    for index in range(len(ring)):
        following = (index + 1) % len(ring)
        faces.append(
            (tip_index, terminal_ring + following, terminal_ring + index)
            if start
            else (terminal_ring + index, terminal_ring + following, tip_index)
        )


class LoftGeometry(MeshGeometry):
    @classmethod
    def from_shell_profiles(
        cls,
        outer_profiles: Iterable[Iterable[Sequence[float]]],
        inner_profiles: Iterable[Iterable[Sequence[float]]],
        *,
        interpolation: str = "linear",
        samples_per_span: int = 1,
        parameterization: str = "uniform",
        tension: float = 0.0,
    ) -> LoftGeometry:
        """Build a hollow loft directly from corresponding outer and inner profiles."""
        outer_rings = [_profile_3d(profile, minimum=3) for profile in outer_profiles]
        inner_rings = [_profile_3d(profile, minimum=3) for profile in inner_profiles]
        if len(outer_rings) != len(inner_rings):
            raise ValueError("outer and inner loft profiles must have the same section count")
        if len(outer_rings) < 2:
            raise ValueError("a hollow loft requires at least two outer and inner profiles")
        count = len(outer_rings[0])
        if any(len(ring) != count for ring in (*outer_rings, *inner_rings)):
            raise ValueError("outer and inner loft profiles must have the same point count")
        if any(
            _v_norm(_v_sub(outer, inner)) <= _EPS
            for outer_ring, inner_ring in zip(outer_rings, inner_rings, strict=True)
            for outer, inner in zip(outer_ring, inner_ring, strict=True)
        ):
            raise ValueError("outer and inner loft profiles must not touch")

        options = {
            "cap": False,
            "closed": True,
            "interpolation": interpolation,
            "samples_per_span": samples_per_span,
            "parameterization": parameterization,
            "tension": tension,
        }
        outer_loft = cls(outer_rings, **options)
        inner_loft = cls(inner_rings, **options)
        if len(outer_loft.vertices) != len(inner_loft.vertices):
            raise ValueError("outer and inner loft interpolation produced different ring counts")

        inner_offset = len(outer_loft.vertices)
        vertices = [*outer_loft.vertices, *inner_loft.vertices]
        faces = [*outer_loft.faces]
        faces.extend(
            (inner_offset + c, inner_offset + b, inner_offset + a) for a, b, c in inner_loft.faces
        )
        outer_end = len(outer_loft.vertices) - count
        inner_end = inner_offset + len(inner_loft.vertices) - count
        _connect_loft_rings(faces, inner_offset, 0, count)
        _connect_loft_rings(faces, outer_end, inner_end, count)

        result = cls.__new__(cls)
        MeshGeometry.__init__(result, vertices=vertices, faces=faces)
        return result

    def __init__(
        self,
        profiles: Iterable[Iterable[Sequence[float]]],
        *,
        cap: bool = True,
        closed: bool = True,
        interpolation: str = "linear",
        samples_per_span: int = 1,
        close_path: bool = False,
        parameterization: str = "uniform",
        tension: float = 0.0,
        cap_style: str = "flat",
        cap_segments: int = 6,
        cap_length: float | None = None,
    ):
        rings = [_profile_3d(profile, minimum=3 if closed else 2) for profile in profiles]
        if len(rings) < 2:
            raise ValueError("loft requires at least two profiles")
        if close_path and len(rings) < 3:
            raise ValueError("a closed loft path requires at least three profiles")
        count = len(rings[0])
        if any(len(ring) != count for ring in rings):
            raise ValueError("loft profiles must have the same point count")
        cap_style = str(cap_style).strip().lower().replace("-", "_")
        if cap_style not in {"flat", "round"}:
            raise ValueError("loft cap_style must be 'flat' or 'round'")
        if isinstance(cap_segments, bool) or not isinstance(cap_segments, Integral):
            raise ValueError("loft cap_segments must be an integer")
        cap_segments = int(cap_segments)
        if cap_segments < 2:
            raise ValueError("loft cap_segments must be at least 2")
        if cap_length is not None:
            cap_length = float(cap_length)
            if cap_length <= 0.0 or not math.isfinite(cap_length):
                raise ValueError("loft cap_length must be finite and positive")

        def ring_center(ring: Sequence[Vec3]) -> Vec3:
            return (
                sum(point[0] for point in ring) / len(ring),
                sum(point[1] for point in ring) / len(ring),
                sum(point[2] for point in ring) / len(ring),
            )

        def ring_normal(ring: Sequence[Vec3]) -> Vec3:
            center = ring_center(ring)
            normal = (0.0, 0.0, 0.0)
            for index, point in enumerate(ring):
                following = ring[(index + 1) % len(ring)]
                normal = _v_add(
                    normal,
                    _v_cross(_v_sub(point, center), _v_sub(following, center)),
                )
            if _v_norm(normal) <= _EPS:
                raise ValueError("loft profiles must enclose a non-zero planar area")
            return _v_normalize(normal)

        centers = [ring_center(ring) for ring in rings]
        center_span_count = len(rings) if close_path else len(rings) - 1
        if any(
            _v_norm(_v_sub(centers[(index + 1) % len(rings)], centers[index])) <= _EPS
            for index in range(center_span_count)
        ):
            raise ValueError("adjacent loft profiles must have distinct centers")

        if closed:
            reference_normal = ring_normal(rings[0])
            if not close_path and _v_dot(reference_normal, _v_sub(centers[-1], centers[0])) < 0.0:
                rings = [list(reversed(ring)) for ring in rings]
                reference_normal = _v_scale(reference_normal, -1.0)
            for index in range(1, len(rings)):
                current_normal = ring_normal(rings[index])
                if _v_dot(current_normal, reference_normal) < 0.0:
                    rings[index] = list(reversed(rings[index]))
                    current_normal = _v_scale(current_normal, -1.0)
                reference_normal = current_normal

        rings = _interpolate_loft_rings(
            rings,
            centers,
            interpolation=interpolation,
            samples_per_span=samples_per_span,
            close_path=close_path,
            parameterization=parameterization,
            tension=tension,
        )

        vertices = [point for ring in rings for point in ring]
        faces: list[Face] = []
        segment_count = count if closed else count - 1
        path_segment_count = len(rings) if close_path else len(rings) - 1
        for ring_index in range(path_segment_count):
            start = ring_index * count
            following_start = ((ring_index + 1) % len(rings)) * count
            for point_index in range(segment_count):
                following = (point_index + 1) % count
                faces.extend(
                    (
                        (start + point_index, start + following, following_start + following),
                        (
                            start + point_index,
                            following_start + following,
                            following_start + point_index,
                        ),
                    )
                )
        if cap and closed and not close_path:
            if cap_style == "round":
                _add_rounded_loft_cap(
                    vertices,
                    faces,
                    ring=rings[0],
                    neighbor=rings[1],
                    base_offset=0,
                    start=True,
                    segments=cap_segments,
                    length=cap_length,
                )
                _add_rounded_loft_cap(
                    vertices,
                    faces,
                    ring=rings[-1],
                    neighbor=rings[-2],
                    base_offset=(len(rings) - 1) * count,
                    start=False,
                    segments=cap_segments,
                    length=cap_length,
                )
                super().__init__(vertices, faces)
                return
            for ring_index, neighbor_index, offset in (
                (0, 1, 0),
                (len(rings) - 1, len(rings) - 2, (len(rings) - 1) * count),
            ):
                ring = rings[ring_index]
                normal = _v_cross(_v_sub(ring[1], ring[0]), _v_sub(ring[2], ring[0]))
                axis = max(range(3), key=lambda index: abs(normal[index]))
                projected: list[Vec2]
                if axis == 0:
                    projected = [(point[1], point[2]) for point in ring]
                elif axis == 1:
                    projected = [(point[0], point[2]) for point in ring]
                else:
                    projected = [(point[0], point[1]) for point in ring]
                order = list(range(count))
                if _polygon_area(projected) < 0.0:
                    order.reverse()
                triangles = [
                    (order[a], order[b], order[c])
                    for a, b, c in _triangulate_simple([projected[index] for index in order])
                ]
                center = ring_center(ring)
                neighbor = ring_center(rings[neighbor_index])
                outward = _v_sub(center, neighbor)
                for a, b, c in triangles:
                    face = (offset + a, offset + b, offset + c)
                    face_normal = _v_cross(
                        _v_sub(vertices[face[1]], vertices[face[0]]),
                        _v_sub(vertices[face[2]], vertices[face[0]]),
                    )
                    faces.append(
                        (face[2], face[1], face[0]) if _v_dot(face_normal, outward) < 0.0 else face
                    )
        super().__init__(vertices, faces)


class ExtrudeGeometry(MeshGeometry):
    @classmethod
    def centered(
        cls,
        profile: Iterable[Sequence[float]],
        height: float,
        *,
        cap: bool = True,
        closed: bool = True,
    ) -> ExtrudeGeometry:
        return cls(profile, height, cap=cap, center=True, closed=closed)

    @classmethod
    def from_z0(
        cls,
        profile: Iterable[Sequence[float]],
        height: float,
        *,
        cap: bool = True,
        closed: bool = True,
    ) -> ExtrudeGeometry:
        return cls(profile, height, cap=cap, center=False, closed=closed)

    def __init__(
        self,
        profile: Iterable[Sequence[float]],
        height: float,
        *,
        cap: bool = True,
        center: bool = True,
        closed: bool = True,
    ):
        height = float(height)
        if height <= 0.0:
            raise ValueError("extrude height must be positive")
        points = _ensure_ccw(_profile_2d(profile))
        z0 = -height * 0.5 if center else 0.0
        loft = LoftGeometry(
            [[(x, y, z0) for x, y in points], [(x, y, z0 + height) for x, y in points]],
            cap=cap and closed,
            closed=closed,
        )
        super().__init__(loft.vertices, loft.faces)


class ExtrudeWithHolesGeometry(MeshGeometry):
    def __init__(
        self,
        outer_profile: Iterable[Sequence[float]],
        hole_profiles: Iterable[Iterable[Sequence[float]]],
        height: float,
        *,
        cap: bool = True,
        center: bool = True,
        closed: bool = True,
    ):
        height = float(height)
        if height <= 0.0:
            raise ValueError("extrude height must be positive")
        outer = _ensure_ccw(_profile_2d(outer_profile))
        holes = [_ensure_ccw(_profile_2d(profile)) for profile in hole_profiles]
        if not holes:
            plain = ExtrudeGeometry(outer, height, cap=cap, center=center, closed=closed)
            super().__init__(plain.vertices, plain.faces)
            return
        z0 = -height * 0.5 if center else 0.0
        z1 = z0 + height
        vertices: list[Vec3] = []
        faces: list[Face] = []
        bottom_indices: list[int] = []
        top_indices: list[int] = []
        for ring, inward in ((outer, False), *((hole, True) for hole in holes)):
            oriented_ring = list(reversed(ring)) if inward else ring
            start = len(vertices)
            vertices.extend((x, y, z0) for x, y in oriented_ring)
            vertices.extend((x, y, z1) for x, y in oriented_ring)
            count = len(oriented_ring)
            bottom_indices.extend(range(start, start + count))
            top_indices.extend(range(start + count, start + 2 * count))
            for index in range(count):
                following = (index + 1) % count
                faces.extend(
                    (
                        (start + index, start + following, start + count + following),
                        (start + index, start + count + following, start + count + index),
                    )
                )
        if cap and closed:
            flat, triangles = _triangulate_with_holes(outer, holes)
            if len(flat) != len(bottom_indices):
                raise RuntimeError("hole triangulation returned an unexpected vertex layout")
            faces.extend(
                (bottom_indices[c], bottom_indices[b], bottom_indices[a]) for a, b, c in triangles
            )
            faces.extend((top_indices[a], top_indices[b], top_indices[c]) for a, b, c in triangles)
        super().__init__(vertices, faces)


def build123d_to_mesh(
    shape: Shape, *, tolerance: float = 0.0001, angular_tolerance: float = 0.1
) -> MeshGeometry:
    """Tessellate a build123d shape, including its current Location, into a mesh."""

    from build123d.topology import Shape

    if not isinstance(shape, Shape):
        raise TypeError("shape must be a build123d Shape")
    if (
        tolerance <= 0.0
        or angular_tolerance <= 0.0
        or not math.isfinite(tolerance)
        or not math.isfinite(angular_tolerance)
    ):
        raise ValueError("tessellation tolerances must be positive")
    try:
        validity = shape.is_valid
        if not (validity() if callable(validity) else validity):
            raise ValidationError("build123d shape is not valid")
    except AttributeError:
        pass
    vertices, faces = shape.tessellate(float(tolerance), float(angular_tolerance))
    if not vertices or not faces:
        raise ValidationError("build123d shape tessellated to an empty mesh")
    geometry = MeshGeometry(
        vertices=[(float(vertex.X), float(vertex.Y), float(vertex.Z)) for vertex in vertices],
        faces=[(int(face[0]), int(face[1]), int(face[2])) for face in faces],
    )
    mesh = geometry.to_trimesh(process=True)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return MeshGeometry.from_trimesh(mesh)


def geometry_to_trimesh(geometry: object, tolerance: float) -> trimesh.Trimesh:
    """Convert either supported SDK geometry type to the shared mesh representation."""

    from build123d.topology import Shape

    if isinstance(geometry, Shape):
        vertices, faces = geometry.tessellate(tolerance)
        if not vertices or not faces:
            raise ValidationError("build123d shape produced an empty mesh")
        raw_vertices = np.asarray([[v.X, v.Y, v.Z] for v in vertices], dtype=np.float64)
        raw_faces = np.asarray(faces, dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=raw_vertices, faces=raw_faces, process=False)
    elif isinstance(geometry, MeshGeometry):
        mesh = geometry.to_trimesh()
    else:
        raise ValidationError("geometry must be a build123d Shape or MeshGeometry")
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValidationError("geometry produced an empty mesh")
    return mesh


__all__ = [
    "BoxGeometry",
    "CapsuleGeometry",
    "ConeGeometry",
    "CylinderGeometry",
    "DomeGeometry",
    "ExtrudeGeometry",
    "ExtrudeWithHolesGeometry",
    "LatheGeometry",
    "LoftGeometry",
    "MeshGeometry",
    "RoundedBoxGeometry",
    "SphereGeometry",
    "SuperellipsoidGeometry",
    "TorusGeometry",
    "build123d_to_mesh",
]
