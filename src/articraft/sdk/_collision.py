from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import fcl
import manifold3d
import numpy as np
import trimesh

from articraft.sdk._mesh.core import MeshGeometry, geometry_to_trimesh
from articraft.sdk.errors import LoopClosureError, ValidationError
from articraft.sdk.joints import (
    Articulation,
    ArticulationType,
    Origin,
    SpanTo,
    partition_articulations,
)
from articraft.sdk.object import ArticulatedObject, Geometry

Vec3: TypeAlias = tuple[float, float, float]
Bounds: TypeAlias = tuple[Vec3, Vec3]
Mat4: TypeAlias = np.ndarray

_NUMERICAL_CONTACT_DEPTH_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ContactInfo:
    position: Vec3 | None
    normal: Vec3 | None
    depth: float | None


@dataclass(frozen=True)
class CollisionQuery:
    part_a: str
    part_b: str
    collided: bool
    shape_a: str | None = None
    shape_b: str | None = None
    contacts: tuple[ContactInfo, ...] = ()
    overlap_depth: Vec3 | None = None
    overlap_volume: float = 0.0

    @property
    def max_depth(self) -> float | None:
        depths = [contact.depth for contact in self.contacts if contact.depth is not None]
        return max(depths) if depths else None


@dataclass(frozen=True)
class DistanceFinding:
    part_a: str
    part_b: str
    distance: float
    collided: bool
    shape_a: str | None = None
    shape_b: str | None = None
    nearest_a: Vec3 | None = None
    nearest_b: Vec3 | None = None
    contacts: tuple[ContactInfo, ...] = ()


DistanceQuery: TypeAlias = DistanceFinding


@dataclass(frozen=True)
class GeometryConnectivityFinding:
    part: str
    connected: int
    total: int
    disconnected: tuple[str, ...]
    contact_tol: float


@dataclass(frozen=True)
class _LocalMesh:
    mesh: trimesh.Trimesh
    fcl_model: object


@dataclass(frozen=True)
class _CollisionEntry:
    part_name: str
    shape_name: str
    obj: object
    bounds: Bounds
    world_vertices: np.ndarray
    world_mesh: trimesh.Trimesh


@dataclass(frozen=True)
class _GeometryComponent:
    name: str
    obj: object
    mesh: trimesh.Trimesh


class MeshCollisionKernel:
    def __init__(self, model: ArticulatedObject, *, mesh_tolerance: float) -> None:
        if mesh_tolerance <= 0.0 or not math.isfinite(mesh_tolerance):
            raise ValidationError("mesh_tolerance must be a positive finite number")
        self.model = model
        self.mesh_tolerance = float(mesh_tolerance)
        self._mesh_cache: dict[tuple[object, ...], _LocalMesh] = {}
        self._loop_plan: tuple[list[Articulation], list[tuple[Articulation, np.ndarray]]] | None = (
            None
        )
        self._loop_cache: dict[tuple[tuple[str, float], ...], dict[str, float]] = {}

    def part_world_position(self, part_name: str, pose: dict[str, float]) -> Vec3:
        transform = self.world_transforms(pose).get(part_name)
        if transform is None:
            raise ValidationError(f"unknown part: {part_name!r}")
        return _array_to_vec3(transform[:3, 3]) or (0.0, 0.0, 0.0)

    def part_world_vertices(self, part_name: str, pose: dict[str, float]) -> np.ndarray:
        entries = self._selector_entries(part_name, None, pose)
        vertices = [entry.world_vertices for entry in entries]
        if not vertices:
            raise ValidationError(f"part {part_name!r} has no geometry")
        # Bounds vertices are sufficient for all projection and bounds checks.
        return np.concatenate(vertices, axis=0)

    def part_world_bounds(self, part_name: str, pose: dict[str, float]) -> Bounds:
        return _points_bounds(self.part_world_vertices(part_name, pose))

    def shape_world_bounds(
        self,
        part_name: str,
        shape_name: str,
        pose: dict[str, float],
    ) -> Bounds:
        return self._selector_entries(part_name, shape_name, pose)[0].bounds

    def world_transforms(self, pose: dict[str, float]) -> dict[str, Mat4]:
        return self._place(self._solve(pose))

    def _solve(self, pose: dict[str, float]) -> dict[str, float]:
        """Complete an authored pose: drives first, then any closed loops.

        A ram is solved in closed form by its drives. A linkage with no closed
        form -- a four bar hood hinge, a folding brace -- is solved numerically
        instead, so its follower joints take the values that keep every loop pin
        closed.
        """

        candidates, targets = self._loop_solution_plan()
        # Whatever the caller posed leads; the rest of the ring follows.
        followers = [item for item in candidates if item.name not in pose]
        if not followers:
            return self._resolve_drives(pose)
        key = tuple(sorted((name, float(value)) for name, value in pose.items()))
        cached = self._loop_cache.get(key)
        if cached is not None:
            return dict(cached)
        solved = _solve_loops(self, pose, followers, targets)
        if len(self._loop_cache) >= _LOOP_CACHE_LIMIT:
            self._loop_cache.clear()
        self._loop_cache[key] = dict(solved)
        return solved

    def _loop_solution_plan(
        self,
    ) -> tuple[list[Articulation], list[tuple[Articulation, np.ndarray]]]:
        """Which joints follow, and where each loop pin sits in its child.

        The pin location is read once at the authored rest pose, which is the
        assembled configuration, so it is the same fact the exporter writes into
        the USD joint frames.
        """

        if self._loop_plan is not None:
            return self._loop_plan
        tree, loops = partition_articulations(self.model.articulations)
        candidates: list[Articulation] = []
        targets: list[tuple[Articulation, np.ndarray]] = []
        if loops:
            parent_of = {item.child: item for item in tree}

            def chain(part: str) -> list[Articulation]:
                walked: list[Articulation] = []
                while part in parent_of:
                    walked.append(parent_of[part])
                    part = parent_of[part].parent
                return walked

            rest = self._place(self._resolve_drives({}))
            for closure in loops:
                # Every movable joint on either side of the pin belongs to the
                # loop. Which of them lead and which follow is decided per pose:
                # whatever the caller poses drives, the rest are solved for.
                ring = {
                    item.name: item
                    for item in (*chain(closure.parent), *chain(closure.child))
                    if item.drive is None and item.articulation_type != ArticulationType.FIXED
                }
                if not ring:
                    continue
                pin = _apply(
                    rest[closure.parent] @ _origin_matrix(closure.origin),
                    np.zeros(3),
                )
                targets.append((closure, _apply(np.linalg.inv(rest[closure.child]), pin)))
                candidates.extend(
                    item for item in ring.values() if all(item is not seen for seen in candidates)
                )
        self._loop_plan = (candidates, targets)
        return self._loop_plan

    def _resolve_drives(self, pose: dict[str, float]) -> dict[str, float]:
        """Fill in the joints the mechanism decides rather than the author.

        Values handed in for driven joints are discarded: the mechanism owns
        them. Drives are solved shallowest parent first, re-placing the model
        after each, so a driven joint hanging off another driven joint reads a
        final frame rather than a stale one.
        """

        driven = [item for item in self.model.articulations if item.drive is not None]
        if not driven:
            return pose
        resolved = {
            name: value
            for name, value in pose.items()
            if name not in {item.name for item in driven}
        }
        tree, _loops = partition_articulations(self.model.articulations)
        parent_of = {item.child: item.parent for item in tree}

        def _depth(part: str) -> int:
            depth = 0
            while part in parent_of:
                part = parent_of[part]
                depth += 1
            return depth

        for articulation in sorted(driven, key=lambda item: _depth(item.parent)):
            drive = articulation.drive
            if drive is None:
                continue
            placed = self._place(resolved)
            target = placed.get(drive.part)
            frame = placed.get(articulation.parent)
            if target is None or frame is None:
                continue
            joint = frame @ _origin_matrix(articulation.origin)
            anchor = _apply(target, np.asarray(drive.point, dtype=np.float64))
            local = _apply(np.linalg.inv(joint), anchor)
            if isinstance(drive, SpanTo):
                # The axial component, signed: an anchor behind the origin must
                # retract the joint, and any perpendicular offset is the frame's
                # business, not extra travel.
                axial = float(np.dot(local, _normalize(articulation.axis)))
                resolved[articulation.name] = axial - drive.rest_length
            else:
                resolved[articulation.name] = _angle_about(
                    _normalize(articulation.axis),
                    np.asarray(drive.reference, dtype=np.float64),
                    local,
                )
        return resolved

    def _place(self, pose: dict[str, float]) -> dict[str, Mat4]:
        parts = {part.name: part for part in self.model.parts}
        children: dict[str, list[Articulation]] = {name: [] for name in parts}
        child_names: set[str] = set()
        # Only the spanning tree places parts. A loop closing joint is a
        # constraint on the mechanism, not a second place to hang its child.
        tree, _loops = partition_articulations(self.model.articulations)
        for articulation in tree:
            if articulation.parent in parts and articulation.child in parts:
                children[articulation.parent].append(articulation)
                child_names.add(articulation.child)

        roots = [name for name in parts if name not in child_names]
        transforms: dict[str, Mat4] = {}
        stack = [(root, _identity()) for root in roots]
        while stack:
            part_name, transform = stack.pop()
            if part_name in transforms:
                continue
            transforms[part_name] = transform
            for articulation in children.get(part_name, []):
                child_transform = (
                    transform
                    @ _origin_matrix(articulation.origin)
                    @ _motion_matrix(articulation, float(pose.get(articulation.name, 0.0)))
                )
                stack.append((articulation.child, child_transform))
        return transforms

    def collision_between(
        self,
        part_a: str,
        part_b: str,
        pose: dict[str, float],
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
        max_contacts: int = 16,
    ) -> CollisionQuery:
        pairs = self._entry_pairs(part_a, part_b, shape_a, shape_b, pose)
        collisions = [
            result
            for entry_a, entry_b in pairs
            if (result := _collide_entries(entry_a, entry_b, max_contacts=max_contacts)).collided
        ]
        if collisions:
            return max(collisions, key=_collision_rank)
        return CollisionQuery(
            part_a=part_a,
            part_b=part_b,
            shape_a=shape_a,
            shape_b=shape_b,
            collided=False,
        )

    def distance_between(
        self,
        part_a: str,
        part_b: str,
        pose: dict[str, float],
        *,
        shape_a: str | None = None,
        shape_b: str | None = None,
        max_contacts: int = 16,
    ) -> DistanceQuery:
        pairs = self._entry_pairs(part_a, part_b, shape_a, shape_b, pose)
        distances = [
            _distance_entries(entry_a, entry_b, max_contacts=max_contacts)
            for entry_a, entry_b in pairs
        ]
        if not distances:
            raise ValidationError("geometry selectors must identify two different shapes")
        return min(distances, key=lambda item: item.distance)

    def pair_distances(
        self,
        pose: dict[str, float],
        *,
        max_contacts: int = 16,
    ) -> list[DistanceQuery]:
        parts = [part.name for part in self.model.parts]
        by_part: dict[str, list[_CollisionEntry]] = {part: [] for part in parts}
        for entry in self._all_entries(pose):
            by_part[entry.part_name].append(entry)

        results: list[DistanceQuery] = []
        for index, part_a in enumerate(parts):
            for part_b in parts[index + 1 :]:
                distances = [
                    _distance_entries(entry_a, entry_b, max_contacts=max_contacts)
                    for entry_a in by_part[part_a]
                    for entry_b in by_part[part_b]
                ]
                if not distances:
                    raise ValidationError("parts must contain geometry")
                results.append(min(distances, key=lambda item: item.distance))
        return results

    def meaningful_overlaps(
        self,
        pose: dict[str, float],
        *,
        overlap_tol: float,
        overlap_volume_tol: float,
        max_contacts: int = 16,
    ) -> list[CollisionQuery]:
        entries = self._all_entries(pose)
        found: list[CollisionQuery] = []
        for index, entry_a in enumerate(entries):
            for entry_b in entries[index + 1 :]:
                if entry_a.part_name == entry_b.part_name:
                    continue
                overlap_depth = _bounds_overlap_depth(entry_a.bounds, entry_b.bounds)
                overlap_volume = math.prod(overlap_depth)
                if not all(depth > overlap_tol for depth in overlap_depth):
                    continue
                if overlap_volume <= overlap_volume_tol:
                    continue
                query = _collide_entries(entry_a, entry_b, max_contacts=max_contacts)
                if not query.collided:
                    continue
                solid_intersection = _solid_intersection_volume(entry_a, entry_b)
                if solid_intersection is not None:
                    if solid_intersection <= overlap_volume_tol:
                        continue
                elif (
                    query.max_depth is not None
                    and query.max_depth <= _NUMERICAL_CONTACT_DEPTH_TOLERANCE
                ):
                    continue
                found.append(
                    CollisionQuery(
                        part_a=query.part_a,
                        part_b=query.part_b,
                        shape_a=query.shape_a,
                        shape_b=query.shape_b,
                        collided=True,
                        contacts=query.contacts,
                        overlap_depth=overlap_depth,
                        overlap_volume=overlap_volume,
                    )
                )
        return found

    def disconnected_geometry_islands(
        self,
        *,
        contact_tol: float,
    ) -> list[GeometryConnectivityFinding]:
        findings: list[GeometryConnectivityFinding] = []
        for part in self.model.parts:
            components = self._geometry_components(part.name)
            if len(components) <= 1:
                continue
            groups, nearest = _component_connectivity(components, contact_tol=contact_tol)
            largest = max(groups, key=lambda group: (len(group), -min(group)))
            if len(largest) == len(components):
                continue

            disconnected: list[str] = []
            for index, component in enumerate(components):
                if index in largest:
                    continue
                nearest_index, distance = nearest.get(index, (None, None))
                if nearest_index is None or distance is None:
                    disconnected.append(component.name)
                else:
                    disconnected.append(
                        f"{component.name} nearest={components[nearest_index].name} "
                        f"distance={distance:.6g}"
                    )
            findings.append(
                GeometryConnectivityFinding(
                    part=part.name,
                    connected=len(largest),
                    total=len(components),
                    disconnected=tuple(disconnected),
                    contact_tol=contact_tol,
                )
            )
        return findings

    def _all_entries(self, pose: dict[str, float]) -> list[_CollisionEntry]:
        transforms = self.world_transforms(pose)
        return [
            self._entry(part.name, shape.name, shape.geometry, transforms[part.name])
            for part in self.model.parts
            for shape in part._iter_shapes()
        ]

    def _selector_entries(
        self,
        part_name: str,
        shape_name: str | None,
        pose: dict[str, float],
    ) -> list[_CollisionEntry]:
        part = self.model.get_part(part_name)
        transform = self.world_transforms(pose).get(part.name)
        if transform is None:
            raise ValidationError(f"unknown part: {part.name!r}")
        if shape_name is not None:
            shape_name = str(shape_name).strip()
            geometry = part.get_shape(shape_name)
            return [self._entry(part.name, shape_name, geometry, transform)]
        return [
            self._entry(part.name, shape.name, shape.geometry, transform)
            for shape in part._iter_shapes()
        ]

    def _entry_pairs(
        self,
        part_a: str,
        part_b: str,
        shape_a: str | None,
        shape_b: str | None,
        pose: dict[str, float],
    ) -> list[tuple[_CollisionEntry, _CollisionEntry]]:
        entries_a = self._selector_entries(part_a, shape_a, pose)
        entries_b = self._selector_entries(part_b, shape_b, pose)
        return [
            (entry_a, entry_b)
            for entry_a in entries_a
            for entry_b in entries_b
            if not (
                entry_a.part_name == entry_b.part_name and entry_a.shape_name == entry_b.shape_name
            )
        ]

    def _entry(
        self,
        part_name: str,
        shape_name: str,
        geometry: Geometry,
        transform: Mat4,
    ) -> _CollisionEntry:
        local_mesh = self._local_mesh(geometry)
        world_mesh = local_mesh.mesh.copy()
        world_mesh.apply_transform(transform)
        world_vertices = np.asarray(world_mesh.vertices, dtype=np.float64)
        return _CollisionEntry(
            part_name=part_name,
            shape_name=shape_name,
            obj=fcl.CollisionObject(local_mesh.fcl_model, _fcl_transform(transform)),
            bounds=_points_bounds(world_vertices),
            world_vertices=world_vertices,
            world_mesh=world_mesh,
        )

    def _local_mesh(self, geometry: Geometry) -> _LocalMesh:
        cache_key = _geometry_cache_key(geometry, self.mesh_tolerance)
        cached = self._mesh_cache.get(cache_key)
        if cached is not None:
            return cached
        mesh = geometry_to_trimesh(geometry, self.mesh_tolerance)
        cached = _LocalMesh(mesh=mesh, fcl_model=_mesh_to_bvh(mesh))
        self._mesh_cache[cache_key] = cached
        return cached

    def _geometry_components(self, part_name: str) -> tuple[_GeometryComponent, ...]:
        part = self.model.get_part(part_name)
        components: list[_GeometryComponent] = []
        for shape in part._iter_shapes():
            mesh = self._local_mesh(shape.geometry).mesh
            split = _split_mesh(mesh)
            for index, component in enumerate(split, start=1):
                suffix = "" if len(split) == 1 else f"#{index:03d}"
                components.append(
                    _GeometryComponent(
                        name=f"{shape.name}{suffix}",
                        obj=fcl.CollisionObject(_mesh_to_bvh(component), fcl.Transform()),
                        mesh=component,
                    )
                )
        return tuple(components)


def _component_connectivity(
    components: tuple[_GeometryComponent, ...],
    *,
    contact_tol: float,
) -> tuple[list[set[int]], dict[int, tuple[int, float]]]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(components))}
    nearest: dict[int, tuple[int, float]] = {}
    for index, component in enumerate(components):
        for other_index in range(index + 1, len(components)):
            other = components[other_index]
            collided, distance = _object_distance(component.obj, other.obj)
            if not collided and _meshes_have_overlapping_bounds(component.mesh, other.mesh):
                solid_intersection = _mesh_intersection_volume(component.mesh, other.mesh)
                if solid_intersection is not None and solid_intersection > 0.0:
                    collided, distance = True, 0.0
            if collided or distance <= contact_tol:
                adjacency[index].add(other_index)
                adjacency[other_index].add(index)
            _remember_nearest(nearest, index, other_index, distance)
            _remember_nearest(nearest, other_index, index, distance)

    remaining = set(range(len(components)))
    groups: list[set[int]] = []
    while remaining:
        start = min(remaining)
        group: set[int] = set()
        stack = [start]
        remaining.remove(start)
        while stack:
            current = stack.pop()
            group.add(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        groups.append(group)
    return groups, nearest


def _remember_nearest(
    nearest: dict[int, tuple[int, float]],
    index: int,
    other_index: int,
    distance: float,
) -> None:
    current = nearest.get(index)
    if current is None or distance < current[1]:
        nearest[index] = (other_index, distance)


def _collide_entries(
    entry_a: _CollisionEntry,
    entry_b: _CollisionEntry,
    *,
    max_contacts: int,
) -> CollisionQuery:
    request = fcl.CollisionRequest(
        num_max_contacts=max(1, int(max_contacts)),
        enable_contact=True,
    )
    result = fcl.CollisionResult()
    collided = int(fcl.collide(entry_a.obj, entry_b.obj, request, result)) > 0
    overlap_depth = _bounds_overlap_depth(entry_a.bounds, entry_b.bounds)
    if not collided and all(depth > 0.0 for depth in overlap_depth):
        solid_intersection = _solid_intersection_volume(entry_a, entry_b)
        collided = solid_intersection is not None and solid_intersection > 0.0
    contacts = tuple(_contact_info(contact) for contact in getattr(result, "contacts", []) or [])
    return CollisionQuery(
        part_a=entry_a.part_name,
        part_b=entry_b.part_name,
        shape_a=entry_a.shape_name,
        shape_b=entry_b.shape_name,
        collided=collided,
        contacts=contacts,
        overlap_depth=overlap_depth,
        overlap_volume=math.prod(overlap_depth),
    )


def _distance_entries(
    entry_a: _CollisionEntry,
    entry_b: _CollisionEntry,
    *,
    max_contacts: int,
) -> DistanceQuery:
    collision = _collide_entries(entry_a, entry_b, max_contacts=max_contacts)
    if collision.collided:
        return DistanceQuery(
            part_a=entry_a.part_name,
            part_b=entry_b.part_name,
            shape_a=entry_a.shape_name,
            shape_b=entry_b.shape_name,
            distance=0.0,
            collided=True,
            contacts=collision.contacts,
        )

    request = fcl.DistanceRequest(enable_nearest_points=True)
    result = fcl.DistanceResult()
    distance = float(fcl.distance(entry_a.obj, entry_b.obj, request, result))
    nearest_points = getattr(result, "nearest_points", None)
    nearest_a = None
    nearest_b = None
    if isinstance(nearest_points, (list, tuple)) and len(nearest_points) == 2:
        nearest_a = _array_to_vec3(nearest_points[0])
        nearest_b = _array_to_vec3(nearest_points[1])
    return DistanceQuery(
        part_a=entry_a.part_name,
        part_b=entry_b.part_name,
        shape_a=entry_a.shape_name,
        shape_b=entry_b.shape_name,
        distance=distance,
        collided=False,
        nearest_a=nearest_a,
        nearest_b=nearest_b,
    )


def _object_distance(obj_a: object, obj_b: object) -> tuple[bool, float]:
    collision_result = fcl.CollisionResult()
    collided = int(fcl.collide(obj_a, obj_b, fcl.CollisionRequest(), collision_result)) > 0
    if collided:
        return True, 0.0
    distance_result = fcl.DistanceResult()
    distance = float(fcl.distance(obj_a, obj_b, fcl.DistanceRequest(), distance_result))
    return False, distance


def _solid_intersection_volume(
    entry_a: _CollisionEntry,
    entry_b: _CollisionEntry,
) -> float | None:
    return _mesh_intersection_volume(entry_a.world_mesh, entry_b.world_mesh)


def _meshes_have_overlapping_bounds(a: trimesh.Trimesh, b: trimesh.Trimesh) -> bool:
    overlap = np.minimum(a.bounds[1], b.bounds[1]) - np.maximum(a.bounds[0], b.bounds[0])
    return bool(np.all(overlap > 0.0))


def _mesh_intersection_volume(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
) -> float | None:
    if not mesh_a.is_watertight or not mesh_b.is_watertight:
        return None
    try:
        manifold_a = manifold3d.Manifold(
            manifold3d.Mesh(
                np.asarray(mesh_a.vertices, dtype=np.float32),
                np.asarray(mesh_a.faces, dtype=np.uint32),
            )
        )
        manifold_b = manifold3d.Manifold(
            manifold3d.Mesh(
                np.asarray(mesh_b.vertices, dtype=np.float32),
                np.asarray(mesh_b.faces, dtype=np.uint32),
            )
        )
        if (
            manifold_a.status() != manifold3d.Error.NoError
            or manifold_b.status() != manifold3d.Error.NoError
        ):
            return None
        intersection = manifold_a ^ manifold_b
        if intersection.status() != manifold3d.Error.NoError or intersection.is_empty():
            return 0.0
        return abs(float(intersection.volume()))
    except Exception:
        return None


def _geometry_cache_key(geometry: Geometry, tolerance: float) -> tuple[object, ...]:
    if isinstance(geometry, MeshGeometry):
        return ("mesh", *geometry._cache_token, tolerance)
    return ("build123d", id(geometry), hash(geometry), tolerance)


def _split_mesh(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    try:
        split = list(mesh.split(only_watertight=False))
    except Exception:
        split = []
    return split or [mesh]


def _mesh_to_bvh(mesh: trimesh.Trimesh) -> object:
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    model = fcl.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(vertices, faces)
    model.endModel()
    return model


def _identity() -> Mat4:
    return np.identity(4, dtype=np.float64)


def _origin_matrix(origin: Origin) -> Mat4:
    matrix = _rpy_matrix(origin.rpy)
    matrix[:3, 3] = np.asarray(origin.xyz, dtype=np.float64)
    return matrix


def _motion_matrix(articulation: Articulation, value: float) -> Mat4:
    if articulation.articulation_type == ArticulationType.FIXED:
        return _identity()
    axis = _normalize(articulation.axis)
    if articulation.articulation_type == ArticulationType.PRISMATIC:
        matrix = _identity()
        matrix[:3, 3] = axis * value
        return matrix
    return _axis_angle_matrix(axis, value)


def _rpy_matrix(rpy: Vec3) -> Mat4:
    return np.asarray(trimesh.transformations.euler_matrix(*rpy, axes="sxyz"), dtype=np.float64)


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> Mat4:
    return np.asarray(
        trimesh.transformations.rotation_matrix(angle, axis),
        dtype=np.float64,
    )


_LOOP_CACHE_LIMIT = 512
_LOOP_TOLERANCE = 1e-6
_LOOP_STEPS = 5


def _solve_loops(
    kernel: MeshCollisionKernel,
    pose: dict[str, float],
    followers: list[Articulation],
    targets: list[tuple[Articulation, np.ndarray]],
) -> dict[str, float]:
    """Find follower values that keep every loop pin closed at this pose.

    Walks from the rest pose to the authored one in a few steps, solving at each
    and carrying the answer forward. A linkage has more than one way to be
    assembled -- a four bar can sit elbow up or elbow down -- and stepping out
    from rest keeps the model on the branch it was authored in instead of
    letting it snap through itself. Rest is exact by construction: drives solve
    to zero there and the pins were measured there.
    """

    def assemble(scale: float, values: np.ndarray) -> dict[str, float]:
        stepped = kernel._resolve_drives({name: value * scale for name, value in pose.items()})
        stepped.update(
            {item.name: float(value) for item, value in zip(followers, values, strict=True)}
        )
        return stepped

    def residual(scale: float, values: np.ndarray) -> np.ndarray:
        placed = kernel._place(assemble(scale, values))
        rows: list[float] = []
        for closure, local in targets:
            pin = _apply(placed[closure.parent] @ _origin_matrix(closure.origin), np.zeros(3))
            rows.extend(pin - _apply(placed[closure.child], local))
        return np.asarray(rows, dtype=np.float64)

    values = np.zeros(len(followers), dtype=np.float64)
    for step in range(1, _LOOP_STEPS + 1):
        scale = step / _LOOP_STEPS
        damping = 1e-6
        for _ in range(40):
            current = residual(scale, values)
            error = float(np.linalg.norm(current))
            if error < _LOOP_TOLERANCE:
                break
            jacobian = np.empty((len(current), len(values)), dtype=np.float64)
            for index in range(len(values)):
                nudged = values.copy()
                nudged[index] += 1e-6
                jacobian[:, index] = (residual(scale, nudged) - current) / 1e-6
            # Levenberg-Marquardt: an over-center linkage passes through poses
            # where the Jacobian is singular, and plain Gauss-Newton throws the
            # mechanism across the room there.
            normal = jacobian.T @ jacobian + damping * np.eye(len(values))
            delta = np.linalg.solve(normal, -jacobian.T @ current)
            trial = values + delta
            if float(np.linalg.norm(residual(scale, trial))) < error:
                values, damping = trial, max(damping * 0.5, 1e-9)
            else:
                damping *= 8.0
                if damping > 1e6:
                    break

    error = float(np.linalg.norm(residual(1.0, values)))
    if error > 1e-4:
        names = ", ".join(repr(closure.name) for closure, _ in targets)
        raise LoopClosureError(
            f"pose leaves loop closure {names} open by {error * 1000:.1f} mm; the mechanism "
            "cannot reach this pose. Tighten the driving joint's motion limits or fix the "
            "link geometry that sets the range."
        )
    return assemble(1.0, values)


def _apply(matrix: Mat4, point: np.ndarray) -> np.ndarray:
    return (matrix[:3, :3] @ point) + matrix[:3, 3]


def _angle_about(axis: np.ndarray, reference: np.ndarray, target: np.ndarray) -> float:
    """Signed rotation about ``axis`` that swings ``reference`` onto ``target``.

    Both vectors are flattened into the plane the joint actually turns in, so a
    target off the plane still gives the best in-plane answer instead of nothing.
    """

    flat_reference = reference - axis * float(np.dot(reference, axis))
    flat_target = target - axis * float(np.dot(target, axis))
    if np.linalg.norm(flat_reference) < 1e-12 or np.linalg.norm(flat_target) < 1e-12:
        return 0.0
    flat_reference = flat_reference / np.linalg.norm(flat_reference)
    flat_target = flat_target / np.linalg.norm(flat_target)
    cosine = float(np.clip(np.dot(flat_reference, flat_target), -1.0, 1.0))
    sine = float(np.dot(axis, np.cross(flat_reference, flat_target)))
    return math.atan2(sine, cosine)


def _normalize(axis: Vec3) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    length = math.hypot(*axis)
    if length <= 0.0:
        raise ValidationError("articulation axis must be non-zero")
    return vector / length


def _fcl_transform(matrix: Mat4) -> object:
    return fcl.Transform(
        np.ascontiguousarray(matrix[:3, :3], dtype=np.float64),
        np.ascontiguousarray(matrix[:3, 3], dtype=np.float64),
    )


def _points_bounds(points: np.ndarray) -> Bounds:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    return (
        (float(minimum[0]), float(minimum[1]), float(minimum[2])),
        (float(maximum[0]), float(maximum[1]), float(maximum[2])),
    )


def _bounds_overlap_depth(bounds_a: Bounds, bounds_b: Bounds) -> Vec3:
    return tuple(
        max(
            0.0,
            min(bounds_a[1][axis], bounds_b[1][axis]) - max(bounds_a[0][axis], bounds_b[0][axis]),
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def _contact_info(contact: object) -> ContactInfo:
    return ContactInfo(
        position=_array_to_vec3(getattr(contact, "pos", None)),
        normal=_array_to_vec3(getattr(contact, "normal", None)),
        depth=_optional_float(getattr(contact, "penetration_depth", None)),
    )


def _array_to_vec3(value: object) -> Vec3 | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) < 3:
        return None
    return (float(array[0]), float(array[1]), float(array[2]))


def _optional_float(value: object) -> float | None:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _collision_rank(query: CollisionQuery) -> tuple[float, float]:
    overlap = query.overlap_depth or (0.0, 0.0, 0.0)
    return (min(overlap), query.overlap_volume)


def _pair_key(part_a: str, part_b: str) -> tuple[str, str]:
    return (part_a, part_b) if part_a <= part_b else (part_b, part_a)
