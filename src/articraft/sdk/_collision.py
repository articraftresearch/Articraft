from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import fcl
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from articraft.sdk._mesh.core import MeshGeometry, geometry_to_trimesh
from articraft.sdk._overlap import (
    Bounds,
    CollisionQuery,
    ContactInfo,
    Vec3,
    _array_to_vec3,
    _collide_entries,
    _CollisionEntry,
    _meaningful_overlaps,
    _mesh_intersection_volume,
    _mesh_to_bvh,
    _points_bounds,
)
from articraft.sdk.assembly import PhysicsState, RigidBodyAssembly
from articraft.sdk.bodies import Geometry
from articraft.sdk.errors import ValidationError

Mat4: TypeAlias = np.ndarray
Pose: TypeAlias = Mapping[str, float] | PhysicsState


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
class _GeometryComponent:
    name: str
    obj: object
    mesh: trimesh.Trimesh


class MeshCollisionKernel:
    def __init__(self, model: RigidBodyAssembly, *, mesh_tolerance: float) -> None:
        if mesh_tolerance <= 0.0 or not math.isfinite(mesh_tolerance):
            raise ValidationError("mesh_tolerance must be a positive finite number")
        self.model = model
        self.resolved = model.resolve()
        self.mesh_tolerance = float(mesh_tolerance)
        self._mesh_cache: dict[tuple[object, ...], _LocalMesh] = {}

    def part_world_position(self, part_name: str, pose: Pose) -> Vec3:
        transform = self.world_transforms(pose).get(part_name)
        if transform is None:
            raise ValidationError(f"unknown part: {part_name!r}")
        return _array_to_vec3(transform[:3, 3]) or (0.0, 0.0, 0.0)

    def part_world_vertices(self, part_name: str, pose: Pose) -> np.ndarray:
        entries = self._selector_entries(part_name, None, pose)
        vertices = [entry.world_vertices for entry in entries]
        if not vertices:
            raise ValidationError(f"part {part_name!r} has no geometry")
        # Bounds vertices are sufficient for all projection and bounds checks.
        return np.concatenate(vertices, axis=0)

    def part_world_bounds(self, part_name: str, pose: Pose) -> Bounds:
        return _points_bounds(self.part_world_vertices(part_name, pose))

    def shape_world_bounds(
        self,
        part_name: str,
        shape_name: str,
        pose: Pose,
    ) -> Bounds:
        return self._selector_entries(part_name, shape_name, pose)[0].bounds

    def world_transforms(self, pose: Pose) -> dict[str, Mat4]:
        """Where every body sits, for a pose keyed by joint or by DOF."""

        if isinstance(pose, PhysicsState):
            return self.resolved.world_transforms(pose)
        state = self.resolved.forward_kinematics(self._dof_positions(pose))
        return self.resolved.world_transforms(state)

    def _dof_positions(self, pose: Mapping[str, float]) -> dict[str, float]:
        """Accept ``{"hinge": 0.5}`` as well as ``{"hinge.rotZ": 0.5}``.

        A joint with one degree of freedom is named unambiguously by the joint
        alone, which is how every pose was written before joints could carry
        six. The qualified form is only needed where it genuinely is.
        """

        single = {
            item.joint.name: item.joint.dof_id(item.joint.dofs[0])
            for item in self.resolved.joints
            if len(item.joint.dofs) == 1
        }
        return {single.get(key, key): float(value) for key, value in pose.items()}

    def collision_between(
        self,
        part_a: str,
        part_b: str,
        pose: Pose,
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
        pose: Pose,
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
        pose: Pose,
        *,
        max_contacts: int = 16,
    ) -> list[DistanceQuery]:
        parts = [body.name for body in self.resolved.rigid_bodies]
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
        pose: Pose,
        *,
        overlap_tol: float,
        overlap_volume_tol: float,
        max_contacts: int = 16,
    ) -> list[CollisionQuery]:
        entries = self._all_entries(pose)
        return _meaningful_overlaps(
            entries,
            overlap_tol=overlap_tol,
            overlap_volume_tol=overlap_volume_tol,
            max_contacts=max_contacts,
        )

    def disconnected_geometry_islands(
        self,
        *,
        contact_tol: float,
    ) -> list[GeometryConnectivityFinding]:
        findings: list[GeometryConnectivityFinding] = []
        for part in self.resolved.rigid_bodies:
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

    def _all_entries(self, pose: Pose) -> list[_CollisionEntry]:
        transforms = self.world_transforms(pose)
        return [
            self._entry(part.name, shape.name, shape.geometry, transforms[part.name])
            for part in self.resolved.rigid_bodies
            for shape in part._iter_shapes()
        ]

    def _selector_entries(
        self,
        part_name: str,
        shape_name: str | None,
        pose: Pose,
    ) -> list[_CollisionEntry]:
        part = self.resolved.get_rigid_body(part_name)
        transform = self.world_transforms(pose).get(part.name)
        if transform is None:
            raise ValidationError(f"unknown part: {part.name!r}")
        if shape_name is not None:
            shape_name = str(shape_name).strip()
            geometry = part.shape(shape_name)
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
        pose: Pose,
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
        part = self.resolved.get_rigid_body(part_name)
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
    touching: list[tuple[int, int]] = []
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
                touching.append((index, other_index))
            _remember_nearest(nearest, index, other_index, distance)
            _remember_nearest(nearest, other_index, index, distance)

    count = len(components)
    graph = sparse.coo_matrix(
        (np.ones(len(touching)), tuple(np.asarray(touching, dtype=np.int64).reshape(-1, 2).T)),
        shape=(count, count),
    )
    group_count, labels = connected_components(graph, directed=False)
    groups = [set(map(int, np.flatnonzero(labels == label))) for label in range(group_count)]
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


def _meshes_have_overlapping_bounds(a: trimesh.Trimesh, b: trimesh.Trimesh) -> bool:
    overlap = np.minimum(a.bounds[1], b.bounds[1]) - np.maximum(a.bounds[0], b.bounds[0])
    return bool(np.all(overlap > 0.0))


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


def _fcl_transform(matrix: Mat4) -> object:
    return fcl.Transform(
        np.ascontiguousarray(matrix[:3, :3], dtype=np.float64),
        np.ascontiguousarray(matrix[:3, 3], dtype=np.float64),
    )


def _collision_rank(query: CollisionQuery) -> tuple[float, float]:
    overlap = query.overlap_depth or (0.0, 0.0, 0.0)
    return (min(overlap), query.overlap_volume)


def _pair_key(part_a: str, part_b: str) -> tuple[str, str]:
    return (part_a, part_b) if part_a <= part_b else (part_b, part_a)
