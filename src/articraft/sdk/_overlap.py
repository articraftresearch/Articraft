"""When placed meshes meaningfully overlap: FCL surface contact, then the solids'
shared volume, or for open meshes a contact deeper than noise or one mesh buried in
the other. It reads only placed triangles, so it loads no geometry kernel."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import fcl
import igl
import manifold3d
import numpy as np
import trimesh

Vec3: TypeAlias = tuple[float, float, float]
Bounds: TypeAlias = tuple[Vec3, Vec3]

_NUMERICAL_CONTACT_DEPTH_TOLERANCE = 1e-6
DEFAULT_OVERLAP_TOLERANCE = 0.005
DEFAULT_OVERLAP_VOLUME_TOLERANCE = 5e-7


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
class _CollisionEntry:
    part_name: str
    shape_name: str
    obj: object
    bounds: Bounds
    world_vertices: np.ndarray
    world_mesh: trimesh.Trimesh


def mesh_overlaps(
    meshes: Mapping[str, Sequence[trimesh.Trimesh]],
    *,
    overlap_tol: float = DEFAULT_OVERLAP_TOLERANCE,
    overlap_volume_tol: float = DEFAULT_OVERLAP_VOLUME_TOLERANCE,
    max_contacts: int = 16,
) -> list[CollisionQuery]:
    """Meaningful overlaps between meshes already placed in the world, by the rule
    ``fail_if_parts_overlap_in_current_pose`` applies to an assembly's parts.

    ``meshes`` maps a part name to its meshes. Meshes of one part never count
    against each other; each query names its mesh by its index in the part.
    """

    entries = [
        _CollisionEntry(
            part_name=part,
            shape_name=str(index),
            obj=fcl.CollisionObject(_mesh_to_bvh(mesh), fcl.Transform()),
            bounds=_points_bounds(np.asarray(mesh.vertices, dtype=np.float64)),
            world_vertices=np.asarray(mesh.vertices, dtype=np.float64),
            world_mesh=mesh,
        )
        for part, part_meshes in meshes.items()
        for index, mesh in enumerate(part_meshes)
    ]
    return _meaningful_overlaps(
        entries,
        overlap_tol=overlap_tol,
        overlap_volume_tol=overlap_volume_tol,
        max_contacts=max_contacts,
    )


def _meaningful_overlaps(
    entries: Sequence[_CollisionEntry],
    *,
    overlap_tol: float,
    overlap_volume_tol: float,
    max_contacts: int,
) -> list[CollisionQuery]:
    """Pairs of different parts whose bounds overlap past ``overlap_tol`` on every
    axis and whose solids share more than ``overlap_volume_tol``; for open
    meshes, a contact deeper than numerical noise or one buried in the other."""

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
        if solid_intersection is not None:
            collided = solid_intersection > 0.0
        else:
            collided = _open_mesh_buried(entry_a.world_mesh, entry_b.world_mesh)
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


def _solid_intersection_volume(
    entry_a: _CollisionEntry,
    entry_b: _CollisionEntry,
) -> float | None:
    return _mesh_intersection_volume(entry_a.world_mesh, entry_b.world_mesh)


def _open_mesh_buried(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> bool:
    """Whether a mesh hides entirely inside a watertight one.

    Surface collision misses full containment, and the boolean volume needs
    two watertight operands. One watertight side is still decisive: with no
    surface crossing, the other mesh is either wholly inside it or wholly
    outside, so a few of its vertices answer for all of them. Two open
    meshes stay undecidable and report no collision.
    """

    for container, contents in ((mesh_a, mesh_b), (mesh_b, mesh_a)):
        if not container.is_watertight:
            continue
        vertices = np.asarray(contents.vertices, dtype=np.float64)
        probes = vertices[:: max(1, len(vertices) // 8)][:8]
        # Winding numbers are immune to the edge-grazing rays that break
        # crossing-parity tests; the majority vote stays as cheap insurance
        # against probes that land exactly on the container's surface.
        winding = igl.fast_winding_number(
            np.ascontiguousarray(container.vertices, dtype=np.float64),
            np.ascontiguousarray(container.faces, dtype=np.int64),
            np.ascontiguousarray(probes),
        )
        inside = np.asarray(winding) > 0.5
        if len(inside) and int(np.count_nonzero(inside)) * 2 > len(inside):
            return True
    return False


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


def _mesh_to_bvh(mesh: trimesh.Trimesh) -> object:
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    model = fcl.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(vertices, faces)
    model.endModel()
    return model


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
